"""
Dataset verifier for hedgementation patch datasets.

Performs per-ID consistency checks (file presence, geometry alignment, shape,
cross-file agreement) for the X / X_cloud / y / y_id / rpg_unbuffered patches
exported by the construct_dataset pipeline.

The per-ID work is dominated by rasterio file reads, which release the GIL during
I/O. `verify_ids_async` runs each ID's checks in a thread via `asyncio.to_thread`
and dispatches them through the project's `WorkerQueue`, so many IDs are
verified concurrently. The sync `verify_ids` keeps the existing call site
working from both scripts and Jupyter.
"""
import asyncio
import logging
import os
import pathlib
import re
import time
from collections import Counter
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
import shapely

from hedgementation_utils.io.io_manager import HedgementationIOManager

from src.worker_queue import WorkerQueue


logger = logging.getLogger(__name__)


def _ensure_logger_visible(target_logger: logging.Logger,
                           level: int = logging.INFO) -> None:
    """Attach a stdout StreamHandler to `target_logger` if no logging
    configuration exists yet. Safe to call repeatedly."""
    has_handler = bool(target_logger.handlers) or bool(logging.root.handlers)
    if not has_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        ))
        target_logger.addHandler(handler)
    if target_logger.level == logging.NOTSET or target_logger.level > level:
        target_logger.setLevel(level)


# Patterns for grouping defect strings into categories at summary time.
# Each entry is (regex, label_template). The first match wins; the template
# is formatted with the regex's capture groups (e.g. the offending key/folder).
_DEFECT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^Corresponding file not found in (\S+)"), "Missing file in {0}/"),
    (re.compile(r"^(\S+) geometry does not match metadata geometry"), "Geometry mismatch ({0})"),
    (re.compile(r"^Unable to reshape (\S+) data to"), "Reshape failed ({0})"),
    (re.compile(r"^Count of images and dates differ for (\S+)"), "Image/date count mismatch ({0})"),
    (re.compile(r"^X and X_cloud have different dates counts"), "X vs X_cloud date count mismatch"),
    (re.compile(r"^Dates for X and X cloud differ"), "X vs X_cloud dates differ"),
    (re.compile(r"^(\S+) shape differs from expected"), "Shape mismatch ({0})"),
    (re.compile(r"^y and y_id are not in agreement"), "y vs y_id pixel disagreement"),
]


def categorize_defect(defect: str) -> str:
    for pattern, template in _DEFECT_PATTERNS:
        m = pattern.match(defect)
        if m:
            return template.format(*m.groups())
    return f"Other: {defect[:60]}"


def _is_suppressed(defect: str, suppress: list[str]) -> bool:
    category = categorize_defect(defect)
    return any(s.lower() in category.lower() for s in suppress)


class DatasetVerifier:
    def __init__(self,
                 dataset_root: str,
                 manager: HedgementationIOManager,
                 metadata: gpd.GeoDataFrame = None,
                 img_height: int = 128,
                 img_width: int = 128,
                 X_bands: int = 10,
                 X_cloud_bands: int = 2,
                 geom_tolerance: float = 1e-4,
                 max_concurrent: int = 16,
                 X_folder: str = "X",
                 X_cloud_folder: str = "X_cloud",
                 y_folder: str = "y",
                 y_id_folder: str = "y_id",
                 rpg_unbuffered_folder: str = "rpg_unbuffered"):

        self.dataset_root = dataset_root
        self.metadata = metadata if metadata is not None else gpd.read_file(
            pathlib.Path(self.dataset_root, "metadata.geojson")
        )
        self.manager = manager

        self.img_height = img_height
        self.img_width = img_width
        self.X_bands = X_bands
        self.X_cloud_bands = X_cloud_bands
        self.geom_tolerance = geom_tolerance
        self.max_concurrent = max_concurrent

        # Single source of truth for each data type's folder, filename pattern, and band count.
        self.data_types = {
            "X":              {"folder": X_folder,              "pattern": "X_{}.tif",       "bands": X_bands},
            "X_cloud":        {"folder": X_cloud_folder,        "pattern": "X_cloud_{}.tif", "bands": X_cloud_bands},
            "y":              {"folder": y_folder,              "pattern": "y_{}.tif"},
            "y_id":           {"folder": y_id_folder,           "pattern": "y_id_{}.tif"},
            "rpg_unbuffered": {"folder": rpg_unbuffered_folder, "pattern": "rpg_{}.npy"},
        }

    # ---------- geometry helpers ----------
    def _poly_to_str(self, poly: shapely.geometry.polygon):
        return ", ".join([f"{d:.4f}" for d in poly.bounds])

    def _bounds_to_geom(self, bounds: rasterio.coords.BoundingBox):
        return shapely.geometry.box(bounds.left, bounds.bottom, bounds.right, bounds.top)

    def _geoms_equal(self, geom1: shapely.Geometry, geom2: shapely.Geometry):
        return geom1.equals_exact(geom2, tolerance=self.geom_tolerance, normalize=True)

    def _check_geometry(self, id: int, src, name: str):
        """Compare a raster's bounds against the metadata geometry for `id`."""
        row = self.metadata[self.metadata["ID_PATCH"] == id]
        src_geom = self._bounds_to_geom(src.bounds)
        metadata_geom = row.to_crs(str(src.crs)).iloc[0].geometry
        if not self._geoms_equal(src_geom, metadata_geom):
            return [
                f"{name} geometry does not match metadata geometry"
                f"({self._poly_to_str(src_geom)} for {name} vs. {self._poly_to_str(metadata_geom)} for metadata)"
            ]
        return []

    # ---------- file presence ----------
    def _file_path(self, key: str, id: int) -> pathlib.Path:
        cfg = self.data_types[key]
        return pathlib.Path(self.dataset_root, cfg["folder"], cfg["pattern"].format(id))

    def _files_present(self, id: int) -> dict:
        return {k: os.path.exists(self._file_path(k, id)) for k in self.data_types}

    def verify_files(self, id: int):
        return [
            f"Corresponding file not found in {self.data_types[k]['folder']}"
            for k, present in self._files_present(id).items() if not present
        ]

    # ---------- raster loading ----------
    def _open_raster(self, key: str, id: int):
        loaders = {
            "X":       self.manager.load_X,
            "X_cloud": self.manager.load_X_cloud,
            "y":       self.manager.load_y,
            "y_id":    self.manager.load_y_id,
        }
        return loaders[key](identifier=id, file_type="tif", return_as_dataset=True)

    # ---------- X / X_cloud ----------
    def _verify_X_like(self, id: int, key: str):
        """Open one X-style raster: check geometry, reshape, validate timesteps vs. dates.
        Returns (dates_list_or_None, defects)."""
        defects = []
        bands = self.data_types[key]["bands"]
        with self._open_raster(key, id) as src:
            defects.extend(self._check_geometry(id, src, key))
            data = src.read()
            dates = [s.split("_")[0] for s in src.descriptions[::bands]]

        target_shape = (-1, bands, self.img_height, self.img_width)
        try:
            data = data.reshape(target_shape)
            timesteps = data.shape[0]
            if timesteps != len(dates):
                defects.append(
                    f"Count of images and dates differ for {key} "
                    f"({timesteps} images vs. {len(dates)} dates)"
                )
        except Exception:
            defects.append(f"Unable to reshape {key} data to {target_shape}")

        return defects

    def verify_X(self, id: int, present: Optional[dict] = None,
                 skip: Optional[set[str]] = None):
        """
        Verifies X (and X_cloud) for whichever of the two files exist:
        1. Geometry of each raster matches metadata
        2. Each raster reshapes to its expected (T, bands, H, W) shape
        3. Timestep count matches date count for each
        4. If both are present, their date lists agree
        """
        present = present if present is not None else self._files_present(id)
        skip = skip or set()
        defects = []

        for key in ("X", "X_cloud"):
            if present.get(key) and key not in skip:
                d = self._verify_X_like(id, key)
                defects.extend(d)

        return defects

    # ---------- y / y_id ----------
    def _verify_y_like(self, id: int, key: str):
        """Open one y-style raster: check geometry and shape. Returns (array_or_None, defects)."""
        defects = []
        with self._open_raster(key, id) as src:
            defects.extend(self._check_geometry(id, src, key))
            arr = np.squeeze(src.read())

        target_shape = (self.img_height, self.img_width)
        if arr.shape != target_shape:
            defects.append(
                f"{key} shape differs from expected(Expected {target_shape}, actual {arr.shape})"
            )
        return arr, defects

    def verify_Y(self, id: int, present: Optional[dict] = None,
                 skip: Optional[set[str]] = None):
        """
        Verifies y (and y_id) for whichever of the two files exist:
        1. Geometry of each raster matches metadata
        2. Each raster has the expected (H, W) shape
        3. If both are present, y_id matches y when masked to [0,1]
        """
        present = present if present is not None else self._files_present(id)
        skip = skip or set()
        defects = []
        arrays = {}

        for key in ("y", "y_id"):
            if present.get(key) and key not in skip:
                arr, d = self._verify_y_like(id, key)
                defects.extend(d)
                arrays[key] = arr

        y_np, y_id_np = arrays.get("y"), arrays.get("y_id")
        if y_np is not None and y_id_np is not None and y_np.shape == y_id_np.shape:
            if not np.array_equal(y_np > 0, y_id_np != -1):
                defects.append("y and y_id are not in agreement on hedgerow vs. background pixels")

        return defects

    # ---------- top-level orchestration ----------
    def verify_id(self, id: int, skip: Optional[set[str]] = None):
        skip = skip or set()
        present = {k: v for k, v in self._files_present(id).items() if k not in skip}
        file_defects = [
            f"Corresponding file not found in {self.data_types[k]['folder']}"
            for k, p in present.items() if not p
        ]
        return file_defects + self.verify_X(id, present, skip=skip) + self.verify_Y(id, present, skip=skip)

    async def _verify_id_async(self, id: int, skip: Optional[set[str]] = None):
        try:
            defects = await asyncio.to_thread(self.verify_id, id, skip)
        except Exception as e:
            defects = [str(e)]
        return id, defects

    def _resolve_ids(self,
                    ids: Optional[list[int]],
                    start: Optional[int],
                    limit: Optional[int]) -> list[int]:
        if ids:
            return list(ids)
        start = start or 0
        end = (start + limit) if limit is not None else None
        return list(self.metadata["ID_PATCH"][start:end])

    async def verify_ids_async(self,
                               ids: Optional[list[int]] = None,
                               start: Optional[int] = None,
                               limit: Optional[int] = None,
                               worker_logger: Optional[logging.Logger] = None,
                               print_summary: bool = True,
                               progress_interval: float = 2.0,
                               log_level: int = logging.INFO,
                               suppress: Optional[list[str]] = None,
                               verbose: bool = False,
                               skip: Optional[list[str]] = None):
        ids = self._resolve_ids(ids, start, limit)
        skip_set: set[str] = set(skip) if skip else set()
        total = len(ids)
        _ensure_logger_visible(logger, level=log_level)

        # WorkerQueue.wait_all logs on every completion (see worker_queue.py:64),
        # which is thousands of lines for a full run. Suppress its info-level
        # stats but keep task-failure errors flowing.
        wq_logger = worker_logger or logger.getChild("workerqueue")
        if worker_logger is None:
            wq_logger.setLevel(logging.WARNING)

        logger.info(
            f"Starting verification of {total} ID(s) "
            f"(max_concurrent={self.max_concurrent})"
        )
        wq = WorkerQueue(logger=wq_logger, max_concurrent=self.max_concurrent)
        t0 = time.perf_counter()

        async def _report_progress():
            try:
                while True:
                    await asyncio.sleep(progress_interval)
                    done = len(wq.completed) + len(wq.failed)
                    elapsed = max(time.perf_counter() - t0, 1e-9)
                    rate = done / elapsed
                    eta = (total - done) / rate if rate > 0 else float("inf")
                    eta_str = f"{eta:.0f}s" if eta != float("inf") else "?"
                    pct = (100.0 * done / total) if total else 100.0
                    logger.info(
                        f"Progress: {done}/{total} ({pct:.1f}%) | "
                        f"running={len(wq.running)} waiting={len(wq.waiting)} "
                        f"failed={len(wq.failed)} | "
                        f"{rate:.1f} IDs/s, ETA {eta_str}"
                    )
            except asyncio.CancelledError:
                pass

        monitor = asyncio.create_task(_report_progress())
        try:
            for id in ids:
                await wq.add_task(self._verify_id_async(int(id), skip=skip_set))
            await wq.wait_all()
        finally:
            monitor.cancel()
            try:
                await monitor
            except asyncio.CancelledError:
                pass

        elapsed = time.perf_counter() - t0

        order = {id: i for i, id in enumerate(ids)}
        defective_ids: list[int] = []
        defects_by_id: dict[int, list[str]] = {}
        for id, defects in wq.completed:
            if defects:
                defective_ids.append(id)
                defects_by_id[id] = defects
        defective_ids.sort(key=lambda x: order.get(x, 0))

        if suppress:
            defects_by_id = {
                id: [d for d in defects if not _is_suppressed(d, suppress)]
                for id, defects in defects_by_id.items()
            }
            defects_by_id = {id: d for id, d in defects_by_id.items() if d}
            defective_ids = [id for id in defective_ids if id in defects_by_id]

        rate = total / elapsed if elapsed > 0 else 0.0
        logger.info(
            f"Verification complete in {elapsed:.1f}s "
            f"({rate:.1f} IDs/s) | with defects: {len(defective_ids)}, "
            f"clean: {total - len(defective_ids)}, "
            f"task failures: {len(wq.failed)}"
        )
        print(f"IDS with defects: {defective_ids}")
        if verbose and defects_by_id:
            print()
            print("Defects per ID:")
            for id in defective_ids:
                print(f"  ID {id}:")
                for d in defects_by_id[id]:
                    print(f"    - {d}")
        if print_summary:
            self.summarize_defects(defects_by_id, total_ids=total)

        return defective_ids, defects_by_id

    def summarize_defects(self,
                          defects_by_id: dict[int, list[str]],
                          total_ids: Optional[int] = None,
                          top_n_ids: int = 5) -> Counter:
        """Print a summary of the defects produced by a verification run and
        return a `Counter` of category labels (descending). Safe to call
        independently with any `defects_by_id` mapping."""
        n_defective = len(defects_by_id)
        total_defects = sum(len(d) for d in defects_by_id.values())

        category_counts: Counter = Counter()
        for defects in defects_by_id.values():
            for d in defects:
                category_counts[categorize_defect(d)] += 1

        sep = "=" * 60
        print()
        print(sep)
        print("Verification summary")
        print(sep)
        if total_ids is not None:
            n_clean = total_ids - n_defective
            pct_clean = (100 * n_clean / total_ids) if total_ids else 0.0
            pct_defective = (100 * n_defective / total_ids) if total_ids else 0.0
            print(f"  Total IDs verified : {total_ids}")
            print(f"  Clean              : {n_clean} ({pct_clean:.2f}%)")
            print(f"  With defects       : {n_defective} ({pct_defective:.2f}%)")
        else:
            print(f"  IDs with defects   : {n_defective}")
        print(f"  Total defects      : {total_defects}")

        if not category_counts:
            print()
            print("No defects found.")
            return category_counts

        print()
        print("Defects by category (descending):")
        label_width = max(len(label) for label in category_counts)
        for label, count in category_counts.most_common():
            print(f"  {label:<{label_width}}  {count}")

        if top_n_ids and defects_by_id:
            print()
            print(f"Top {min(top_n_ids, len(defects_by_id))} IDs by defect count:")
            worst = sorted(defects_by_id.items(), key=lambda kv: -len(kv[1]))[:top_n_ids]
            for id, defects in worst:
                print(f"  ID {id}: {len(defects)} defect(s)")

        return category_counts

    def verify_ids(self,
                   ids: Optional[list[int]] = None,
                   start: Optional[int] = None,
                   limit: Optional[int] = None,
                   suppress: Optional[list[str]] = None,
                   verbose: bool = False,
                   skip: Optional[list[str]] = None):
        return asyncio.run(
            self.verify_ids_async(ids=ids, start=start, limit=limit,
                                  suppress=suppress, verbose=verbose, skip=skip)
        )


if __name__ == "__main__":
    import argparse
    from hedgementation_utils.io.io_manager import HedgementationIOManager

    parser = argparse.ArgumentParser(
        description="Verify the integrity of a hedgementation patch dataset."
    )
    parser.add_argument("dataset_root", help="Path to the dataset root directory.")
    parser.add_argument(
        "--ids", nargs="+", type=int, default=None, metavar="ID",
        help="Specific patch IDs to verify. Mutually exclusive with --start/--limit.",
    )
    parser.add_argument("--start", type=int, default=None, help="Start index into metadata.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of IDs to verify.")
    parser.add_argument("--img-height", type=int, default=128)
    parser.add_argument("--img-width", type=int, default=128)
    parser.add_argument("--X-bands", type=int, default=10)
    parser.add_argument("--X-cloud-bands", type=int, default=2)
    parser.add_argument("--geom-tolerance", type=float, default=1e-4)
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--X-folder", default="X", help="Subdirectory name for X patches (default: X).")
    parser.add_argument("--X-cloud-folder", default="X_cloud", help="Subdirectory name for X_cloud patches (default: X_cloud).")
    parser.add_argument("--y-folder", default="y", help="Subdirectory name for y patches (default: y).")
    parser.add_argument("--y-id-folder", default="y_id", help="Subdirectory name for y_id patches (default: y_id).")
    parser.add_argument("--rpg-unbuffered-folder", default="rpg_unbuffered", help="Subdirectory name for rpg_unbuffered patches (default: rpg_unbuffered).")
    parser.add_argument(
        "--suppress", nargs="+", default=None, metavar="CATEGORY",
        help=(
            "Defect categories to ignore for this run (case-insensitive substring "
            "match against the category label). Suppressed defects are excluded from "
            "all counts, the defective-ID list, and the summary. "
            "Possible categories (use any substring to match):\n"
            "  'Missing file'                     – file absent in X/, X_cloud/, y/, y_id/, or rpg_unbuffered/\n"
            "  'Geometry mismatch'                – raster bounds don't match metadata geometry (X, X_cloud, y, y_id)\n"
            "  'Reshape failed'                   – X or X_cloud data could not be reshaped to (T, bands, H, W)\n"
            "  'Image/date count mismatch'        – timestep count != date count for X or X_cloud\n"
            "  'X vs X_cloud date count mismatch' – X and X_cloud have different numbers of dates\n"
            "  'X vs X_cloud dates differ'        – X and X_cloud date lists don't agree\n"
            "  'Shape mismatch'                   – y or y_id array is not (H, W)\n"
            "  'y vs y_id pixel disagreement'     – y and y_id disagree on hedgerow vs. background pixels\n"
            "  'Other'                            – any defect that doesn't match the above patterns\n"
            "Example: --suppress 'Missing file' 'Geometry mismatch'"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each defective ID with the full text of all its defects.",
    )
    parser.add_argument(
        "--skip", nargs="+", default=None, metavar="DATA_TYPE",
        help=(
            "Data types to skip entirely (no file-presence or content checks). "
            "Valid values: X, X_cloud, y, y_id, rpg_unbuffered. "
            "Example: --skip rpg_unbuffered y_id"
        ),
    )

    args = parser.parse_args()

    _ensure_logger_visible(logger)
    manager = HedgementationIOManager(dataset_root=args.dataset_root)
    verifier = DatasetVerifier(
        dataset_root=args.dataset_root,
        manager=manager,
        img_height=args.img_height,
        img_width=args.img_width,
        X_bands=args.X_bands,
        X_cloud_bands=args.X_cloud_bands,
        geom_tolerance=args.geom_tolerance,
        max_concurrent=args.max_concurrent,
        X_folder=args.X_folder,
        X_cloud_folder=args.X_cloud_folder,
        y_folder=args.y_folder,
        y_id_folder=args.y_id_folder,
        rpg_unbuffered_folder=args.rpg_unbuffered_folder,
    )
    verifier.verify_ids(ids=args.ids, start=args.start, limit=args.limit,
                        suppress=args.suppress, verbose=args.verbose, skip=args.skip)
