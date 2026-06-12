"""
Supertile-orchestrated dataset construction.

The AOI is divided into a grid of supertiles. For each supertile, this
script extracts patches+hedgerows from the source GPKG and runs the
configured per-stage processors (folds, tile group, class). Each stage's
per-supertile output is written as its own parquet file, so a crashed or
interrupted run can be resumed by re-invoking the same command — already
written outputs are skipped. An optional final step concatenates everything
into a single dataset file.

Layout under --output_dir:
  manifest/
    aoi.json
    supertile_grid.parquet    # supertile polygons (supertile_id -> bounds)
    cell_grid.parquet         # global fold cells + assigned fold
    tile_grid.parquet         # global tile polygons
  patches/
    patches_NNNNN.parquet     # patch_i, patch_j, geometry
  hedgerows/
    hedgerows_NNNNN.parquet   # patch_i, patch_j, hedgerows (WKT)
  folds/
    folds_NNNNN.parquet       # patch_i, patch_j, fold
  tiles_assignment/
    tiles_NNNNN.parquet       # patch_i, patch_j, tile
  class/
    class_NNNNN.parquet       # patch_i, patch_j, <class columns>
  full_dataset.parquet        # produced when --final_output is given
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from dotenv import load_dotenv
from shapely.geometry import box
from tqdm import tqdm

from src.construct_dataset.assign_patch_folds import create_cells_with_gaps
from src.construct_dataset.assign_tile_and_tilegroup import create_tiles
from src.construct_dataset.extract_patches_supertiled import (
    WORK_CRS,
    _rows_cols_for_supertile_size,
    process_supertile_patches_only as process_supertile_patches,
    process_supertile_hedgerows,
)
from src.construct_dataset.patch_metadata_calculation import generate_class_gdf
from src.construct_dataset.utils import categorize_using_bounds

load_dotenv()

STAGE_DIRS = {
    "patches": "patches",
    "hedgerows": "hedgerows",
    "folds": "folds",
    "tiles": "tiles_assignment",
    "class": "class",
}
ALL_TARGETS = ("hedgerows", "folds", "tiles", "class")

PATCH_SIZE_M = int(os.getenv("PATCH_SIZE_M", 1280))
PATCHES_PER_CELL = int(os.getenv("PATCHES_PER_CELL", 12))
PATCHES_BETWEEN_CELLS = int(os.getenv("PATCHES_BETWEEN_CELLS", 3))
N_FOLDS = int(os.getenv("N_FOLDS", 5))
TILES_N_ROWS = int(os.getenv("TILES_N_ROWS", 6))
TILES_N_COLS = int(os.getenv("TILES_N_COLS", 6))
TILES_EPSILON = int(os.getenv("TILES_EPSILON", 1))
TILE_BUFFER_PCT = float(os.getenv("TILE_BUFFER_PCT", 0.1))

AEZ_ROOT = os.getenv("AEZ_ROOT", "data/aez")
INITIAL_BDA_PATH = os.getenv("INITIAL_BDA_PATH")
DEFAULT_OUTPUT_DIR = os.getenv("DATASET_OUTPUT_DIR", "test_data/dataset_supertiled")
LEGACY_MANIFESTS_DIR = os.getenv("LEGACY_MANIFESTS_DIR")

RANDOM_SEED = os.getenv("RANDOM_SEED", 15)

# ============================================================
# Global manifests
# ============================================================

def _get_aoi(gpkg_path: str, aoi_bounds_override: tuple | None = None):
    info = pyogrio.read_info(gpkg_path)
    src_crs = info["crs"]
    src_bounds = info["total_bounds"]

    if aoi_bounds_override is not None:
        aoi_gdf = gpd.GeoDataFrame(geometry=[box(*aoi_bounds_override)], crs=WORK_CRS)
    else:
        # Densify the bounding polygon before reprojecting. For conic projections
        # (e.g. Lambert 93 / EPSG:2154), converting only the 4 bbox corners to
        # EPSG:3857 underestimates ymax by ~42 km for France: the northernmost
        # EPSG:3857 latitude is at the central meridian, not at the corners.
        # Segmentizing at 1/100th of the bbox edge adds enough intermediate points
        # that the resulting hull correctly captures the true northern extent.
        edge_len = max(src_bounds[2] - src_bounds[0], src_bounds[3] - src_bounds[1])
        bbox_dense = shapely.segmentize(box(*src_bounds), max_segment_length=edge_len / 100)
        aoi_gdf = gpd.GeoDataFrame(geometry=[bbox_dense], crs=src_crs).to_crs(WORK_CRS)

    return src_crs, src_bounds, aoi_gdf


def _build_or_load_supertile_grid(
    manifest_dir: Path, aoi_gdf, supertile_size, rows, cols, logger
) -> gpd.GeoDataFrame:
    path = manifest_dir / "supertile_grid.parquet"
    if path.exists():
        logger.info("Loading supertile grid from %s", path)
        return gpd.read_parquet(path)

    aoi_bounds = aoi_gdf.total_bounds
    if rows is None or cols is None:
        rows, cols = _rows_cols_for_supertile_size(aoi_bounds, supertile_size)
    polys = create_tiles(aoi_gdf, rows=rows, cols=cols, epsilon=0, buffer_pct=0)
    grid = gpd.GeoDataFrame(
        {"supertile_id": np.arange(len(polys))},
        geometry=list(polys),
        crs=WORK_CRS,
    )
    grid.to_parquet(path)
    logger.info("Built %d supertiles (%dx%d), saved to %s", len(grid), rows, cols, path)
    return grid


def _build_or_load_cell_fold_grid(
    manifest_dir: Path,
    aoi_gdf,
    patch_size,
    patches_per_cell,
    patches_between_cells,
    n_folds,
    seed,
    logger,
) -> gpd.GeoDataFrame:
    path = manifest_dir / "cell_grid.parquet"
    if path.exists():
        logger.info("Loading cell+fold grid from %s", path)
        return gpd.read_parquet(path)

    cells_gdf = create_cells_with_gaps(
        aoi_gdf,
        patch_size=patch_size,
        patches_per_cell=patches_per_cell,
        patches_between_cells=patches_between_cells,
    )
    rng = random.Random(seed)
    cells_gdf["fold"] = [rng.randint(0, n_folds - 1) for _ in range(len(cells_gdf))]
    cells_gdf.to_parquet(path)
    logger.info("Built %d cells (%d folds, seed=%d), saved to %s", len(cells_gdf), n_folds, seed, path)
    return cells_gdf


def _build_or_load_tile_grid(
    manifest_dir: Path,
    aoi_gdf,
    tile_rows,
    tile_cols,
    tile_epsilon,
    tile_buffer_pct,
    logger,
) -> gpd.GeoDataFrame:
    path = manifest_dir / "tile_grid.parquet"
    if path.exists():
        logger.info("Loading tile grid from %s", path)
        return gpd.read_parquet(path)

    polys = create_tiles(
        aoi_gdf,
        rows=tile_rows,
        cols=tile_cols,
        epsilon=tile_epsilon,
        buffer_pct=tile_buffer_pct,
    )
    tile_gdf = gpd.GeoDataFrame(
        {"tile": np.arange(len(polys))},
        geometry=list(polys),
        crs=WORK_CRS,
    )
    tile_gdf.to_parquet(path)
    logger.info("Built %d tiles (%dx%d, buffer_pct=%s), saved to %s",
                len(tile_gdf), tile_rows, tile_cols, tile_buffer_pct, path)
    return tile_gdf


def _stage_legacy_manifests(
    legacy_dir: Path, manifest_dir: Path, targets, log
) -> None:
    """
    Copy pre-built (legacy-labeled) grid manifests into the run's manifest
    directory so the _build_or_load_* helpers pick them up instead of
    generating fresh grids. See build_legacy_manifests.py for how these are
    produced. Grids already present in the manifest dir (a resumed run) are
    kept as-is.
    """
    needed = []
    if "folds" in targets:
        needed.append("cell_grid.parquet")
    if "tiles" in targets:
        needed.append("tile_grid.parquet")

    for name in needed:
        src = legacy_dir / name
        dst = manifest_dir / name
        if not src.exists():
            raise FileNotFoundError(
                f"--legacy_manifests_dir given but {src} does not exist. "
                "Run src/construct_dataset/build_legacy_manifests.py first."
            )
        if dst.exists():
            log.warning(
                "%s already exists; keeping it (resumed run). Delete the "
                "manifest directory to force the legacy grids.", dst
            )
        else:
            shutil.copy(src, dst)
            log.info("Using legacy %s from %s", name, src)


# ============================================================
# Per-stage processors (one supertile each)
# ============================================================


def _stage_folds(patches_path: Path, cell_grid: gpd.GeoDataFrame,
                 drop_unassigned: bool, output_path: Path) -> Optional[Path]:
    if output_path.exists():
        return output_path
    patches = gpd.read_parquet(patches_path)
    if patches.empty:
        return None

    fold_unions = {
        int(f): shapely.unary_union(
            cell_grid[cell_grid["fold"] == f].geometry.values
        )
        for f in cell_grid["fold"].unique()
    }
    assignments = categorize_using_bounds(fold_unions, patches, crs=WORK_CRS)

    out = patches[["patch_i", "patch_j"]].copy()
    out["fold"] = assignments
    if drop_unassigned:
        out = out[out["fold"] != -1]
    if out.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    return output_path


def _stage_tiles(patches_path: Path, tile_grid: gpd.GeoDataFrame,
                 output_path: Path) -> Optional[Path]:
    if output_path.exists():
        return output_path
    patches = gpd.read_parquet(patches_path)
    if patches.empty:
        return None

    tile_bounds = {int(row.tile): row.geometry for row in tile_grid.itertuples()}
    assignments = categorize_using_bounds(tile_bounds, patches, crs=WORK_CRS)

    out = patches[["patch_i", "patch_j"]].copy()
    out["tile"] = assignments
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    return output_path


def _stage_class(patches_path: Path, column_names, tif_paths, clr_paths,
                 output_path: Path) -> Optional[Path]:
    if output_path.exists():
        return output_path
    patches = gpd.read_parquet(patches_path)
    if patches.empty:
        return None

    classified = generate_class_gdf(
        patches.copy(),
        column_names=list(column_names),
        tif_paths=list(tif_paths),
        clr_paths=list(clr_paths),
        return_new_data_only=False,
    )
    out = classified[["patch_i", "patch_j", *column_names]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    return output_path


# ============================================================
# Per-supertile worker
# ============================================================


def _process_one_supertile(job):
    """Run patches + each enabled downstream stage for a single supertile."""
    (
        supertile_idx,
        supertile_polygon,
        gpkg_path,
        src_crs,
        patch_size,
        grid_origin,
        targets,
        paths,
        cell_grid,
        tile_grid,
        drop_unassigned,
        column_names,
        tif_paths,
        clr_paths,
    ) = job

    written = {}
    try:
        patches_path = paths["patches"]
        if not patches_path.exists():
            result = process_supertile_patches(
                supertile_polygon, patch_size, grid_origin, patches_path,
            )
            written["patches"] = result
            if result is None:
                # No patches in this supertile — downstream stages have nothing to do.
                return supertile_idx, written
        else:
            written["patches"] = patches_path

        if "hedgerows" in targets:
            written["hedgerows"] = process_supertile_hedgerows(
                patches_path, supertile_polygon, gpkg_path, src_crs,
                patch_size, paths["hedgerows"],
            )
        if "folds" in targets:
            written["folds"] = _stage_folds(
                patches_path, cell_grid, drop_unassigned, paths["folds"]
            )
        if "tiles" in targets:
            written["tiles"] = _stage_tiles(
                patches_path, tile_grid, paths["tiles"]
            )
        if "class" in targets:
            written["class"] = _stage_class(
                patches_path, column_names, tif_paths, clr_paths, paths["class"]
            )
    except Exception as e:
        logging.exception("Supertile %d failed: %s", supertile_idx, e)
    return supertile_idx, written


# ============================================================
# Orchestrator
# ============================================================


def run_pipeline(
    gpkg_path: str,
    output_dir: Path,
    targets: list[str],
    patch_size: int = PATCH_SIZE_M,
    supertile_size: Optional[float] = 50 * PATCH_SIZE_M,
    supertile_rows: Optional[int] = None,
    supertile_cols: Optional[int] = None,
    patches_per_cell: int = PATCHES_PER_CELL,
    patches_between_cells: int = PATCHES_BETWEEN_CELLS,
    n_folds: int = N_FOLDS,
    fold_seed: int = 15,
    drop_unassigned: bool = True,
    tile_rows: int = TILES_N_ROWS,
    tile_cols: int = TILES_N_COLS,
    tile_epsilon: int = TILES_EPSILON,
    tile_buffer_pct: float = TILE_BUFFER_PCT,
    column_names: Optional[list[str]] = None,
    tif_paths: Optional[list[str]] = None,
    clr_paths: Optional[list[str]] = None,
    workers: int = 1,
    aoi_bounds: Optional[tuple[float, float, float, float]] = None,
    legacy_manifests_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Run the full supertiled dataset pipeline. Returns the output directory."""
    log = logger or logging.getLogger(__name__)

    output_dir = Path(output_dir)
    manifest_dir = output_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / STAGE_DIRS["patches"]).mkdir(parents=True, exist_ok=True)
    for t in targets:
        (output_dir / STAGE_DIRS[t]).mkdir(parents=True, exist_ok=True)

    if legacy_manifests_dir is not None:
        _stage_legacy_manifests(Path(legacy_manifests_dir), manifest_dir, targets, log)

    # AOI + grids
    src_crs, src_bounds, aoi_gdf = _get_aoi(gpkg_path, aoi_bounds_override=aoi_bounds)
    aoi_bounds_work = tuple(aoi_gdf.total_bounds)
    log.info("Source CRS=%s bounds=%s", src_crs, src_bounds)
    log.info("AOI bounds in %s: %s", WORK_CRS, aoi_bounds_work)

    with open(manifest_dir / "aoi.json", "w") as fh:
        json.dump(
            {
                "gpkg_path": gpkg_path,
                "src_crs": str(src_crs),
                "src_bounds": list(map(float, src_bounds)),
                "work_crs": WORK_CRS,
                "work_bounds": list(map(float, aoi_bounds_work)),
                "patch_size": patch_size,
                "supertile_size": supertile_size,
                "fold_seed": fold_seed,
                "n_folds": n_folds,
                "legacy_manifests_dir": str(legacy_manifests_dir) if legacy_manifests_dir else None,
            },
            fh,
            indent=2,
        )

    supertile_grid = _build_or_load_supertile_grid(
        manifest_dir, aoi_gdf, supertile_size, supertile_rows, supertile_cols, log
    )

    cell_grid = None
    if "folds" in targets:
        cell_grid = _build_or_load_cell_fold_grid(
            manifest_dir, aoi_gdf, patch_size,
            patches_per_cell, patches_between_cells, n_folds, fold_seed, log,
        )

    tile_grid = None
    if "tiles" in targets:
        tile_grid = _build_or_load_tile_grid(
            manifest_dir, aoi_gdf, tile_rows, tile_cols, tile_epsilon, tile_buffer_pct, log,
        )

    # Use the AOI's lower-left corner directly as the patch grid origin so
    # patches align with any reference dataset that was built from the same
    # bounds. Snapping to patch_size multiples shifts the origin by up to
    # patch_size-1 metres and misaligns patches with the original pipeline.
    grid_origin = (aoi_bounds_work[0], aoi_bounds_work[1])
    log.info("Global patch grid origin: %s", grid_origin)

    jobs = []
    for row in supertile_grid.itertuples():
        idx = int(row.supertile_id)
        paths = {
            "patches":   output_dir / STAGE_DIRS["patches"]   / f"patches_{idx:05d}.parquet",
            "hedgerows": output_dir / STAGE_DIRS["hedgerows"] / f"hedgerows_{idx:05d}.parquet",
            "folds":     output_dir / STAGE_DIRS["folds"]     / f"folds_{idx:05d}.parquet",
            "tiles":     output_dir / STAGE_DIRS["tiles"]     / f"tiles_{idx:05d}.parquet",
            "class":     output_dir / STAGE_DIRS["class"]     / f"class_{idx:05d}.parquet",
        }
        jobs.append((
            idx,
            row.geometry,
            gpkg_path,
            src_crs,
            patch_size,
            grid_origin,
            tuple(targets),
            paths,
            cell_grid,
            tile_grid,
            drop_unassigned,
            tuple(column_names or []),
            tuple(tif_paths or []),
            tuple(clr_paths or []),
        ))

    log.info("Processing %d supertiles with %d worker(s); targets=%s",
             len(jobs), workers, list(targets))
    if workers == 1:
        for job in tqdm(jobs, desc="supertiles"):
            _process_one_supertile(job)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_process_one_supertile, j) for j in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="supertiles"):
                fut.result()

    return output_dir


# ============================================================
# Final concatenation across all supertiles + stages
# ============================================================


def concatenate_full_dataset(
    output_dir: Path,
    final_path: Path,
    targets: list[str],
    save_crs: Optional[str] = None,
    drop_unassigned_folds: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Read every per-supertile parquet, merge stages by (patch_i, patch_j) within
    each supertile, then concatenate across supertiles and write to final_path.
    Rows are sorted by (patch_i, patch_j) — west-to-east columns, south-to-north
    within a column — matching the legacy asset's row order. Patches without a
    fold are dropped when drop_unassigned_folds is set, otherwise kept with
    fold=-1 (the legacy convention). Output format is inferred from
    final_path's suffix.
    """
    log = logger or logging.getLogger(__name__)
    output_dir = Path(output_dir)
    patches_files = sorted((output_dir / STAGE_DIRS["patches"]).glob("patches_*.parquet"))
    if not patches_files:
        raise FileNotFoundError(f"No patches parquets found under {output_dir}")

    parts = []
    for pfile in tqdm(patches_files, desc="merging supertiles"):
        idx = int(pfile.stem.split("_")[1])
        patches = gpd.read_parquet(pfile)

        for stage in targets:
            stage_path = output_dir / STAGE_DIRS[stage] / f"{stage}_{idx:05d}.parquet"
            if stage_path.exists():
                patches = patches.merge(
                    pd.read_parquet(stage_path),
                    on=["patch_i", "patch_j"],
                    how="left",
                )

        if "hedgerows" in targets:
            if "hedgerows" not in patches.columns:
                continue  # no hedgerows file for this supertile
            patches = patches[patches["hedgerows"].notna()]
        if patches.empty:
            continue
        parts.append(patches)

    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), geometry="geometry", crs=WORK_CRS
    )

    if "folds" in targets and "fold" in combined.columns:
        if drop_unassigned_folds:
            combined = combined[combined["fold"].notna() & (combined["fold"] != -1)]
        else:
            combined["fold"] = combined["fold"].fillna(-1).astype(int)

    combined = combined.sort_values(["patch_i", "patch_j"], ignore_index=True)

    if save_crs and save_crs != WORK_CRS:
        combined = combined.to_crs(save_crs)
        if "hedgerows" in combined.columns:
            combined["hedgerows"] = gpd.GeoSeries.from_wkt(
                combined["hedgerows"], crs=WORK_CRS
            ).to_crs(save_crs).to_wkt()

    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.suffix == ".parquet":
        combined.to_parquet(final_path)
    else:
        combined.to_file(final_path)
    log.info("Wrote final dataset to %s (%d rows)", final_path, len(combined))
    return final_path


# ============================================================
# CLI
# ============================================================


def _parse_targets(raw):
    if not raw or (isinstance(raw, list) and len(raw) == 0):
        return list(ALL_TARGETS)
    if isinstance(raw, list) and raw == ["none"]:
        return []
    if isinstance(raw, str):
        if raw == "all":
            return list(ALL_TARGETS)
        if raw == "none":
            return []
        return [raw]
    return list(raw)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--targets", nargs="*", default=[],
                        help=f"Subset of {ALL_TARGETS} (default: all). Pass 'none' for patches-only.")
    parser.add_argument("--initial_bda_path", default=INITIAL_BDA_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final_output", default=os.environ.get("BASE_ASSET_PATH", None),
                        help="Optional path for the concatenated final dataset (.parquet or .gpkg).")

    # Supertile grid
    parser.add_argument("--supertile_size", type=float, default=50 * PATCH_SIZE_M,
                        help="Approximate supertile edge length in meters. Ignored if --supertile_rows/cols are given.")
    parser.add_argument("--supertile_rows", type=int, default=None)
    parser.add_argument("--supertile_cols", type=int, default=None)

    # Patch / fold
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE_M)
    parser.add_argument("--patches_between_cells", type=int, default=PATCHES_BETWEEN_CELLS)
    parser.add_argument("--patches_per_cell", type=int, default=PATCHES_PER_CELL)
    parser.add_argument("--n_folds", type=int, default=N_FOLDS)
    parser.add_argument("--fold_seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--keep_patches_not_assigned_folds", action="store_true")

    # Tiles (spatial split)
    parser.add_argument("--tile_rows", type=int, default=TILES_N_ROWS)
    parser.add_argument("--tile_cols", type=int, default=TILES_N_COLS)
    parser.add_argument("--tile_epsilon", type=int, default=TILES_EPSILON)
    parser.add_argument("--tile_buffer_pct", type=float, default=TILE_BUFFER_PCT)

    # Class
    parser.add_argument("--column_names", nargs="+",
                        default=["aez_class", "mst_class", "thz_class"])
    parser.add_argument("--tif_paths", nargs="+",
                        default=[
                            f"{AEZ_ROOT}/aez_v9v2_CRUTS32_Hist_8110_100_avg.tif",
                            f"{AEZ_ROOT}/mst_class_CRUTS32_Hist_8110_100_avg.tif",
                            f"{AEZ_ROOT}/thz_class_CRUTS32_Hist_8110_100_avg.tif",
                        ])
    parser.add_argument("--clr_paths", nargs="+",
                        default=[
                            f"{AEZ_ROOT}/AEZ_57classes.clr",
                            f"{AEZ_ROOT}/MoistureRegime_class.clr",
                            f"{AEZ_ROOT}/ThermalRegime_class.clr",
                        ])

    # AOI override
    parser.add_argument(
        "--aoi_bounds", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
        default=[-569582.6015267381, 5066798.146412208,
                 1064977.3984732619, 6634798.146412208],
        help=(
            "Explicit AOI bounding box in EPSG:3857 (xmin ymin xmax ymax). "
            "Overrides the bounds derived from the source GPKG. The default is "
            "the full-precision bounds of haie_2024_france_patches_2024_merged"
            ".gpkg, which the legacy grids were derived from. Do not round "
            "these: a millimetre shift of the patch-grid origin flips "
            "patches whose hedgerows overlap them by less than that."
        ),
    )

    # Legacy-compatible grids
    parser.add_argument(
        "--legacy_manifests_dir", default=LEGACY_MANIFESTS_DIR,
        help=(
            "Directory holding tile_grid.parquet / cell_grid.parquet produced "
            "by build_legacy_manifests.py. When given, tile and fold labels "
            "reproduce the legacy dataset instead of being generated by the "
            "new assignment procedures (default: configs/legacy_manifests via "
            "LEGACY_MANIFESTS_DIR, or unset for the new procedures)."
        ),
    )

    # Execution
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save_crs", default="EPSG:3857")
    parser.add_argument("--log_level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("pipeline")

    targets = _parse_targets(args.targets)
    output_dir = Path(args.output_dir)

    keep_unassigned_folds = args.keep_patches_not_assigned_folds
    if args.legacy_manifests_dir and not keep_unassigned_folds:
        logger.info("Legacy mode: keeping patches without a fold as fold=-1, "
                    "matching the legacy asset.")
        keep_unassigned_folds = True

    run_pipeline(
        gpkg_path=args.initial_bda_path,
        output_dir=output_dir,
        targets=targets,
        patch_size=args.patch_size,
        supertile_size=args.supertile_size,
        supertile_rows=args.supertile_rows,
        supertile_cols=args.supertile_cols,
        patches_per_cell=args.patches_per_cell,
        patches_between_cells=args.patches_between_cells,
        n_folds=args.n_folds,
        fold_seed=args.fold_seed,
        drop_unassigned=not keep_unassigned_folds,
        tile_rows=args.tile_rows,
        tile_cols=args.tile_cols,
        tile_epsilon=args.tile_epsilon,
        tile_buffer_pct=args.tile_buffer_pct,
        column_names=args.column_names,
        tif_paths=args.tif_paths,
        clr_paths=args.clr_paths,
        workers=args.workers,
        aoi_bounds=tuple(args.aoi_bounds) if args.aoi_bounds else None,
        legacy_manifests_dir=args.legacy_manifests_dir,
        logger=logger,
    )

    if args.final_output:
        concatenate_full_dataset(
            output_dir=output_dir,
            final_path=Path(args.final_output),
            targets=targets,
            save_crs=args.save_crs,
            drop_unassigned_folds=not keep_unassigned_folds,
            logger=logger,
        )
