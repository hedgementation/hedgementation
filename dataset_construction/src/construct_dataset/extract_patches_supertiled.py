"""
Supertile-based patch extraction pipeline.

Splits the AOI into coarse supertiles (using the project's existing
`create_tiles` helper), streams only the linestrings inside each supertile
from the source GPKG via pyogrio's bbox filter, builds the patch grid
locally on a globally-aligned origin so adjacent supertiles tile seamlessly,
runs a spatial join, and writes one parquet file per supertile.

Supertiles whose output already exists are skipped so a crashed run can be
resumed by re-invoking the same command. An optional final step
concatenates all per-tile parquets into a single output.
"""

import argparse
import logging
import math
import os
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

from src.construct_dataset.assign_tile_and_tilegroup import create_tiles

load_dotenv()

INITIAL_BDA_PATH = os.getenv("INITIAL_BDA_PATH")
PATCH_SIZE_M = int(os.getenv("PATCH_SIZE_M", 1280))
WORK_CRS = "EPSG:3857"  # create_tiles always works in EPSG:3857.


def _rows_cols_for_supertile_size(aoi_bounds, supertile_size):
    """Pick row/col counts so each supertile is roughly supertile_size meters per side."""
    xmin, ymin, xmax, ymax = aoi_bounds
    cols = max(1, math.ceil((xmax - xmin) / supertile_size))
    rows = max(1, math.ceil((ymax - ymin) / supertile_size))
    return rows, cols


def _query_bbox_in_src_crs(query_poly, src_crs):
    """
    Reproject a WORK_CRS query polygon to the source CRS and return its
    bounds for pyogrio's bbox filter. The polygon is densified first: for
    conic source CRSs (e.g. Lambert 93) reprojecting only the corners
    underestimates the extent between them, and features in the missed
    sliver near supertile edges would be silently dropped.
    """
    edge_len = max(
        query_poly.bounds[2] - query_poly.bounds[0],
        query_poly.bounds[3] - query_poly.bounds[1],
    )
    dense = shapely.segmentize(query_poly, max_segment_length=edge_len / 100)
    return gpd.GeoSeries([dense], crs=WORK_CRS).to_crs(src_crs).total_bounds


def _patches_in_tile(tile_bounds, patch_size, grid_origin):
    """
    Generate (patch_id_x, patch_id_y, polygon) for every patch whose origin
    lies inside tile_bounds, anchored on a global grid_origin so patches in
    neighbouring supertiles never overlap or leave gaps.
    """
    ox, oy = grid_origin
    tx0, ty0, tx1, ty1 = tile_bounds

    i_start = math.ceil((tx0 - ox) / patch_size)
    i_end = math.ceil((tx1 - ox) / patch_size)
    j_start = math.ceil((ty0 - oy) / patch_size)
    j_end = math.ceil((ty1 - oy) / patch_size)

    ids = []
    geoms = []
    for i in range(i_start, i_end):
        x0 = ox + i * patch_size
        for j in range(j_start, j_end):
            y0 = oy + j * patch_size
            ids.append((i, j))
            geoms.append(box(x0, y0, x0 + patch_size, y0 + patch_size))
    return ids, geoms


def process_supertile_patches_only(
    tile_polygon: shapely.Polygon,
    patch_size: float,
    grid_origin: tuple[float, float],
    output_path: Path,
) -> Optional[Path]:
    """
    Build the patch grid for a supertile without loading any hedgerow data.
    Returns the output path if patches were written, else None.
    """
    if output_path.exists():
        return output_path

    tile_bounds = tile_polygon.bounds
    patch_ids, patch_geoms = _patches_in_tile(tile_bounds, patch_size, grid_origin)
    if not patch_geoms:
        return None

    out_gdf = gpd.GeoDataFrame(
        {
            "patch_i": [pid[0] for pid in patch_ids],
            "patch_j": [pid[1] for pid in patch_ids],
        },
        geometry=patch_geoms,
        crs=WORK_CRS,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_parquet(output_path)
    return output_path


def process_supertile_hedgerows(
    patches_path: Path,
    tile_polygon: shapely.Polygon,
    gpkg_path: str,
    src_crs: str,
    patch_size: float,
    output_path: Path,
) -> Optional[Path]:
    """
    Assign hedgerow geometries to patches in a supertile. Reads the existing
    patches parquet, streams lines from the GPKG, then writes a parquet with
    (patch_i, patch_j, hedgerows) for every patch that intersects a line.
    Returns the output path if written, else None.
    """
    if output_path.exists():
        return output_path

    patches = gpd.read_parquet(patches_path)
    if patches.empty:
        return None

    tile_bounds = tile_polygon.bounds
    # Expand bbox by patch_size on the high side so line segments falling
    # inside patches whose origin sits on the supertile boundary are included.
    query_poly = box(
        tile_bounds[0],
        tile_bounds[1],
        tile_bounds[2] + patch_size,
        tile_bounds[3] + patch_size,
    )
    bbox_src = _query_bbox_in_src_crs(query_poly, src_crs)

    lines = pyogrio.read_dataframe(
        gpkg_path, bbox=tuple(bbox_src), use_arrow=True, columns=[]
    )
    if lines.empty:
        return None

    lines = lines.to_crs(WORK_CRS).reset_index(drop=True)

    joined = gpd.sjoin(
        patches[["patch_i", "patch_j", "geometry"]],
        lines[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return None

    line_geoms = lines.geometry.values
    patch_lookup = {
        (row.patch_i, row.patch_j): row.geometry for row in patches.itertuples()
    }

    results = []
    for (pi, pj), group in joined.groupby(["patch_i", "patch_j"], sort=False):
        idxs = group["index_right"].values
        union = shapely.unary_union(line_geoms[idxs])
        patch_poly = patch_lookup[(pi, pj)]
        clipped = union.intersection(patch_poly)
        if clipped.is_empty:
            continue
        results.append({"patch_i": int(pi), "patch_j": int(pj), "hedgerows": clipped})

    if not results:
        return None

    out_df = pd.DataFrame(results)
    # Store as WKT so the parquet has no extra geometry column.
    out_df["hedgerows"] = gpd.GeoSeries(out_df["hedgerows"], crs=WORK_CRS).to_wkt()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)
    return output_path


def process_supertile(
    tile_polygon: shapely.Polygon,
    gpkg_path: str,
    src_crs: str,
    patch_size: float,
    grid_origin: tuple[float, float],
    output_path: Path,
) -> Optional[Path]:
    """
    Process a single supertile: stream lines via bbox, build patches on the
    global grid, sjoin, dissolve+clip per patch, write parquet. Returns the
    output path if a file was written, else None.
    """
    if output_path.exists():
        return output_path

    tile_bounds = tile_polygon.bounds  # in WORK_CRS (EPSG:3857)

    # Expand the GPKG bbox query by patch_size on the high side so we still
    # catch line segments that fall inside patches whose origin is on the
    # supertile boundary.
    query_poly = box(
        tile_bounds[0],
        tile_bounds[1],
        tile_bounds[2] + patch_size,
        tile_bounds[3] + patch_size,
    )
    bbox_src = _query_bbox_in_src_crs(query_poly, src_crs)

    lines = pyogrio.read_dataframe(
        gpkg_path, bbox=tuple(bbox_src), use_arrow=True, columns=[]
    )
    if lines.empty:
        return None

    lines = lines.to_crs(WORK_CRS).reset_index(drop=True)

    patch_ids, patch_geoms = _patches_in_tile(tile_bounds, patch_size, grid_origin)
    if not patch_geoms:
        return None

    patches = gpd.GeoDataFrame(
        {
            "patch_i": [pid[0] for pid in patch_ids],
            "patch_j": [pid[1] for pid in patch_ids],
        },
        geometry=patch_geoms,
        crs=WORK_CRS,
    )

    joined = gpd.sjoin(
        patches[["patch_i", "patch_j", "geometry"]],
        lines[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    if joined.empty:
        return None

    line_geoms = lines.geometry.values

    # Dissolve hedgerow geometry per patch, then clip to the patch polygon.
    results = []
    patch_lookup = {
        (row.patch_i, row.patch_j): row.geometry for row in patches.itertuples()
    }
    for (pi, pj), group in joined.groupby(["patch_i", "patch_j"], sort=False):
        idxs = group["index_right"].values
        union = shapely.unary_union(line_geoms[idxs])
        patch_poly = patch_lookup[(pi, pj)]
        clipped = union.intersection(patch_poly)
        if clipped.is_empty:
            continue
        results.append(
            {
                "patch_i": int(pi),
                "patch_j": int(pj),
                "geometry": patch_poly,
                "hedgerows": clipped,
            }
        )

    if not results:
        return None

    out_gdf = gpd.GeoDataFrame(results, crs=WORK_CRS, geometry="geometry")
    # Store hedgerows as WKT so the parquet has a single active geometry
    # column; downstream readers rehydrate via shapely.wkt.loads.
    out_gdf["hedgerows"] = gpd.GeoSeries(
        out_gdf["hedgerows"], crs=WORK_CRS
    ).to_wkt()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_parquet(output_path)
    return output_path


def _worker(args):
    tile_polygon, gpkg_path, src_crs, patch_size, grid_origin, output_path = args
    try:
        return process_supertile(
            tile_polygon, gpkg_path, src_crs, patch_size, grid_origin, output_path
        )
    except Exception as e:
        logging.exception("Supertile %s failed: %s", output_path.name, e)
        return None


def run_pipeline(
    gpkg_path: str,
    output_dir: Path,
    patch_size: float = PATCH_SIZE_M,
    supertile_size: Optional[float] = 50_000.0,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    workers: int = 1,
    logger: Optional[logging.Logger] = None,
) -> list[Path]:
    """
    Run the supertile pipeline. Provide either supertile_size (used to derive
    rows/cols) or rows+cols directly. Returns the list of parquet files
    written (including ones skipped because they already existed).
    """
    log = logger or logging.getLogger(__name__)

    info = pyogrio.read_info(gpkg_path)
    src_crs = info["crs"]
    src_bounds = info["total_bounds"]
    log.info(
        "Source: %s features, CRS=%s, bounds=%s",
        info["features"],
        src_crs,
        src_bounds,
    )

    # Reproject AOI to the work CRS once for grid construction.
    aoi_gdf = gpd.GeoDataFrame(
        geometry=[box(*src_bounds)], crs=src_crs
    ).to_crs(WORK_CRS)
    aoi_bounds_work = aoi_gdf.total_bounds
    log.info("AOI bounds in %s: %s", WORK_CRS, aoi_bounds_work)

    if rows is None or cols is None:
        rows, cols = _rows_cols_for_supertile_size(aoi_bounds_work, supertile_size)
    log.info("Tiling AOI into %d rows x %d cols = %d supertiles", rows, cols, rows * cols)

    tiles = create_tiles(aoi_gdf, rows=rows, cols=cols, epsilon=1, buffer_pct=0, logger=log)

    # Global patch grid origin: snap AOI's lower-left to a patch_size grid.
    grid_origin = (
        math.floor(aoi_bounds_work[0] / patch_size) * patch_size,
        math.floor(aoi_bounds_work[1] / patch_size) * patch_size,
    )
    log.info("Global patch grid origin: %s", grid_origin)

    output_dir.mkdir(parents=True, exist_ok=True)
    job_args = []
    for i, tile in enumerate(tiles):
        out_path = output_dir / f"patches_tile_{i:05d}.parquet"
        job_args.append((tile, gpkg_path, src_crs, patch_size, grid_origin, out_path))

    written: list[Path] = []
    if workers == 1:
        for args in tqdm(job_args, desc="supertiles"):
            result = _worker(args)
            if result is not None:
                written.append(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, a) for a in job_args]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="supertiles"):
                result = fut.result()
                if result is not None:
                    written.append(result)

    log.info("Wrote %d non-empty supertile parquets", len(written))
    return written


def concatenate_parquets(
    parquet_dir: Path,
    output_path: Path,
    save_crs: Optional[str] = None,
) -> Path:
    """
    Concatenate all per-tile parquet files into a single output. The output
    format is inferred from the suffix of output_path (.parquet, .gpkg, or
    anything supported by GeoPandas).
    """
    files = sorted(parquet_dir.glob("patches_tile_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No tile parquets found in {parquet_dir}")

    parts = [gpd.read_parquet(f) for f in tqdm(files, desc="reading parts")]
    combined = gpd.GeoDataFrame(
        pd.concat(parts, ignore_index=True), crs=WORK_CRS, geometry="geometry"
    )

    if save_crs and save_crs != WORK_CRS:
        combined = combined.to_crs(save_crs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        combined.to_parquet(output_path)
    else:
        combined.to_file(output_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdf_path", default=INITIAL_BDA_PATH)
    parser.add_argument(
        "--output_dir",
        default="test_data/supertiles",
        help="Directory for per-supertile parquet files.",
    )
    parser.add_argument(
        "--final_output",
        default=None,
        help="Optional concatenated output path (.parquet or .gpkg). "
        "If omitted, only per-tile parquets are written.",
    )
    parser.add_argument("--patch_size", type=float, default=PATCH_SIZE_M)
    parser.add_argument(
        "--supertile_size",
        type=float,
        default=50_000.0,
        help="Approximate supertile edge length (m). Ignored if --rows/--cols given.",
    )
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cols", type=int, default=None)
    parser.add_argument("--save_crs", default="EPSG:3857")
    parser.add_argument("--workers", type=int, default=1)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger("supertiled")

    output_dir = Path(args.output_dir)
    run_pipeline(
        gpkg_path=args.gdf_path,
        output_dir=output_dir,
        patch_size=args.patch_size,
        supertile_size=args.supertile_size,
        rows=args.rows,
        cols=args.cols,
        workers=args.workers,
        logger=logger,
    )

    if args.final_output:
        concatenate_parquets(
            parquet_dir=output_dir,
            output_path=Path(args.final_output),
            save_crs=args.save_crs,
        )
