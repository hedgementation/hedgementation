"""
One-time canonicalization of the legacy tile / fold-cell grids.

The legacy dataset (bda_2024_patches_2024_classified.gpkg) was produced by a
notebook whose tile ordering depended on floating-point noise in a centroid
sort and whose fold assignment used an unseeded RNG: neither can be
regenerated from first principles. This script recovers both as data instead,
by reading the legacy asset once and writing two small grid manifests:

  manifest/tile_grid.parquet   6x6 buffered tile polygons labeled with the
                               legacy tile indices (recovered by matching
                               patch membership between the grids).
  manifest/cell_grid.parquet   fold checkerboard cells at the legacy pitch
                               (12 patches per cell, 2 between), each labeled
                               with the fold of the legacy patches inside it.

Point construct_base_asset.py at the output directory with
--legacy_manifests_dir to build a dataset whose tile and fold labels match
the legacy asset, without the pipeline ever reading the legacy asset itself.
"""

import argparse
import json
import logging
import random
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import box

from src.construct_dataset.assign_patch_folds import create_cells_with_gaps
from src.construct_dataset.assign_tile_and_tilegroup import create_tiles
from src.construct_dataset.extract_patches_supertiled import WORK_CRS
from src.construct_dataset.utils import categorize_using_bounds

# Full-precision bounds of the asset the legacy grids were derived from
# (haie_2024_france_patches_2024_merged.gpkg, EPSG:3857) — the same default
# construct_base_asset.py uses for --aoi_bounds. Do not round: a millimetre
# shift of the grid origin flips boundary-touching patch assignments.
DEFAULT_AOI_BOUNDS = (-569582.6015267381, 5066798.146412208,
                      1064977.3984732619, 6634798.146412208)

# The legacy notebook run used create_checkerboard_with_gaps' default of 2
# patches between cells (its source now says 3, but the saved asset's gap
# bands sit on a 1280*(12+2)+10 m pitch).
LEGACY_PATCHES_BETWEEN_CELLS = 2

logger = logging.getLogger("legacy_manifests")


def recover_tile_grid(legacy: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame,
                      rows: int, cols: int, epsilon: int, buffer_pct: float):
    """Build the tile polygons and relabel them with the legacy tile indices."""
    polys = create_tiles(aoi_gdf, rows=rows, cols=cols,
                         epsilon=epsilon, buffer_pct=buffer_pct)
    assignments = pd.Series(
        categorize_using_bounds({i: p for i, p in enumerate(polys)}, legacy)
    )

    mapping = {}  # row-major grid index -> legacy tile label
    impure = 0
    legacy_tiles = legacy["tile"].reset_index(drop=True)
    for legacy_label in sorted(legacy_tiles[legacy_tiles >= 0].unique()):
        counts = assignments[legacy_tiles == legacy_label].value_counts()
        grid_idx = int(counts.idxmax())
        impure += int(counts.sum() - counts.max())
        if grid_idx in mapping:
            raise RuntimeError(
                f"Legacy tiles {mapping[grid_idx]} and {legacy_label} both map "
                f"to grid index {grid_idx}; patch membership is ambiguous."
            )
        mapping[grid_idx] = int(legacy_label)
    if impure:
        logger.warning("%d legacy patches fell in a different tile than their "
                       "label's majority; mapping used the majority.", impure)

    # Tiles with no legacy patches never label a patch; hand the unused legacy
    # labels to the unmatched grid indices in deterministic (row-major) order.
    unused_labels = sorted(set(range(len(polys))) - set(mapping.values()))
    unmatched_idxs = sorted(set(range(len(polys))) - set(mapping.keys()))
    mapping.update(zip(unmatched_idxs, unused_labels))

    tile_gdf = gpd.GeoDataFrame(
        {"tile": [mapping[i] for i in range(len(polys))],
         "grid_index": np.arange(len(polys))},
        geometry=list(polys),
        crs=WORK_CRS,
    )

    relabeled = {int(row.tile): row.geometry for row in tile_gdf.itertuples()}
    check = pd.Series(categorize_using_bounds(relabeled, legacy))
    agreement = float((check.values == legacy_tiles.values).mean())
    logger.info("Tile validation: %.4f%% of legacy patches get their legacy "
                "label back (incl. -1).", 100 * agreement)
    return tile_gdf, {"tile_agreement": agreement, "impure_patches": impure}


def recover_cell_grid(legacy: gpd.GeoDataFrame, aoi_gdf: gpd.GeoDataFrame,
                      patch_size: int, patches_per_cell: int,
                      patches_between_cells: int, n_folds: int, seed: int):
    """Build the fold cells and label each with its legacy patches' fold."""
    cells_gdf = create_cells_with_gaps(
        aoi_gdf,
        patch_size=patch_size,
        patches_per_cell=patches_per_cell,
        patches_between_cells=patches_between_cells,
    )
    assignments = pd.Series(categorize_using_bounds(
        {int(c.cell_id): c.geometry for c in cells_gdf.itertuples()}, legacy
    ))

    legacy_folds = legacy["fold"].reset_index(drop=True)
    votes = pd.DataFrame({"cell": assignments, "fold": legacy_folds})
    votes = votes[(votes["cell"] >= 0) & (votes["fold"] >= 0)]
    majority = votes.groupby("cell")["fold"].agg(lambda s: s.mode().iloc[0])
    disagreeing = int(
        (votes["fold"].values != majority.reindex(votes["cell"]).values).sum()
    )
    if disagreeing:
        logger.warning("%d legacy patches disagree with their cell's majority "
                       "fold; majority wins.", disagreeing)

    # Cells with no legacy patches inside still need a fold so that new source
    # data appearing there gets covered: draw one with a seeded RNG.
    rng = random.Random(seed)
    folds, sources = [], []
    for cell_id in cells_gdf["cell_id"]:
        if cell_id in majority.index:
            folds.append(int(majority.loc[cell_id]))
            sources.append("legacy")
        else:
            folds.append(rng.randint(0, n_folds - 1))
            sources.append("random")
    cells_gdf["fold"] = folds
    cells_gdf["fold_source"] = sources
    n_random = sources.count("random")
    logger.info("Cells: %d total, %d labeled from the legacy asset, %d empty "
                "(seeded random fold).", len(cells_gdf), len(cells_gdf) - n_random, n_random)

    fold_unions = {
        f: shapely.unary_union(cells_gdf[cells_gdf["fold"] == f].geometry.values)
        for f in range(n_folds)
    }
    check = pd.Series(categorize_using_bounds(fold_unions, legacy))
    agreement = float((check.values == legacy_folds.values).mean())
    assigned = legacy_folds.values >= 0
    agreement_assigned = float(
        (check.values[assigned] == legacy_folds.values[assigned]).mean()
    )
    logger.info("Fold validation: %.4f%% overall agreement with legacy labels, "
                "%.4f%% among legacy-assigned patches.",
                100 * agreement, 100 * agreement_assigned)
    return cells_gdf, {
        "fold_agreement": agreement,
        "fold_agreement_assigned": agreement_assigned,
        "cells_without_legacy_patches": n_random,
        "disagreeing_patches": disagreeing,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy_asset",
                        default="data/gpkgs/bda_2024_patches_2024_classified.gpkg")
    parser.add_argument("--output_dir", default="configs/legacy_manifests")
    parser.add_argument("--aoi_bounds", nargs=4, type=float,
                        metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        default=list(DEFAULT_AOI_BOUNDS),
                        help="AOI bounding box in EPSG:3857; must match the "
                             "--aoi_bounds the pipeline will be run with.")
    parser.add_argument("--tile_rows", type=int, default=6)
    parser.add_argument("--tile_cols", type=int, default=6)
    parser.add_argument("--tile_epsilon", type=int, default=1)
    parser.add_argument("--tile_buffer_pct", type=float, default=0.1)
    parser.add_argument("--patch_size", type=int, default=1280)
    parser.add_argument("--patches_per_cell", type=int, default=12)
    parser.add_argument("--patches_between_cells", type=int,
                        default=LEGACY_PATCHES_BETWEEN_CELLS)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=15,
                        help="Seed for folds of cells with no legacy patches.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    logger.info("Reading legacy asset %s ...", args.legacy_asset)
    legacy = gpd.read_file(args.legacy_asset, columns=["fold", "tile"])
    legacy = legacy.to_crs(WORK_CRS)
    logger.info("Read %d legacy patches.", len(legacy))

    aoi_gdf = gpd.GeoDataFrame(geometry=[box(*args.aoi_bounds)], crs=WORK_CRS)

    tile_gdf, tile_stats = recover_tile_grid(
        legacy, aoi_gdf, args.tile_rows, args.tile_cols,
        args.tile_epsilon, args.tile_buffer_pct,
    )
    cells_gdf, cell_stats = recover_cell_grid(
        legacy, aoi_gdf, args.patch_size, args.patches_per_cell,
        args.patches_between_cells, args.n_folds, args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tile_gdf.to_parquet(output_dir / "tile_grid.parquet")
    cells_gdf.to_parquet(output_dir / "cell_grid.parquet")
    with open(output_dir / "provenance.json", "w") as fh:
        json.dump({
            "legacy_asset": args.legacy_asset,
            "created": date.today().isoformat(),
            "aoi_bounds": args.aoi_bounds,
            "params": {k: getattr(args, k) for k in (
                "tile_rows", "tile_cols", "tile_epsilon", "tile_buffer_pct",
                "patch_size", "patches_per_cell", "patches_between_cells",
                "n_folds", "seed")},
            "validation": {**tile_stats, **cell_stats},
        }, fh, indent=2)
    logger.info("Wrote tile_grid.parquet, cell_grid.parquet and "
                "provenance.json to %s", output_dir)
