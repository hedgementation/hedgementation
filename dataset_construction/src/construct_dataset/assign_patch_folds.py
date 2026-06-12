import logging
import math
import random
from typing import Callable
import rasterio
import shapely

import pandas as pd

import geopandas as gpd
import numpy as np
import os
from collections import defaultdict
import argparse
from dotenv import load_dotenv

from src.construct_dataset.utils import categorize_using_bounds, get_try_logging
load_dotenv()

PATCH_SIZE = os.getenv("PATCH_SIZE", 128)
PATCHES_BETWEEN_CELLS = os.getenv("PATCHES_BETWEEN_CELLS", 2)
PATCHES_PER_CELL = os.getenv("PATCHES_PER_CELL", 12)
N_FOLDS = os.getenv("N_FOLDS", 5)

BASE_PATCHES_GDF_PATH = os.getenv("BASE_PATCHES_GDF_PATH")
FOLDS_GDF_PATH = os.getenv("FOLDS_GDF_PATH")

RANDOM_SEED = os.getenv("RANDOM_SEED", 15)


def create_cells_with_gaps(gdf, 
                                  patch_size=PATCH_SIZE,
                                  patches_per_cell=12,
                                  patches_between_cells=2,
                                  epsilon=10):
    gdf_projected = gdf.to_crs('EPSG:3857')
    xmin, ymin, xmax, ymax = gdf_projected.total_bounds

    patches = []

    for x0 in np.arange(xmin, xmax, patch_size * (patches_per_cell + patches_between_cells) + epsilon):
        for y0 in np.arange(ymin, ymax, patch_size * (patches_per_cell + patches_between_cells) + epsilon):
            x1 = x0 + patch_size * patches_per_cell
            y1 = y0 + patch_size * patches_per_cell
            patches.append(shapely.geometry.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    
    cells_gdf = gpd.GeoDataFrame(geometry=patches, crs=gdf_projected.crs)
    cells_gdf["cell_id"] = [i for i in range(len(cells_gdf))]
    return cells_gdf

def get_cell_adjacency(cell_gdf: gpd.GeoDataFrame,
                       patch_size: int = PATCH_SIZE,
                       patches_between_cells: int = PATCHES_BETWEEN_CELLS,
                       patches_per_cell: int = PATCHES_PER_CELL):

    neighbor_distance = patch_size * ( patches_between_cells + patches_per_cell * 0.5)

    cell_gdf = cell_gdf.to_crs('EPSG:3857')
    cell_gdf['buffered'] = cell_gdf.geometry.to_crs("EPSG:3857").buffer(neighbor_distance)

    temp_gdf = cell_gdf.copy()
    temp_gdf = temp_gdf.set_geometry('buffered')

    neighbors = gpd.sjoin(temp_gdf, temp_gdf.copy(), predicate='intersects')

    adjacency = defaultdict(set)
    for idx, row in neighbors.iterrows():
        if idx != row['index_right']:
            adjacency[idx].add(row['index_right'])

    cell_gdf = cell_gdf.drop(columns=['buffered'])
    return adjacency

def assign_folds_to_cells_randomly(cell_gdf: gpd.GeoDataFrame,
                                   n_folds: int = N_FOLDS):
    folds = [
        random.randint(0, n_folds - 1) for _ in range(len(cell_gdf))
    ]
    folds = pd.Series(folds)
    return folds




def assign_patch_folds_from_cells(patch_gdf: gpd.GeoDataFrame,
                                  cell_gdf: gpd.GeoDataFrame,
                                  drop_unassigned: bool = True):
    fold_spatial_boundaries = {}

    for i in range(N_FOLDS):
        fold_spatial_boundaries[i] = shapely.unary_union(
            cell_gdf[cell_gdf["fold"] == i]["geometry"]
        )
    patch_folds = categorize_using_bounds(fold_spatial_boundaries,
                                          patch_gdf)
    
    patch_gdf["fold"] = patch_folds
    
    if drop_unassigned:
        patch_gdf = patch_gdf[patch_gdf["fold"] != -1]

    return patch_gdf

def generate_fold_gdf(patches_gdf,
         patch_size,
         patches_per_cell,
         patches_between_cells,
         drop_unassigned,
         n_folds=N_FOLDS,
         start=0,
         limit=-1,
         return_new_data_only=False,
         logger=None,
         random_assignment=True) -> gpd.GeoDataFrame:
    try_logging: Callable = get_try_logging(logger)

    initial_msg = "Constructing folds GDF for" + (
        "all rows" if limit == -1 else f"first {limit} rows"
    ) + (
        "" if start == 0 else f"starting from {start}"
    )

    try_logging(initial_msg)


    if isinstance(patches_gdf, str):
        try_logging(f"Reading patches from {patches_gdf}")
        patches_gdf = gpd.read_file(
            patches_gdf,
            rows=[start,limit],
            )
        try_logging("Patches read successfully.")
        
    try_logging(f"Creating cells with patches of size {patch_size}({patches_per_cell} patches per cell, {patches_between_cells} patches between cells)")
    cells_gdf = create_cells_with_gaps(patches_gdf, 
                                       patch_size, 
                                       patches_between_cells,
                                       patches_per_cell)
    try_logging("Cells created successfully.")

    if random_assignment:
        try_logging(f"Randomly assigning one of {n_folds} folds to each cell.")
        cells_gdf["fold"] = assign_folds_to_cells_randomly(n_folds=n_folds, cell_gdf=cells_gdf)
        try_logging("Folds assigned successfully.")
    else:
        raise NotImplementedError()
    
    try_logging("Assigning patches to cells using cell spatial bounds.")
    patches_gdf = assign_patch_folds_from_cells(
        patches_gdf,
        cells_gdf,
        drop_unassigned=drop_unassigned
    )
    try_logging("Patches assigned succesfully.")
    if return_new_data_only:
        return patches_gdf[["patch_id", "fold"]]
    return patches_gdf
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--BASE_PATCHES_GDF_PATH", default=BASE_PATCHES_GDF_PATH)
    parser.add_argument("--save_path", default=FOLDS_GDF_PATH)
    parser.add_argument("--patch_size", type="int", default=PATCH_SIZE)
    parser.add_argument("--patches_between_cells", type="int", default=PATCHES_BETWEEN_CELLS)
    parser.add_argument("--patches_per_cell", type="int", default=PATCHES_PER_CELL)

    args = parser.parse_args()
    folds_gdf = generate_fold_gdf(patches_gdf=args.BASE_PATCHES_GDF_PATH,
                     patch_size=args.patch_size,
                     patches_between_cells=args.patches_between_cells,
                     patches_per_cell=args.patches_per_cell)
    
    folds_gdf.to_file(args.save_path)