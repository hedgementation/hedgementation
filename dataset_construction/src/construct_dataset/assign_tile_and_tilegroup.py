import argparse
import logging
from typing import Callable

import numpy as np
import shapely

from src.construct_dataset.utils import categorize_using_bounds, get_try_logging
import geopandas as gpd

import os

import argparse

import numpy as np
import shapely

from src.construct_dataset.utils import categorize_using_bounds
import geopandas as gpd

TILES_N_ROWS = os.getenv("TILES_N_ROWS")
TILES_N_COLS = os.getenv("TILES_N_COLS")
TILES_EPSILON = os.getenv("TILES_EPSILON")
TILE_BUFFER_PCT = os.getenv("TILE_BUFFER_PCT")
TILE_GDF_PATH = os.getenv("TILE_GDF_PATH")

RANDOM_SEED = os.getenv("RANDOM_SEED", 15)


def create_tiles(gdf, 
                    rows=2, 
                    cols=2, 
                    epsilon=1, 
                    buffer_pct=None,
                    logger:logging.Logger= None) -> list[shapely.Polygon]:
    #TODO return the buffer distance between tiles

    try_logging: Callable = get_try_logging(logger)
    gdf = gdf.to_crs("EPSG:3857")
    minx, miny, maxx, maxy = gdf.total_bounds
    step_x = (maxx - minx ) / (cols)
    step_y = (maxy - miny ) / (rows)
    x_steps = np.arange(minx, maxx + step_x-epsilon, step_x)
    y_steps = np.arange(miny, maxy + step_y-epsilon, step_y)
    
    try_logging(
        f"Creating a grid of tiles with {rows} rows and {cols} columns({rows * cols} total tiles)"
    )

    if buffer_pct:
        try_logging(f"Using buffer distance of {int(buffer_pct * step_x) // 1000} km horizontally, {int(buffer_pct * step_y) // 1000} km vertically")

    tile_width_km = (step_x * (1 - buffer_pct)) // 1000
    tile_height_km = (step_y * (1 - buffer_pct)) // 1000

    try_logging(
        f"Tile dimensions: {tile_width_km} km wide, {tile_height_km} high"
    )

    sections = []
    for i in range(len(y_steps)-1):
        for j in range(len(x_steps)-1):
            if not buffer_pct:
                sec = shapely.geometry.box(x_steps[j], y_steps[i], x_steps[j+1], y_steps[i+1])
                
            else:
                buffer_amt_x = step_x * (buffer_pct/2)
                buffer_amt_y = step_y * (buffer_pct/2)
                sec = shapely.geometry.box(x_steps[j] + buffer_amt_x, 
                                           y_steps[i] + buffer_amt_y, 
                                           x_steps[j+1] - buffer_amt_x, 
                                           y_steps[i+1] - buffer_amt_y)
            sections.append(sec)
    # Deterministic row-major ordering: top row first, west-to-east within
    # each row. Sorting on raw centroid coordinates is not reliable here:
    # centroids of boxes in the same row can differ by 1 ulp in y (shapely's
    # centroid formula mixes x into the y computation), so a y-primary sort
    # scrambles the within-row order depending on coordinate bit patterns
    # and the GEOS version.
    n_rows = len(y_steps) - 1
    n_cols = len(x_steps) - 1
    sections = [
        sections[i * n_cols + j]
        for i in range(n_rows - 1, -1, -1)
        for j in range(n_cols)
    ]
    try_logging("Tiles created.")

    return sections
    
def generate_tiles_gdf(patches_gdf: gpd.GeoDataFrame,
                            rows: int,
                            cols: int,
                            epsilon: int,
                            buffer_pct: float,
                            return_new_data_only: bool = False,
                            logger: logging.Logger = None) -> gpd.GeoDataFrame:
    try_logging: Callable = get_try_logging(logger)

    if isinstance(patches_gdf, str):
        try_logging(f"Reading GDF from {patches_gdf}...")
        patches_gdf = gpd.read_file(patches_gdf)

    
    tiles = create_tiles(patches_gdf, rows, cols, epsilon=epsilon, buffer_pct=buffer_pct)
    tile_num_to_tile_bounds = {
        i:tile for i,tile in enumerate(tiles)
    }
    try_logging("Assigning patches to tiles using spatial bounds.")
    tile_assigments = categorize_using_bounds(bounds_dict=tile_num_to_tile_bounds, gdf=patches_gdf)

    patches_gdf["tile"] = tile_assigments
    
    if return_new_data_only:
        return patches_gdf[["tile", "patch_id"]]
    return patches_gdf
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gdf_path")
    parser.add_argument("--rows", type=int, default=TILES_N_ROWS)
    parser.add_argument("--cols", type=int, default=TILES_N_COLS)
    parser.add_argument("--epsilon", type=int, default=TILES_EPSILON)
    parser.add_argument("--buffer_pct", type=float, default=TILE_BUFFER_PCT)
    parser.add_argument("--output_path", type=str, default=TILE_GDF_PATH)
    args = parser.parse_args()

    gdf = generate_tiles_gdf(
        patches_gdf=args.gdf_path,
        rows=args.rows,
        cols=args.cols,
        epsilon=args.epsilon,
        buffer_pct=args.buffer_pct,
    )

    gdf.to_file(args.output_path)