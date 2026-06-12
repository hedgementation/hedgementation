import logging
import numpy as np
import geopandas as gpd
import os
from shapely import Polygon
from dotenv import load_dotenv
import shapely

load_dotenv()
PATCH_SIZE_M = int(os.getenv("PATCH_SIZE_M", 1280))


def create_patches(gdf: gpd.GeoDataFrame,
                   patch_size:int=PATCH_SIZE_M) ->gpd.GeoDataFrame:
    """
    Create an array of square patches that covers the full breadth of 
    the geodataframe. 
    """

    gdf_projected = gdf.to_crs('EPSG:3857')
    xmin, ymin, xmax, ymax = gdf_projected.total_bounds
    grid_cells = []

    for x0 in np.arange(xmin, xmax, patch_size):
        for y0 in np.arange(ymin, ymax, patch_size):
            x1 = x0 + patch_size
            y1 = y0 + patch_size
            grid_cells.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    patches_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=gdf_projected.crs)

    return patches_gdf

def get_try_logging(logger):
    def try_logging(logger,msg):
        if logger:
            logger.log(level=logging.DEBUG, msg=msg)

    return lambda x: try_logging(logger, x)

def categorize_using_bounds(bounds_dict, gdf, crs="EPSG:3857"):
    gdf = gdf.to_crs(crs) 
    tree = shapely.STRtree(gdf.geometry)
    assignments = [-1] * len(gdf)

    for k in bounds_dict:
        current_bounds = bounds_dict[k]

        new_indices = tree.query(current_bounds, predicate="intersects")
        new_indices = set(new_indices)

        for j in new_indices:
            assignments[j] = k

    return assignments