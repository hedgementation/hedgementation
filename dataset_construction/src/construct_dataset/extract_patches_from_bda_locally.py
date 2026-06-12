import logging
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely import Polygon, wkt
from rtree import index
from shapely import MultiLineString
import shapely
import argparse
import dask_geopandas

import ee

from tqdm import tqdm
from dotenv import load_dotenv
import os

from rtree import index
from src.construct_dataset.utils import create_patches

load_dotenv()

tqdm.pandas()

INITIAL_BDA_PATH = os.getenv("INITIAL_BDA_PATH")
PATCHES_GDF_PATH = os.getenv("PATCHES_GDF_PATH", "test_data/patches_gdf.geojson")
PATCH_SIZE_M = int(os.getenv("PATCH_SIZE_M", 1280))

def prepare_gdf(gdf, crs="EPSG:3857"):
    if isinstance(gdf["geometry"].iloc[0],str):
        gdf["geometry"] = gdf["geometry"].apply(wkt.loads)
    gdf = gdf.to_crs(crs)
    return gdf

def get_intersecting_lines_legacy(polygon: shapely.geometry.Polygon, 
                           linestrings_series: gpd.GeoSeries,
                           idx: index.Index):
    """
    Efficiently finds all the linestring geometries that overlap with a 
    polygon using the provided index. The indices of geometries in 
    linestring_series should match those in idx. First, the indices of
    overlapping linestrings are found by intersecting the polygon's bounds
    with idx. These indices are then used to subset linestring_series.
    """

    candidates = list(idx.intersection(polygon.bounds))
    
    intersecting_lines = []
    for i in candidates:
        line_geom = linestrings_series.iloc[i].geometry
        if polygon.intersects(line_geom):
            intersecting_lines.append(line_geom)
    
    return MultiLineString(intersecting_lines).intersection(polygon) if intersecting_lines else None

def get_intersecting_lines(polygon: shapely.geometry.Polygon, 
                           linestrings_series: gpd.GeoSeries):
    """
    Efficiently finds all the linestring geometries that overlap with a 
    polygon using the provided index. The linestring series is subset to 
    only the linestrings that overlap the provided polygon. These are then
    merged to a MultiLineString geometry, which is cropped to within the 
    bounds of the polygon.
    """

    intersecting_lines = linestrings_series[linestrings_series.intersects(polygon)]
    return MultiLineString(intersecting_lines.values).intersection(polygon) if not intersecting_lines.empty else None


def get_intersecting_lines_index(polygon: shapely.geometry.Polygon, 
                                linestrings_gdf: gpd.GeoSeries,
                                idx: index.Index):
    """
    Efficiently finds all the linestring geometries that overlap with a 
    polygon using the provided index. The linestring series is subset to 
    only the linestrings that overlap the provided polygon. These are then
    merged to a MultiLineString geometry, which is cropped to within the 
    bounds of the polygon.
    """
    candidates = list(idx.intersection(polygon.bounds))
    
    intersecting_lines = []
    for i in candidates:
        line_geom = linestrings_gdf.iloc[i].geometry
        if polygon.intersects(line_geom):
            intersecting_lines.append(line_geom)
    
    return MultiLineString(intersecting_lines).intersection(polygon) if intersecting_lines else None

def generate_patches_gdf(linestrings_gdf=str | gpd.GeoDataFrame, 
                         start=0,
                         limit=-1,
                         dask=False,
                         use_idx=False,
                         partition_count: Optional[int] = None,
                         logger: logging.Logger = None) -> gpd.GeoDataFrame:
    if isinstance(linestrings_gdf, str):
        if logger:
            logger.log(msg=f"Loading GDF from {linestrings_gdf}", level=logging.DEBUG)
        linestrings_gdf = gpd.read_file(linestrings_gdf, engine="pyogrio", use_arrow=True)

    patches_gdf = create_patches(linestrings_gdf)[start:limit]
    if logger:  
        logger.log(msg="Patches GDF created.", level=logging.DEBUG)

    patches_gdf = prepare_gdf(patches_gdf)
    linestrings_gdf = prepare_gdf(linestrings_gdf)

    linestrings_series = linestrings_gdf.geometry

    if use_idx:
        if logger:
            logger.log(msg="Creating spatial index...", level=logging.DEBUG)

        idx = index.Index()
        for i, geom in enumerate(linestrings_gdf.geometry):
            idx.insert(i, geom.bounds)
        if logger:
            logger.log(msg="Spatial index created.", level=logging.DEBUG)

        apply_func = lambda x: get_intersecting_lines_index(x, linestrings_series, idx)

    else:
        apply_func = lambda x: get_intersecting_lines(x, linestrings_series)
    
    if dask:
        dask_patches = dask_geopandas.from_geopandas(patches_gdf, npartitions=partition_count)

        patch_hedgerows = dask_patches.geometry.apply(
            apply_func, meta=('geometry', 'geometry')
        ).compute()

    else:
        patch_hedgerows = patches_gdf.geometry.progress_apply(
            apply_func
        )
    patches_gdf["hedgerows"] = patch_hedgerows

    return patches_gdf

def create_and_save_patches_gdf(linestrings_gdf:str | gpd.GeoDataFrame,
                                save_path:str,
                                save_crs:str="EPSG:3857",
                                dropna_hedgerows:bool = True,
                                return_new_data_only:bool = True
                                ):
    patches_gdf = generate_patches_gdf(linestrings_gdf)
    if dropna_hedgerows:
        patches_gdf.dropna(how="any", inplace=True)
    
    patches_gdf.to_crs(save_crs, inplace=True)
    patches_gdf["hedgerows"] = gpd.GeoSeries(patches_gdf["hedgerows"]).to_crs(save_crs)

    patches_gdf.to_file(save_path)

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()

    parser.add_argument("--gdf_path", default=INITIAL_BDA_PATH)
    parser.add_argument("--save_path", default=PATCHES_GDF_PATH)
    parser.add_argument("--start", type=int, default=0, help="The index of the row to start from(defaults to the very first)")
    parser.add_argument("--limit", type=int, default=-1, help="The total number of rows to process(defaults to processing all rows)")

    parser.add_argument("--dask", action="store_true")
    parser.add_argument("--use_idx", action="store_true")
    parser.add_argument("--partition_count", type=int, default=16)

    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.DEBUG)
    patches_gdf = generate_patches_gdf(
        args.gdf_path, 
        start=args.start,
        limit=args.limit,
        dask=args.dask, 
        partition_count=args.partition_count,
        logger=logger)
    patches_gdf["hedgerows"] = gpd.GeoSeries(patches_gdf["hedgerows"]).to_wkt()
    patches_gdf.to_file(args.save_path)