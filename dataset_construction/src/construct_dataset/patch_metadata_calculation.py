import argparse
import math
import rasterio

import geopandas as gpd
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

PATCH_SIZE = os.getenv("PATCH_SIZE", 128)
AEZ_ROOT = os.getenv("AEZ_ROOT", "data/aez")


def geo_to_pixel(geo_x, geo_y, transform):
    """Convert geographic coordinates to pixel coordinates."""
    col = int((geo_x - transform.c) / transform.a)
    row = int((geo_y - transform.f) / transform.e)
    return row, col

def pixel_to_geo(row, col, transform):
    """Convert pixel coordinates to geographic coordinates."""
    geo_x = transform.c + col * transform.a
    geo_y = transform.f + row * transform.e
    return geo_x, geo_y

def extract_patches_from_geotiff(src, gdf_patches, ensure_window=True):
    if gdf_patches.crs != src.crs:
        gdf_patches = gdf_patches.to_crs(src.crs)
    
    patches_data = []
    
    for idx, patch in gdf_patches.iterrows():
        try:
            
            bounds = patch.geometry.bounds  
            row_min, col_min = geo_to_pixel(bounds[0], bounds[3], src.transform)
            row_max, col_max = geo_to_pixel(bounds[2], bounds[1], src.transform)
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform)

            if ensure_window:
                col_start = int(math.floor(window.col_off))
                row_start = int(math.floor(window.row_off))
                col_end = int(math.ceil(window.col_off + window.width))
                row_end = int(math.ceil(window.row_off + window.height))
                
                width = max(1, col_end - col_start)
                height = max(1, row_end - row_start)
                
                window = rasterio.windows.Window(col_off=col_start, row_off=row_start, width=width, height=height)
            window_img = src.read(1,window=window) 
            

            patches_data.append({
                'patch_id': idx,
                'pixel_data': window_img.flatten(),
                'pixel_bounds': (row_min, row_max, col_min, col_max),
                'geo_bounds': bounds,
            })
            
        except Exception as e:
            print(f"Error processing patch {idx}: {e}")
    
    return patches_data

def get_zone_classification(gdf, tif_path, clr_path):
    src = rasterio.open(tif_path) 

    pixel_data_df = pd.DataFrame(extract_patches_from_geotiff(src, gdf))

    class_index_to_class_label = {}
    with open(clr_path, "r") as infile:
        for line in infile:
            parts = line.strip().split()
            class_index_to_class_label[int(parts[0])] = " ".join(parts[5:])

    class_index_to_class_label[0] = "Unclassified"

    def get_majority_nonzero_element(arr, remove_zero=True):
        if remove_zero:
            arr = arr[arr != 0]
        if len(arr) == 0:
            return 0
        values, counts = np.unique(arr, return_counts=True)

        return values[np.argmax(counts)]

    classes = pixel_data_df["pixel_data"].apply(lambda x: get_majority_nonzero_element(x, remove_zero=True))
    classes = [class_index_to_class_label[c] for c in classes]
    return classes




def generate_class_gdf(gdf: gpd.GeoDataFrame,
         column_names: list[str],
         tif_paths: list[str],
         clr_paths: list[str],
         return_new_data_only:bool = False):
    
    if not (len(set([len(column_names), len(tif_paths), len(clr_paths)])) == 1):
        raise ValueError("All 3 passed lists must be of equal length.")
    
    for col, tif, clr in zip(column_names, tif_paths, clr_paths):
        classification = get_zone_classification(gdf, tif, clr)
        gdf[col] = classification
    if return_new_data_only:
        return gdf[["patch_id", *column_names]]
    return gdf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdf_path", 
                        default="data/gpkgs/haie_2020_france_patches_2024_merged.gpkg",
                        help="The path from which to read the initial GDF.")
    parser.add_argument("--save_path", 
                        default="test_data/gdf_classified.geojson",
                        help="The path to save the resulting GDF.")
    parser.add_argument("--column_names", 
                        nargs="+", 
                        default=[
                            "aez_class",
                            "mst_class",
                            "thz_class"
                        ],
                        help="The columns names to write the classification into.")
    
    parser.add_argument("--tif_paths", 
                        nargs="+", 
                        default=[
                            f"{AEZ_ROOT}/aez_v9v2_CRUTS32_Hist_8110_100_avg.tif",
                            f"{AEZ_ROOT}/mst_class_CRUTS32_Hist_8110_100_avg.tif",
                            f"{AEZ_ROOT}/thz_class_CRUTS32_Hist_8110_100_avg.tif"
                        ],
                        help="The paths for the .tif files used to classify GDF patches.")
    parser.add_argument("--clr_paths", 
                        nargs="+", 
                        default=[
                            f"{AEZ_ROOT}/AEZ_57classes.clr",
                            f"{AEZ_ROOT}/MoistureRegime_class.clr",
                            f"{AEZ_ROOT}/ThermalRegime_class.clr",
                        ],
                        help="The paths for the .clr files used to convert numerical classification into understandable labels.")
    
    args = parser.parse_args()

    gdf = gpd.read_file(args.gdf_path)

    classified_gdf = generate_class_gdf(
        gdf,
        column_names=args.column_names,
        tif_paths=args.tif_paths,
        clr_paths=args.clr_paths
    )

    classified_gdf.to_file(args.save_path)

    