import argparse
import re
from datetime import datetime, timedelta

import os
import geopandas as gpd
import pandas as pd
import rasterio 
from dotenv import load_dotenv
import geopandas as gpd
import numpy as np


load_dotenv()

DATASET_ROOT = os.environ["DATASET_ROOT"]

METADATA_PATH = os.path.join(DATASET_ROOT, "metadata.geojson")

def band_names_to_dates(band_names):
    band_names = list(band_names)
    return sorted([
        fix_band_name(bn) for bn in band_names
    ])[::-1]



def fix_band_name(band_name):
    match = re.match(r'(\d{4})-(\d{2})-(\d+)(_.*)', band_name)
    if not match:
        return band_name  
    
    week_year, month, day_of_year, _ = match.groups()
    week_year = int(week_year)
    month = int(month)
    day_of_year = int(day_of_year)
    
    base_date = datetime(week_year, 1, 1)
    
    actual_date = base_date + timedelta(days=day_of_year - 1)
    
    correct_date = actual_date.strftime('%Y-%m-%d')
    
    return correct_date.replace("-","")

def get_dates_S2(ind, X_tif_pattern="Xtif/X_{}.tif"):
    full_path = os.path.join(DATASET_ROOT, X_tif_pattern.format(ind))

    src = rasterio.open(full_path)

    dates_s2 = band_names_to_dates(src.descriptions)  

    if len(dates_s2) != len(src.descriptions) // 10:
        raise Exception(f"Error for {ind}: dates is {len(dates_s2)}, descriptions / 10 is {len(src.descriptions) // 10}") 

    else:
        return {str(i):int(fix_band_name(d)) for i,d in enumerate(dates_s2)}



def add_dates_S2(metadata: gpd.GeoDataFrame,
        X_path: str ="Xtif/X_{}.tif"):
    ids = metadata["ID_PATCH"]

    s2_dates = [get_dates_S2(id, X_path) for id in ids]

    metadata["dates-S2"] = s2_dates

    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--metadata-path", default=METADATA_PATH, help="Path to read metadata geodataframe from.")
    parser.add_argument("--save-path", default=None, help="Path to save resulting metadata frame. Defaults to the path it was read from.")
    parser.add_argument("--X-tif-path", default=None, help="Pattern to fill in order to find the associated .tif file for a given patch.")
    parser.add_argument("--no-save", action="store_true", help="Flag to disable saving the modified metadata.")

    args = parser.parse_args()

    metadata_path = args.metadata_path
    save_path = args.save_path if args.save_path is not None else metadata_path
    X_path = args.X_path
    save = not args.no_save

    metadata = gpd.read_file(metadata_path, use_arrow=True, engine="pyogrio")

    metadata = add_dates_S2(metadata, X_path)

    if save:
        metadata.to_file(save_path)

