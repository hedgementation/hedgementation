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

def get_hedgerow_pixel_count(id, 
                         y_pattern="y/y_{}.npy",
                          dataset_root=DATASET_ROOT,):
    y_path = os.path.join(dataset_root, y_pattern.format(id))
    arr = np.load(y_path)
    return np.sum(arr > 0)

def add_hedge_pixel_count(metadata: gpd.GeoDataFrame,
                          y_pattern: str = "y/y_{}.npy",
                          dataset_root=DATASET_ROOT):
    ids = metadata["ID_PATCH"]

    hedge_pixel_counts = [
        get_hedgerow_pixel_count(id, y_pattern=y_pattern, dataset_root=dataset_root) for id in ids
    ]

    metadata["hedge_pixel_count"] = hedge_pixel_counts

    return metadata