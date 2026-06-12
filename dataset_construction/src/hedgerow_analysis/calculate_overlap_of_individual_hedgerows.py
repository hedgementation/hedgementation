# %%
import rasterio
import rasterio.plot
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
tqdm.pandas()

import geopandas as gpd
import ee
import geemap
from shapely import wkt
import pandas as pd
import shapely 
from pyproj import Transformer
import geopandas as gpd



# %%

cloud_project = "hedgementation"

try:
    ee.Initialize(project=cloud_project)
except:
    ee.Authenticate()
    ee.Initialize(project=cloud_project)


# %%
try:
    gdf_2024 = gpd.GeoDataFrame.from_file(f"../../gdfs/haie_2024.json", engine='pyogrio', use_arrow=True)
    #gdf_2024["geometry"] = gdf_2024["geometry"].apply(wkt.loads)
    #gdf_2024 = gpd.GeoDataFrame(gdf_2024, crs="4326")
except:
    gdf_2024 = gpd.GeoDataFrame.from_file(f"../../csvs/haie_2024.csv", engine='pyogrio', use_arrow=True)
    gdf_2024["geometry"] = gdf_2024["geometry"].apply(wkt.loads)
    gdf_2024 = gpd.GeoDataFrame(gdf_2024, crs="4326")
    #gdf_2024.to_file(f"../gdf/haie_2024/json")

# %%

gdf_2020 = gpd.GeoDataFrame.from_file(f"../gdfs/haie_2020.json", engine='pyogrio', use_arrow=True)
gdf_2020 = gpd.GeoDataFrame(gdf_2020, crs="4326")
#gdf_2020["geometry"] = gdf_2020["geometry"].apply(wkt.loads)
#gdf_2020 = gpd.GeoDataFrame(gdf_2020, crs="4326")



# %%
gdf_2020["cleabs"]

# %%
gdf_2024["cleabs"]

# %%
gdf_2020.crs, gdf_2024.crs

# %%
len(gdf_2024), len(gdf_2020)

# %%
gdf_2024.index

# %%
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.plotting import plot_line
from typing import Tuple

def plot_random_linestring_pairs(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, 
                                 figsize: Tuple[int, int] = (15, 10), 
                                 random_state: int = None) -> None:
    
    # Validate inputs
    if not isinstance(gdf1, gpd.GeoDataFrame) or not isinstance(gdf2, gpd.GeoDataFrame):
        raise TypeError("Both inputs must be GeoDataFrames")
    
    if len(gdf1) == 0 or len(gdf2) == 0:
        raise ValueError("GeoDataFrames cannot be empty")
    
    # Check for LineString geometries (basic check on first few rows)
    sample_size = min(5, len(gdf1), len(gdf2))
    for i in range(sample_size):
        if gdf1.iloc[i].geometry.geom_type != 'LineString':
            print(f"Warning: gdf1 row {i} is not a LineString ({gdf1.iloc[i].geometry.geom_type})")
        if gdf2.iloc[i].geometry.geom_type != 'LineString':
            print(f"Warning: gdf2 row {i} is not a LineString ({gdf2.iloc[i].geometry.geom_type})")
    
    common_indices = gdf1["cleabs"][gdf1["cleabs"].isin(gdf2["cleabs"])] 
    
    if len(common_indices) == 0:
        raise ValueError("No common indices found between the two GeoDataFrames")
    
    if len(common_indices) < 9:
        print(f"Warning: Only {len(common_indices)} common indices available, using all of them")
        selected_indices = common_indices
    else:
        # Randomly select 9 indices
        if random_state is not None:
            np.random.seed(random_state)
        selected_indices = np.random.choice(common_indices, size=9, replace=False)
    n_plots = len(selected_indices)
    rows = int(np.ceil(n_plots / 3))
    _, axes = plt.subplots(rows, 3, figsize=figsize)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    for i, idx in enumerate(selected_indices):
        ax = axes_flat[i]
        
        gdf1_line = gdf1[gdf1["cleabs"] == idx].iloc[0].geometry
        gdf2_line = gdf2[gdf2["cleabs"] == idx].iloc[0].geometry
        plot_line(gdf1_line, ax=ax, color='blue', linewidth=2, label='GDF1', alpha=0.7)
        plot_line(gdf2_line, ax=ax, color='red', linewidth=2, label='GDF2', alpha=0.7)
        
        ax.set_title(f'Index: {idx}', fontsize=10)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Make axis labels smaller
        ax.tick_params(labelsize=8)
    
    # Hide unused subplots
    for i in range(len(selected_indices), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.suptitle('Random Selection of LineString Pairs', fontsize=16, y=1.02)
    plt.show()

plot_random_linestring_pairs(gdf_2020, gdf_2024)

# %%
joined = gdf_2020.merge(gdf_2024,on="cleabs",how="inner")[["cleabs", "geometry_x", "geometry_y"]]

# %%
joined = joined.rename({
    "geometry_x": "geometry_2020",
    "geometry_y": "geometry_2024"
},axis=1)
joined.columns

# %%
transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)

def find_iou(row):
    hedgerows_2024 = shapely.buffer(shapely.ops.transform(transformer.transform, row["geometry_2024"]), 3.5)
    hedgerows_2020 = shapely.buffer(shapely.ops.transform(transformer.transform, row["geometry_2020"]), 3.5)

    intersection = shapely.area(shapely.intersection(hedgerows_2020, hedgerows_2024))
    union = shapely.area(shapely.union(hedgerows_2020, hedgerows_2024))
    iou = intersection / union

    area_2020 = shapely.area(hedgerows_2020)
    area_2024 = shapely.area(hedgerows_2024)
    net_change = area_2024 - area_2020
    change_rate = ((area_2024 - area_2020) / area_2020) * 100
 
    return pd.Series([
        area_2020, area_2024, net_change, change_rate,
        intersection, union, iou
    ], index=[
        "area_2020", "area_2024", "net_change", "change_rate",
        "intersection", "union", "iou"
    ])


amt = 100000
joined_result = joined.sample(n=amt).progress_apply(find_iou, axis="columns", result_type="expand")


# %%
joined_result.to_csv(f"../csvs/change_analysis_by_hedgerow_{amt}.csv")


