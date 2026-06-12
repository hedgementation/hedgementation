from datetime import datetime
from functools import partial
import json
import os
import geopandas as gpd
from matplotlib import pyplot as plt
import pandas as pd
import shapely 
import planet
from planet import Planet
from planet import data_filter
from PIL import Image

import rasterio
import numpy as np

import rasterio
import rioxarray as rxr
from rasterio.warp import calculate_default_transform, reproject, Resampling

from rasterio.plot import reshape_as_image
import asyncio

from dotenv import load_dotenv
import torch

from backbones.unet3d import UNet3D
from backbones.utae import UTAE
from src.training.hedgementation_dataset import HedgementationDataset
from src.training.transforms import Normalization, YTransform, compound_transform
load_dotenv()

DATASET_ROOT = os.environ["DATASET_ROOT"]
NUM_CHANNELS = int(os.environ["NUM_CHANNELS"])


PL_API_KEY = os.getenv("PL_API_KEY")

plsdk_auth = planet.Auth.from_key(key=PL_API_KEY)
sess = planet.Session(plsdk_auth)

pl = Planet(sess)


def to_crs(result,id,dst_crs="EPSG:3857"):


    with rasterio.open(result) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        
        temp_path = f'planet/{id}.byte.wgs84.tif'
        with rasterio.open(temp_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
        return temp_path


def to_4326(result,id):
    return to_crs(result, id, dst_crs="EPSG:4326")

async def download_and_validate_async(
        session,
        item_id,
        asset_type_id="ortho_visual",
        item_type_id="PSScene"
):
    cl = session.client('data')
  
    asset = await cl.get_asset(item_type_id, item_id, asset_type_id)
    await cl.activate_asset(asset)

    asset = await cl.wait_asset(asset, callback=print)
    path = await cl.download_asset(asset)
    cl.validate_checksum(asset, path)

    return path



async def download_single_image(item_id,item_type_id,asset_type_id,dst_crs="EPSG:4326"):
    session = planet.Session(plsdk_auth)

    path = await download_and_validate_async(session, item_id, asset_type_id, item_type_id)
    result = to_crs(path, id=item_id, dst_crs=dst_crs)
    return result

def cut_img_to_within_bounds(src, poly, hflip=False):
    if isinstance(src, str):
        src = rasterio.open(src)
    img_bounds = src.bounds
    poly_bounds = poly.bounds
    poly_bounds = rasterio.coords.BoundingBox(poly_bounds[0], poly_bounds[1], poly_bounds[2], poly_bounds[3])

    if hflip:
        img = reshape_as_image(np.fliplr(src.read()))
    else:
        img = reshape_as_image(src.read())

    img_y = img.shape[0]
    img_x = img.shape[1]

    x_left = int(img_x * max(0, (poly_bounds.left - img_bounds.left) / (img_bounds.right - img_bounds.left)))
    x_right = int(img_x * min(1, (poly_bounds.right - img_bounds.left) / (img_bounds.right - img_bounds.left)))

    y_top = int(img_y * max(0, (img_bounds.top - poly_bounds.top) / (img_bounds.top - img_bounds.bottom)))
    y_bottom = int(img_y * min(1, (img_bounds.top - poly_bounds.bottom) / (img_bounds.top - img_bounds.bottom)))
    
    new_img = img[y_top:y_bottom, x_left:x_right, :].astype(np.float64)

    for i in range(new_img.shape[-1]):
        max_val = np.max(new_img[:, :, i])
        if max_val > 0:  
            new_img[:, :, i] /= max_val

    new_left = img_bounds.left + (x_left / img_x) * (img_bounds.right - img_bounds.left)
    new_right = img_bounds.left + (x_right / img_x) * (img_bounds.right - img_bounds.left)
    new_top = img_bounds.top - (y_top / img_y) * (img_bounds.top - img_bounds.bottom)
    new_bottom = img_bounds.top - (y_bottom / img_y) * (img_bounds.top - img_bounds.bottom)
    
    new_bounds = rasterio.coords.BoundingBox(new_left, new_bottom, new_right, new_top)

    return new_img, new_bounds

def filter_results_by_overlap(row, results):
    row_poly = row["geometry"]
    filtered_r = [r for r in results if data_api_results_to_poly(r).intersects(row_poly)]
    filtered_r.sort(
        key=lambda x: shapely.intersection(data_api_results_to_poly(x), row_poly).area,
        reverse=True
    )

    return filtered_r

def has_large_overlapping_result(row, result_cols, threshold=0.9):
    for col in result_cols:
        res = row[col]
        row_poly = row["geometry"]
        res_poly = data_api_results_to_poly(res)
        
        if (shapely.intersection(row_poly, res_poly).area / row_poly.area) < threshold:
            return False
    return True

def data_api_results_to_poly(res):
    if isinstance(res, list):
        coords = [obs["geometry"]["coordinates"] for obs in res]
        coords_poly = shapely.geometry.MultiPolygon(coords)
        return coords_poly
    else:
        return shapely.geometry.Polygon(res["geometry"]["coordinates"][0])

def data_result_to_polygon(res):
    return shapely.geometry.Polygon(res["geometry"]["coordinates"][0])

def conduct_search(start_date, 
                   end_date, 
                   poly,
                   limit=100,
                   cloud_cover_max=15,
                   light_haze_max=15,
                   clear_percent_min=90,
                   visible_confidence_percent_min=90):
    acquired_filter = data_filter.date_range_filter(
        field_name="acquired",
        gte=datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ"),
        lte=datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
    )

    updated_filter = data_filter.date_range_filter(
        field_name="acquired",
        gte=datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ"),
        lte=datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
    )

    or_filter = data_filter.or_filter(
        [
            updated_filter,
            acquired_filter
        ]
    )

    cloud_filter = data_filter.range_filter(
        field_name="cloud_percent",
        lt=cloud_cover_max
    )

    haze_filter = data_filter.range_filter(
        field_name="light_haze_percent",
        lt=light_haze_max
    )

    clear_filter = data_filter.range_filter(
        field_name="clear_percent",
        gt=clear_percent_min
    )

    visible_confidence_filter = data_filter.range_filter(
        field_name="visible_confidence_percent",
        gt=visible_confidence_percent_min
    )

    and_filter = data_filter.and_filter(
        [
            or_filter,
            cloud_filter,
            haze_filter,
            clear_filter,
            visible_confidence_filter
        ]
    )

    retval = pl.data.search(
        item_types=["SkySatScene"],
        search_filter=and_filter,
        geometry=poly,
        limit=limit
    )
    return retval


def take_high_area_sample(gdf, change_class, area_key="area_2024", quantile=0.9, sample_count=None):
    sub = gdf[gdf["area_change_class"] == change_class]
    sub = gdf[gdf[area_key] >= gdf[area_key].quantile(quantile)]
    if sample_count:    
        return sub.sample(sample_count)
    return sub

def get_single_prediction(id,model,params):
    metadata_frame = pd.DataFrame({"id":[id], "ID_PATCH":[id]})

    dataloader = torch.utils.data.DataLoader(
        HedgementationDataset(metadata_frame=metadata_frame, 
                    root_dir=DATASET_ROOT, 
                    constant_length=10, 
                    transform=partial(compound_transform, 
                                        num_buckets=params["num buckets"],
                                        normalization=Normalization(params["normalization strategy"]),
                                        y_transform=YTransform(params["y transform strategy"]))),
        batch_size=16,
        shuffle=True,
        num_workers=4,  
    )

    for (X,date),y in dataloader:
        y_true_viz = y.squeeze()
        res = model(X,date).squeeze()
        y_pred_viz = torch.argmax(res, dim=0).cpu()

    return y_true_viz, y_pred_viz

def show_pred_on_img(path, row, y_true_viz, y_pred_viz, 
                     k_rot=0, 
                     show_labels=True,
                     show_preds=True,
                     hedgerows=None):
    with rasterio.open(path, "r") as src:
        poly = row["geometry"]


        #array, _ = cut_img_to_within_bounds(src, poly, hflip=True)
        bounds = poly.bounds
        window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
        array = src.read(window=window).transpose(1, 2, 0)
        

        overlap_viz = (y_true_viz == y_pred_viz) & (y_true_viz > 0)

        y_true_rgba = np.zeros((*y_true_viz.shape, 4))
        y_true_rgba[y_true_viz > 0] = [0, 0, 1, 0.6]  
        y_pred_rgba = np.zeros((*y_pred_viz.shape, 4))
        y_pred_rgba[y_pred_viz > 0] = [1, 0, 0, 0.6]  

        overlap_viz_rgba = np.zeros((*overlap_viz.shape, 4))
        overlap_viz_rgba[overlap_viz.cpu().numpy()] = [0.8, 0, 0.8, 0.6]  

        def resize_to_arr(img, planet_img, k_rot):
            img_rotated = np.rot90(img, k=k_rot)
            img_pil = Image.fromarray((img_rotated * 255).astype(np.uint8))
            target_size = (planet_img.shape[1], planet_img.shape[0])  
            img_pil = img_pil.resize(target_size, Image.Resampling.LANCZOS)
            return np.asarray(img_pil) / 255.0

        y_true_rgba = resize_to_arr(y_true_rgba, array, k_rot)
        y_pred_rgba = resize_to_arr(y_pred_rgba, array, k_rot)
        overlap_viz_rgba = resize_to_arr(overlap_viz_rgba, array, k_rot)

        _, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(array, origin="upper")
        if show_labels:
            ax.imshow(y_true_rgba, interpolation='nearest', origin='upper')

        if show_preds:
            ax.imshow(y_pred_rgba, interpolation='nearest', origin='upper')
            ax.imshow(overlap_viz_rgba, interpolation='nearest', origin='upper')
        ax.set_title(f"ID: {row.name}")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        if hedgerows:
            for line in hedgerows.geoms:
                
                hedge_x,hedge_y = line.xy
       

                transform = rasterio.transform.from_bounds(*bounds, array.shape[1], array.shape[0])
                new_hedge_y, new_hedge_x = rasterio.transform.rowcol(transform, hedge_x, hedge_y)
                plt.plot(new_hedge_x, new_hedge_y, color="red", linewidth=4,alpha=0.5)
        plt.show()

async def visualize(row,model,params,
                    hedgerows=None,
                    show_underlying_image=False,
                    show_with_labels_only=False):
    id = row["best_results"]["id"]
    asset_type_id="ortho_visual"
    item_type_id="SkySatScene"


    path = await download_single_image(
        id, 
        item_type_id,
        asset_type_id,
        dst_crs="EPSG:4326"
    )

    y_true_viz, y_pred_viz = get_single_prediction(row["ID_PATCH"],model,params)
    if show_underlying_image:
        show_pred_on_img(
            path, 
            row,
            y_true_viz,
            y_pred_viz,
            show_labels=False,
            show_preds=False,
            hedgerows=hedgerows
        )

    if show_with_labels_only:
        show_pred_on_img(
            path, 
            row,
            y_true_viz,
            y_pred_viz,
            show_labels=True,
            show_preds=False,
            hedgerows=hedgerows
        )

    show_pred_on_img(
        path, 
        row,
        y_true_viz,
        y_pred_viz,
        show_labels=True,
        show_preds=True,
        hedgerows=hedgerows
    )

async def gdf_visualization(
                      model_dir,
                      gdf=f"{DATASET_ROOT}/metadata.geojson",
                      threshold=0.9,
                      num_samples=5,
                      pick_random=False,
                      show_underlying_image=False,
                      show_with_labels_only=False):
    if isinstance(gdf, str):
        gdf = gpd.read_file(gdf)

    with open(f"{model_dir}/params.json", "r+") as infile:
        params = json.load(infile) 

    if params["backbone"] == "UTAE":
        model = UTAE(NUM_CHANNELS, pad_value=0)
    else:
        model = UNet3D(in_channel=NUM_CHANNELS, n_classes=params["num buckets"] + 1, pad_value=0)
    weights = torch.load(f"{model_dir}/checkpoints/best_model_params.pt")
    model.load_state_dict(weights)

    start_bda_2024 = f"2021-01-01T00:00:00Z"
    end_bda_2024 = f"2021-12-31T00:00:00Z"
    collective_bounds = shapely.unary_union(
        list(gdf["geometry"])
    )
    results = conduct_search(
        start_date=start_bda_2024,
        end_date=end_bda_2024,
        poly=collective_bounds.__geo_interface__
    )
    results = list(results)
    poly_results = data_api_results_to_poly(results)


    overlapping_gdf = gdf[
        gdf.intersects(poly_results) 
    ]

    best_results = []

    for _, row in overlapping_gdf.iterrows():
        best_results.append(filter_results_by_overlap(row, results)[0])

    overlapping_gdf["best_results"] = best_results

    overlapping_gdf = overlapping_gdf[overlapping_gdf.apply(lambda x: has_large_overlapping_result(x, ["best_results"], threshold=0.9), axis=1)]

    
    if len(overlapping_gdf) == 0:
        print(f"No rows with threshold {threshold}")
        return
    
    overlapping_gdf["result_overlap"] = overlapping_gdf.apply(lambda x: shapely.intersection(data_result_to_polygon(x["best_results"]), x["geometry"]).area, axis=1)

    overlapping_gdf = overlapping_gdf.sort_values(by="result_overlap",ascending=False)
    overlapping_gdf = overlapping_gdf.drop(columns=["result_overlap"])
    
    num_to_plot = min(num_samples, len(overlapping_gdf))

    if pick_random:
        overlapping_gdf = overlapping_gdf.sample(num_to_plot)
    for i in range(num_to_plot):
        print(f"\nVisualizing row {i+1}/{num_to_plot}")
        await visualize(overlapping_gdf.iloc[i],
                        model,
                        params,
                        show_underlying_image=show_underlying_image,
                        show_with_labels_only=show_with_labels_only)

