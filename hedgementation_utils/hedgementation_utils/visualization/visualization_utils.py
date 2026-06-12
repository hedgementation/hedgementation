from typing import Optional
import numpy as np
import os
from dotenv import load_dotenv
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.axes
from shapely import affinity
import torch
load_dotenv()
import matplotlib.colors as mcolors
import functools

import shapely

RANDOM_SEED = int(os.environ["RANDOM_SEED"])

PROJECT_ID = "hedgementation"

import geopandas as gpd

from hedgementation_utils.io.io_manager import HedgementationIOManager

DATASET_ROOT = os.getenv("DATASET_ROOT")
RPG_MASK_SUBDIR = os.getenv("RPG_MASK_SUBDIR")

DEFAULT_PREDICTION_CMAP = {
    "tp": "blue",
    "fn": "yellow",
    "fp": "magenta"
}

_default_manager: Optional[HedgementationIOManager] = None

def get_io_manager() -> HedgementationIOManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = HedgementationIOManager(dataset_root=DATASET_ROOT)
    return _default_manager

def get_object_to_plot(obj,
                       patch_id,
                       get_func):
    """
    Helper function to load plot targets from the dataset if they are not provided.
    """

    if obj is None:
        if not patch_id:
            raise ValueError("You have to provide either something to plot or an index to fetch it with.")
        obj = get_func(patch_id)
    return obj


def get_satellite_img(patch_id, manager: Optional[HedgementationIOManager] = None):
    if manager is None:
        manager = get_io_manager()
    patch = manager.load_X(identifier=patch_id)
    return patch[0]

def get_hedge_raster(patch_id, manager: Optional[HedgementationIOManager] = None):
    if manager is None:
        manager = get_io_manager()
    raster = manager.load_y(identifier=patch_id)
    return raster > 0

def get_rpg_raster(patch_id, manager: Optional[HedgementationIOManager] = None):
    if manager is None:
        manager = get_io_manager()
    return manager.load_rpg_mask(identifier=patch_id)

def get_multilinestring(patch_id, y_pixels, x_pixels):
    gdf = gpd.read_file(f"{DATASET_ROOT}/metadata.geojson")
    geom = shapely.wkt.loads(gdf.iloc[patch_id]["hedgerows"])
    mls = shapely.intersection(geom, gdf.iloc[patch_id]["geometry"])
    mls = rescale_geometry_to_pixels(mls, mls.bounds, y_pixels, x_pixels)

    return mls

def show_satellite_img(ax: matplotlib.axes.Axes,
                    patch_img: Optional[torch.Tensor] = None,
                    patch_id: Optional[int] = None,
                    grayscale: bool = False,
                    manager: Optional[HedgementationIOManager] = None):
    """
    Code to display a satellite image on an axis.


    :param ax: Axis on which to plot the raster.
    :param patch_img: The satellite image of the patch. Expected as a raw, multi-channel image where the first 3 channels are
    blue, green and red respectively.
    :param patch_id: patch_id used to load the satellite image from the dataset if it is not provided.
    :param alpha: Level of transparency to plot the raster mask with.
    :param manager: IO manager to use for loading. Defaults to a singleton initialized with DATASET_ROOT.
    """

    patch_img = np.array(get_object_to_plot(patch_img, patch_id, lambda pid: get_satellite_img(pid, manager=manager)))

    r = (patch_img[2,:] / np.max(patch_img[2,:,:])) #* 1.5
    g = (patch_img[1,:] / np.max(patch_img[1,:,:])) #* 1.5
    b = (patch_img[0,:] / np.max(patch_img[0,:,:])) #* 1.5
    a = np.ones(r.shape)

    img = np.stack([r,g,b,a],axis=-1)
    if grayscale:
        img = np.mean(img, axis=2)
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)

def show_hedge_raster(
        ax: matplotlib.axes.Axes,
        raster: Optional[torch.Tensor | np.ndarray]=None,
        patch_id: Optional[int]=None,
        color: str = "blue",
        manager: Optional[HedgementationIOManager] = None):
    raster = get_object_to_plot(raster, patch_id, lambda pid: get_hedge_raster(pid, manager=manager))
       
    rgb_raster = np.zeros((*raster.shape, 4))
    rgb_raster[raster > 0] = mcolors.to_rgba(color)
    ax.imshow(rgb_raster)

def show_rpg_raster(
        ax: matplotlib.axes.Axes,
        raster: Optional[np.ndarray]=None,
        patch_id: Optional[int]=None,
        alpha:float=0.5,
        manager: Optional[HedgementationIOManager] = None):
    """
    Code to display a multilinestring geometry overtop of an axis.

    :param ax: Axis on which to plot the raster.
    :param raster: Raster mask to plot. Expected as a 2-dimensional numpy array.
    :param patch_id: patch_id to use as a fallback to load the raster mask in the case it is not provided.
    :param alpha: Level of transparency to plot the raster mask with.
    :param manager: IO manager to use for loading. Defaults to a singleton initialized with DATASET_ROOT.
    """
    raster = get_object_to_plot(raster, patch_id, lambda pid: get_rpg_raster(pid, manager=manager))
    alpha = raster.astype(float) * alpha
    ax.imshow(raster, alpha=alpha)

def show_multilinestring(ax: matplotlib.axes.Axes,
                         mls: Optional[shapely.Geometry]=None,
                         patch_id: Optional[int]=None,
                         y_pixels=128,
                         x_pixels=128,
                         color: str ="blue"):
    """
    Code to display a multilinestring geometry overtop of an axis.
    
    :param ax: Axis on which to plot the geometry.
    :param mls: Geometry object to plot. If not provided, it will be loaded from the metadata using patch_id.
    :param patch_id: patch_id to use as a fallback to load the multilinestring in the case it is not provided.
    :param y_pixels: Height of the image in pixels, used for rescaling the image.
    :param x_pixels: Height of the image in pixels, used for rescaling the image.
    :param color: Color to display the image in.
    """
    mls = get_object_to_plot(mls, patch_id, lambda x: get_multilinestring(x, y_height=y_pixels, x_pixels=x_pixels))
        
    for geom in mls.geoms:
        x, y = geom.xy
        ax.plot(x, y, linewidth=5, color=color)


def visualize_prediction(ax: matplotlib.axes.Axes,
                        preds: torch.Tensor|np.ndarray,
                        labels: torch.Tensor|np.ndarray,
                        show_satellite_img_beneath: bool=True,
                        patch_satellite_img: Optional[torch.Tensor] = None,
                        patch_id: Optional[int] = None,
                        grayscale=True,
                        prediction_cmaps: Optional[dict[str,str]]=None,
                        manager: Optional[HedgementationIOManager] = None
                        ):
    """
    Helper for visualizing a model's predictions versus ground truth overlaid on the underlying satellite image. 
    Excepts both the predictions(preds) and ground truth(labels) to be formatted as 2-dimensional binary torch
    tensors of the same height and width. 
    
    :param ax: The matplotlib Axes object onto which the prediction will be plotted.
    :param preds: A 2D tensor or numpy array cotaining binary predictions made by the model. Must be the same shape as labels.
    :param labels: A 2D tensor or numpy array containing binary ground truth labels. Must be the same shape as preds.
    :param show_satellite_img_beneath: A binary flag indicating whether or not to display the satellite image of the patch
    belowthe predictions.
    :param patch_id: The patch ID to use to display the satellite image. Only used if the patch_satellite_image is not provided.
    :param patch_satellite_image: A torch tensor representing the satellite image to display. If not provided, the image will be
    loaded from the dataset using the patch_id. Expected to be a 3-dimensional tensor with dimensions in [channel, width height]
    order.
    :param grayscale: whether to show the underlying image in grayscale or full RBG color.
    :param prediction_cmaps: A dictionary mapping types of pixels to the colors used to plot them. Should have values for
    "fp"(false positive), "tp"(true positive) and "fn"(fal)
    """

    if not prediction_cmaps:
        prediction_cmaps = DEFAULT_PREDICTION_CMAP
    if show_satellite_img_beneath:
        show_satellite_img(ax, patch_id=patch_id, patch_img=patch_satellite_img, grayscale=grayscale, manager=manager)
    print(preds.shape)
    print(labels.shape)
    tp = preds * labels
    fp = torch.subtract(preds.int(), labels.int())
    fn = torch.subtract(labels.int(), preds.int())

    show_hedge_raster(ax, raster=tp.numpy(), color=prediction_cmaps["tp"])
    show_hedge_raster(ax, raster=fp.numpy(), color=prediction_cmaps["fp"])
    show_hedge_raster(ax, raster=fn.numpy(), color=prediction_cmaps["fn"])




def rescale_geometry_to_pixels(geom, original_bounds, y_pixels=128, x_pixels=128):
    minx, miny, maxx, maxy = original_bounds
    
    width = maxx - minx
    height = maxy - miny
    
    geom_translated = affinity.translate(geom, xoff=-minx, yoff=-miny)
    
    scale_x = y_pixels / width
    scale_y = x_pixels / height
    
    geom_scaled = affinity.scale(geom_translated, xfact=scale_x, yfact=scale_y, origin=(0, 0))
    
    geom_flipped = affinity.scale(geom_scaled, xfact=1, yfact=-1, origin=(0, 0))
    geom_flipped = affinity.translate(geom_flipped, yoff=y_pixels)
    
    return geom_flipped

def plot_for_ind(ax, 
                ind,
                gdf=None,
                show=None):
    if gdf is None:
        gdf = gpd.read_file(f"{DATASET_ROOT}/metadata.geojson")

    show_satellite_img(ax,patch_id=ind)

    if show is not None:
        for option in show:
            if option == "rpg_mask":
                show_rpg_raster(ax, patch_id=ind)
                    
            elif option == "raster":
                show_hedge_raster(ax, patch_id=ind)

            elif option == "geom":
                show_multilinestring(ax, patch_id=ind)
            
        
        
def plot_for_rpg_mask_validation(ind, gdf=None):
    fig, axs = plt.subplots(nrows=1, ncols=3)
    fig.set_figheight(12)
    fig.set_figwidth(36)

    plot_for_ind(axs[0], ind, gdf, show=None)
    plot_for_ind(axs[1], ind, gdf, show=["rpg_mask"])
    plot_for_ind(axs[2], ind, gdf, show=["raster","rpg_mask"])