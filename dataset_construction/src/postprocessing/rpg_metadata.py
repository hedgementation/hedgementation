"""
Post-processing pipeline for computing pixel-level RPG metadata.

Requires the exported Y (hedgerow) rasters and RPG masks to already be
downloaded to local disk. Run this after the main export pipeline completes.
"""
import argparse
import os

import numpy as np
import pandas as pd
import geopandas as gpd

from src.utils.io_utils import load_hedgerow_raster, load_rpg_mask

from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = os.environ["DATASET_ROOT"]

def calculate_masked_and_total_pixels(mask: np.ndarray) -> tuple[int, int]:
    masked_pixels = int(np.sum(mask))
    total_pixels = int(np.prod(mask.shape))
    return masked_pixels, total_pixels


def calculate_masked_and_total_hedgerow_pixels(
        mask: np.ndarray,
        hedgerow_raster: np.ndarray) -> tuple[int, int]:
    masked_hedgerow_pixels = int(np.sum(mask * hedgerow_raster))
    total_hedgerow_pixels = int(np.sum(hedgerow_raster))
    return masked_hedgerow_pixels, total_hedgerow_pixels


def calculate_rpg_metadata(metadata: gpd.GeoDataFrame,
                                   mask_dir: str,
                                   hedge_dir: str,) -> pd.DataFrame:
    """
    Compute per-patch pixel-level statistics from exported RPG masks and
    hedgerow rasters.

    Args:
        datapoint_count: Number of patches (IDs 0..datapoint_count-1).
        mask_dir: Directory containing ``{id}_rpg.tif`` files.
        hedge_dir: Directory containing ``{id}_Y.tif`` files.
        save_metadata: If True, write results to ``{mask_dir}/mask_metadata.csv``.

    Returns:
        DataFrame with one row per patch.
    """
    rows = []
    ids = metadata["ID_PATCH"]
    for i in ids:
        rpg_mask = load_rpg_mask(identifier=i, mask_dir=mask_dir)
        hedgerow_raster = load_hedgerow_raster(identifier=i, hedge_dir=hedge_dir)

        masked_pixels, total_pixels = calculate_masked_and_total_pixels(rpg_mask)
        masked_hedge, total_hedge = calculate_masked_and_total_hedgerow_pixels(rpg_mask, hedgerow_raster)

        rows.append({
            "ID_PATCH": i,
            "masked_pixels": masked_pixels,
            "masked_pixel_percent": masked_pixels / total_pixels,
            "masked_hedge_pixels": masked_hedge,
            "masked_hedge_percent": masked_hedge / total_hedge if total_hedge > 0 else 0.0,
        })

    df = pd.DataFrame(data=rows)
    
    metadata = pd.merge(
        left=metadata,
        right=df,
        how="inner",
        on="ID_PATCH"
    )
    
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute pixel-level RPG metadata from exported masks and hedgerow rasters."
    )
    parser.add_arguemnt("--metadata-path",
                        default=os.path.join(DATASET_ROOT, "metadata.geojson"),
                        help="Filepath to read metadata from.")
    parser.add_argument("--save-path",
                        default=None,
                        help="Path to save the resulting metadata to. If not set, defaults to overwriting the path the metadata was read from.")
    parser.add_arguemnt("--dataset-root",
                        default=DATASET_ROOT,
                        help="The root of the dataset.")
    parser.add_argument("--mask-dir", default="rpg_masks_unbuffered/mask_{}.npy",
                        help="The directory to load RPG masks from(defaults to '{DATASET_ROOT}/rpg_mask_unbuffered'})")
    parser.add_argument("--y-dir", default=os.path.join(DATASET_ROOT, "y"),
                        help="The directory to load hedgerow rasters from(defaults to '{DATASET_ROOT}/y'})")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving the metadata resulting from the computation.")

    args = parser.parse_args()
    metadata = gpd.read_file(args.metadata_path)
    save_path = args.save_path if args.save_path is not None else args.metadata_path

    dataset_root = args.dataset_root
    mask_pattern = args.mask_pattern
    y_pattern = args.y_pattern

    metadata = calculate_rpg_metadata(
        metadata=metadata,
        mask_dir=args.mask_dir,
        hedge_dir=args.hedge_dir,
        save_metadata=not args.no_save,
    )
    if not args.no_save:
        metadata.to_file(save_path)