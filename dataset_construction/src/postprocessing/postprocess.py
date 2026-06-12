import argparse
import logging
import tempfile
import geopandas as gpd
from dotenv import load_dotenv
import os

from src.postprocessing.hedge_metadata import add_hedge_pixel_count
from src.postprocessing.rpg_metadata import calculate_rpg_metadata
from src.postprocessing.s2_dates import add_dates_S2

from src.utils.drive_utils import get_drive_service, upload_file_to_drive

load_dotenv()
DATASET_ROOT = os.environ["DATASET_ROOT"]


def main(
        metadata: gpd.GeoDataFrame,
        dataset_root:str,
        mask_dir:str,
        y_dir:str,
        y_pattern:str,
        X_tif_pattern:str,
        skip_rpg_metadata:bool,
        skip_hedge_metadata:bool,
        skip_s2_dates:bool
):
    if not skip_hedge_metadata:
        metadata = add_hedge_pixel_count(
            metadata=metadata,
            y_pattern=y_pattern,
            dataset_root=dataset_root
        )
    if not skip_rpg_metadata:
        metadata = calculate_rpg_metadata(
            metadata=metadata,
            mask_dir=mask_dir,
            hedge_dir=y_dir
        )
    if not skip_s2_dates:
        metadata = add_dates_S2(
            metadata=metadata,
            X_path=X_tif_pattern
        )
    
    return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_arguemnt("--metadata-path",
                        default=os.path.join(DATASET_ROOT, "metadata.geojson"),
                        help="Filepath to read metadata from.")
    parser.add_argument("--save-path",
                        default=None,
                        help="Path to save the resulting metadata to. If not set, defaults to overwriting the path the metadata was read from.")
    parser.add_arguemnt("--dataset-root",
                        default=DATASET_ROOT,
                        help="The root of the dataset.")
    parser.add_argument("--mask-dir", default=os.path.join(DATASET_ROOT, "rpg_mask_unbuffered"),
                        help="The directory to load RPG masks from(defaults to '{DATASET_ROOT}/rpg_mask_unbuffered'})")
    parser.add_argument("--y-pattern", default=os.path.join(DATASET_ROOT, "y"),
                        help="The directory to load hedgerow rasters from(defaults to '{DATASET_ROOT}/y'})")
    parser.add_argument("--y-dir", default=os.path.join(DATASET_ROOT, "y"),
                        help="The directory to load hedgerow rasters from(defaults to '{DATASET_ROOT}/y'})")
    parser.add_argument("--X-tif-pattern", default="Xtif/X_{}.tif", 
                        help="Pattern to fill in order to find the associated .tif file for a given patch.")
    parser.add_argument("--start", default=None,
                        help="The starting point from which to process the metadata(default to 0, e.g. starting from the beginning)")
    parser.add_argument("--limit", default=None,
                        help="The total number of datapoints to process(defaults to processing all from the start)")
    parser.add_argument("--skip-s2-dates", action="store_true",
                        help="Flag which, if set, will skip adding the S2 dates.")
    parser.add_argument("--skip-rpg-metadata", action="store_true",
                        help="Flag which, if set, will skip adding the RPG metadata.")
    parser.add_argument("--skip-hedge-metadata", action="store_true",
                        help="Flag which, if set, will skip adding the hedge metadata.")
    parser.add_argument("--skip-save", action="store_true",
                        help="Skip saving the metadata resulting from the computation.")
    parser.add_argument("--export-gdrive", action="store_true",
                        help="Skip saving the metadata resulting from the computation.")
    parser.add_argument("--gdrive-save-path",
                        default=None,
                        help="Path to export to in google drive.")
    
    args = parser.parse_args()


    metadata = gpd.read_file(args.metadata_path)

    if args.start:
        metadata = metadata[args.start:]
    if args.limit:
        metadata = metadata[:args.limit]
    save_path = args.save_path if args.save_path is not None else args.metadata_path
    logger = logging.getLogger(__name__)
    metadata = main(
        metadata=metadata,
        dataset_root=args.dataset_root,
        mask_dir=args.mask_dir,
        y_dir=args.y_dir,
        y_pattern=args.y_pattern,
        X_tif_pattern=args.X_tif_pattern,
        skip_rpg_metadata=args.skip_rpg_metadata,
        skip_hedge_metadata=args.skip_hedge_metadata,
        skip_s2_dates=args.skip_s2_dates
    )

    if not args.skip_save:
        metadata.to_file(save_path)
    if args.export_gdrive:
        gdrive_path = args.gdrive_save_path
        with tempfile.TemporaryDirectory() as tempdir:
            filepath = os.path.join(tempdir.name, "metadata.geojson")

            metadata.to_file(filepath)
            service = get_drive_service()
            upload_file_to_drive(
                service=service,
                file_path=filepath,
                folder_path=gdrive_path,
                logger=logger,
                new_filename="metadata.geojson"
            )




    