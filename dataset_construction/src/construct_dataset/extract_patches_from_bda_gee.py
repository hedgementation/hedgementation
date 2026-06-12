from typing import Optional

import ee
import os
import logging
from dotenv import load_dotenv
import geemap
import argparse

load_dotenv()

import geopandas as gpd

from src.construct_dataset.extract_patches_from_bda_locally import create_patches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


PROJECT_ID = os.getenv("GCLOUD_PROJECT_ID", "hedgementation")
PATCHES_ASSET = os.getenv("GEE_PATCHES_ASSET","all_france_patches"),
LINESTRINGS_ASSET = os.getenv("GEE_LINESTRINGS_ASSET", "bda_haie_2024")
INITIAL_BDA_PATH = os.getenv("INITIAL_BDA_PATH")
PATCH_SIZE_M = int(os.getenv("PATCH_SIZE_M", 1280))

logger.debug("Loaded env: PROJECT_ID=%s, PATCH_SIZE_M=%d", PROJECT_ID, PATCH_SIZE_M)





def create_linestrings_asset(
        linestrings_gdf: gpd.GeoDataFrame,
        project_id: str = PROJECT_ID,
        asset_name: str = LINESTRINGS_ASSET):

    new_asset_path = f"projects/{project_id}/assets/{asset_name}"
    logger.info("Creating linestrings asset at '%s'", new_asset_path)

    try:
        ee.Initialize(project=project_id)
        logger.debug("EE re-initialized for project '%s'", project_id)
    except Exception:
        logger.warning("EE re-initialization failed — re-authenticating")
        ee.Authenticate()
        ee.Initialize(project=project_id)

    logger.debug("Reprojecting linestrings GDF to EPSG:4326 (rows=%d)", len(linestrings_gdf))
    linestrings_gdf = linestrings_gdf.to_crs("EPSG:4326")

    logger.debug("Converting GDF to EE FeatureCollection")
    fc = geemap.geopandas_to_ee(linestrings_gdf)

    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description="Creating hedgementation dataset.",
        assetId=new_asset_path
    )
    task.start()
    logger.info("Export task started for linestrings asset (task id: %s)", task.id)


def create_patches_asset(linestrings_gdf: gpd.GeoDataFrame,
                         project_id: str = PROJECT_ID,
                         asset_name: str = PATCHES_ASSET,
                         patch_size_m: int = PATCH_SIZE_M):

    new_asset_path = f"projects/{project_id}/assets/{asset_name}"
    logger.info("Creating patches asset at '%s' (patch_size_m=%d)", new_asset_path, patch_size_m)

    try:
        ee.Initialize(project=project_id)
        logger.debug("EE re-initialized for project '%s'", project_id)
    except Exception:
        logger.warning("EE re-initialization failed — re-authenticating")
        ee.Authenticate()
        ee.Initialize(project=project_id)

    logger.debug("Generating patches from linestrings GDF (rows=%d)", len(linestrings_gdf))
    patches_gdf = create_patches(linestrings_gdf, patch_size=patch_size_m)
    logger.info("Patches created: %d patches generated", len(patches_gdf))

    logger.debug("Reprojecting patches GDF to EPSG:4326")
    patches_gdf = patches_gdf.to_crs("EPSG:4326")

    logger.debug("Converting patches GDF to EE FeatureCollection")
    fc = geemap.geopandas_to_ee(patches_gdf)

    task = ee.batch.Export.table.toAsset(
        collection=fc,
        description="Creating hedgementation dataset.",
        assetId=new_asset_path
    )
    task.start()
    logger.info("Export task started for patches asset (task id: %s)", task.id)


def merge_patches_and_linestrings(project_id: str = PROJECT_ID,
                                  patches_asset: str = PATCHES_ASSET,
                                  linestrings_asset: str = LINESTRINGS_ASSET, 
                                  limit: Optional[int] = None):

    patches_path = f"projects/{project_id}/assets/{patches_asset}"
    linestrings_path = f"projects/{project_id}/assets/{linestrings_asset}"
    new_asset_path = f"projects/{project_id}/assets/{patches_asset}_{linestrings_asset}_joined"

    logger.info("Merging patches and linestrings into '%s'", new_asset_path)
    logger.debug("Patches path: %s", patches_path)
    logger.debug("Linestrings path: %s", linestrings_path)

    try:
        ee.Initialize(project=PROJECT_ID)
        logger.debug("EE re-initialized for project '%s'", PROJECT_ID)
    except Exception:
        logger.warning("EE re-initialization failed — re-authenticating")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

    logger.debug("Loading patches FeatureCollection from EE")
    patches_fc = ee.FeatureCollection(patches_path)

    if limit:
        patches_fc = patches_fc.limit(limit)
        new_asset_path += f"limit_{limit}"

    logger.debug("Loading linestrings FeatureCollection from EE")
    linestrings_fc = ee.FeatureCollection(linestrings_path)

    def process_target_feature(feature, linestrings):
        feature = ee.Feature(feature)
        relevant_bda_haie = ee.FeatureCollection(linestrings.filterBounds(feature.geometry()))
        intersecting_bda = ee.FeatureCollection(relevant_bda_haie.filter(ee.Filter.intersects(".geo", feature.geometry())))

        hedgerow_count = ee.Number(intersecting_bda.size())

        cleabs = ee.List(intersecting_bda.aggregate_array("cleabs"))

        multi_linestring = ee.Algorithms.If(
            hedgerow_count.gt(0),
            ee.Geometry(intersecting_bda.geometry()).intersection(feature.geometry(), maxError=ee.ErrorMargin(1)),
            ee.Geometry.MultiLineString([]),
        )

        return feature.set({
            "hedgerows": multi_linestring,
            "hedgerow_count": hedgerow_count,
            "cleabs_list": cleabs
        })

    logger.info("Mapping patch-linestring intersection over FeatureCollection (this runs server-side on GEE)")
    new_fc = ee.FeatureCollection(patches_fc.map(lambda x: process_target_feature(x, linestrings=linestrings_fc)))

    task = ee.batch.Export.table.toAsset(
        collection=new_fc,
        description="Creating hedgementation dataset.",
        assetId=new_asset_path
    )
    task.start()
    logger.info("Export task started for merged asset (task id: %s)", task.id)


def main(
        tasks: list[str],
        linestrings_gdf: gpd.GeoDataFrame,
        project_id: str = PROJECT_ID,
        patches_asset: str = PATCHES_ASSET,
        linestrings_asset: str = LINESTRINGS_ASSET,
        patch_size_m: str = PATCH_SIZE_M,
        limit: Optional[int] = None,
):
    logger.info("Starting pipeline with tasks: %s", tasks)

    if "merge" in tasks and ("create_linestrings" in tasks or "create_patches" in tasks):
        logger.error("Cannot combine 'merge' with asset-creation tasks in a single run")
        raise ValueError("You can't create patches/linestrings and merge them in the same run! GEE needs time to create the assets before they can be used.")

    for task in tasks:
        logger.info("Running task: '%s'", task)
        if task == "create_patches":
            create_patches_asset(
                linestrings_gdf=linestrings_gdf,
                project_id=project_id,
                asset_name=patches_asset,
                patch_size_m=patch_size_m
            )
        elif task == "create_linestrings":
            create_linestrings_asset(
                linestrings_gdf=linestrings_gdf,
                project_id=project_id,
                asset_name=linestrings_asset
            )
        elif task == "merge":
            merge_patches_and_linestrings(
                project_id=project_id,
                patches_asset=patches_asset,
                linestrings_asset=linestrings_asset,
                limit=limit,
            )
        else:
            logger.warning("Unknown task '%s' — skipping", task)

    logger.info("All tasks completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    try:
        ee.Initialize(project=PROJECT_ID)
        logger.info("Earth Engine initialized with project '%s'", PROJECT_ID)
    except Exception:
        logger.warning("EE initialization failed — attempting authentication")
        ee.Authenticate(auth_mode="gcloud")
        ee.Initialize(project=PROJECT_ID)
        logger.info("Earth Engine authenticated and initialized with project '%s'", PROJECT_ID)

    parser.add_argument("--tasks", nargs="+", default=["create_patches"], choices=["create_linestrings", "create_patches", "merge"])
    parser.add_argument("--gdf-path", default=INITIAL_BDA_PATH)
    parser.add_argument("--project-id", default=PROJECT_ID)
    parser.add_argument("--patches-asset", default=PATCHES_ASSET)
    parser.add_argument("--linestrings-asset", default=LINESTRINGS_ASSET)
    parser.add_argument("--patch-size-m", type=int, default=PATCH_SIZE_M)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging verbosity")
    parser.add_argument("--limit", default=None, help="Limit for the number of patches to merge. Useful for dry runs.")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.debug("Log level set to %s", args.log_level)

    tasks = args.tasks
    gdf_path = args.gdf_path
    project_id = args.project_id
    patches_asset = args.patches_asset
    linestrings_asset = args.linestrings_asset
    patch_size_m = args.patch_size_m
    limit = args.limit

    logger.info("Loading GDF from '%s'", gdf_path)
    linestrings_gdf = gpd.read_file(gdf_path, use_arrow=True, engine="pyogrio")
    logger.info("GDF loaded: %d features", len(linestrings_gdf))

    main(
        tasks,
        linestrings_gdf,
        project_id=project_id,
        patches_asset=patches_asset,
        linestrings_asset=linestrings_asset,
        patch_size_m=patch_size_m,
        limit=limit
    )