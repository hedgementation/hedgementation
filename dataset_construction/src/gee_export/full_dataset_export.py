"""
Unified async-based tool to export satellite imagery, rasterized hedgerow images,
identity rasters, cloud score masks, and RPG agricultural masks to Google Cloud Storage
(default) or Google Drive for machine learning. Supports both local GeoDataFrames and
EE FeatureCollections.
"""
import asyncio
import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Set, Tuple

import ee
import geopandas as gpd
import shapely
import shapely.geometry
from google.cloud import storage as gcs
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from src.utils.drive_utils import get_drive_service, get_existing_ids_from_drive, upload_file_to_drive
from src.worker_queue import WorkerQueue
from src.gee_export.export_configs import ExportConfigs, load_export_configs
from src.gee_export.satellite_images import (
    get_collection,
    get_s2_date_string,
    get_stacked_img_for_patch,
    shapely_to_ee,
)
from src.gee_export.rasterization import rasterize_feature, rasterize_feature_identity
from src.gee_export.rpg_mask import compute_rpg_mask_for_patch
from src.gee_export.aef1_3_export import get_embedding_for_geom
from src.utils.io_utils import save_rpg_mask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PROJECT_ID = os.environ.get("GCLOUD_PROJECT_ID", "hedgementation")

# Task description prefixes for each GEE export type (RPG is local — excluded).
_TASK_DESC_PREFIXES = {
    'satellite': 'export_X',
    'hedgerow': 'export_Y',
    'identity': 'export_Y_id',
    'cloud': 'export_X_cloud',
    'aef': 'export_X_aef',
}


# ============================================================================
# Destination Configuration
# ============================================================================

@dataclass
class DestinationConfig:
    """Configuration for export destination: Google Cloud Storage (default) or Google Drive."""
    destination_type: str = 'gcs'  # 'gcs' or 'drive'

    # GCS: per-type buckets; fall back to unified_bucket when a specific one is absent
    unified_bucket: Optional[str] = None
    satellite_bucket: Optional[str] = None
    hedgerow_bucket: Optional[str] = None
    identity_bucket: Optional[str] = None
    cloud_bucket: Optional[str] = None
    rpg_bucket: Optional[str] = None
    aef_bucket: Optional[str] = None

    # GCS: optional path prefix ("folder") within the bucket. Per-type with unified fallback.
    unified_gcs_folder: Optional[str] = None
    satellite_gcs_folder: Optional[str] = None
    hedgerow_gcs_folder: Optional[str] = None
    identity_gcs_folder: Optional[str] = None
    cloud_gcs_folder: Optional[str] = None
    rpg_gcs_folder: Optional[str] = None
    aef_gcs_folder: Optional[str] = None

    # Drive: folder names (for GEE export tasks) and folder IDs (for dedup checks)
    satellite_folder: Optional[str] = None
    hedgerow_folder: Optional[str] = None
    identity_folder: Optional[str] = None
    cloud_folder: Optional[str] = None
    rpg_folder: Optional[str] = None
    aef_folder: Optional[str] = None
    satellite_folder_id: Optional[str] = None
    hedgerow_folder_id: Optional[str] = None
    identity_folder_id: Optional[str] = None
    cloud_folder_id: Optional[str] = None
    rpg_folder_id: Optional[str] = None
    aef_folder_id: Optional[str] = None

    def resolve_bucket(self, data_type: str) -> Optional[str]:
        """GCS bucket for `data_type`, falling back to unified_bucket."""
        specific = getattr(self, f'{data_type}_bucket', None)
        return specific if specific is not None else self.unified_bucket

    def resolve_gcs_folder(self, data_type: str) -> Optional[str]:
        """GCS path prefix for `data_type`, falling back to unified_gcs_folder. Returns None for Drive."""
        if self.destination_type != 'gcs':
            return None
        specific = getattr(self, f'{data_type}_gcs_folder', None)
        folder = specific if specific is not None else self.unified_gcs_folder
        if folder:
            return folder.strip('/')
        return None

    def get_export_dest(self, data_type: str) -> Optional[str]:
        """Export destination: bucket name (GCS) or folder name (Drive)."""
        if self.destination_type == 'gcs':
            return self.resolve_bucket(data_type)
        return getattr(self, f'{data_type}_folder', None)

    def get_presence_id(self, data_type: str) -> Optional[str]:
        """Identifier for checking file existence: bucket name (GCS) or folder ID (Drive)."""
        if self.destination_type == 'gcs':
            return self.resolve_bucket(data_type)
        return getattr(self, f'{data_type}_folder_id', None)


# ============================================================================
# GCS Utilities
# ============================================================================

def get_existing_ids_from_gcs(bucket_name: str, suffix_indicator: str, logger,
                               folder: Optional[str] = None) -> Set[str]:
    """Fetch object names from a GCS bucket (optionally restricted to a folder prefix)
    and extract feature IDs."""
    if not bucket_name:
        return set()
    location = f"gs://{bucket_name}/{folder}/" if folder else f"gs://{bucket_name}"
    logger.info(f"Checking existing files in {location}...")
    try:
        client = gcs.Client()
        existing_ids = set()
        list_kwargs = {'prefix': f'{folder}/'} if folder else {}
        for blob in client.list_blobs(bucket_name, **list_kwargs):
            name = blob.name.rsplit('/', 1)[-1]  # use basename so folder prefix is stripped
            if (name.endswith('.tif') or name.endswith('.npy')) and suffix_indicator in name:
                base_name = name.rsplit('.', 1)[0]
                if base_name.endswith(suffix_indicator):
                    extracted_id = base_name.rsplit(suffix_indicator, 1)[0]
                    existing_ids.add(extracted_id)
        logger.info(f"Found {len(existing_ids)} matching '{suffix_indicator}' files in {location}")
        return existing_ids
    except Exception as e:
        logger.error(f"Error fetching files from {location}: {e}")
        return set()


def _gcs_file_name_prefix(basename: str, folder: Optional[str]) -> str:
    """Prepend `folder/` to a GEE fileNamePrefix when a GCS folder is configured."""
    return f'{folder}/{basename}' if folder else basename


def upload_file_to_gcs(bucket_name: str, file_path: str, blob_name: str, logger):
    """Upload a local file to a GCS bucket."""
    try:
        client = gcs.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        logger.info(f"Uploaded {blob_name} to gs://{bucket_name}/{blob_name}")
    except Exception as e:
        logger.error(f"Failed to upload {file_path} to gs://{bucket_name}/{blob_name}: {e}")
        raise


def _get_existing_ids(dest_config: DestinationConfig) -> Dict[str, Set[str]]:
    """Fetch existing feature IDs for all export types based on destination config."""
    data_types = [
        ('satellite', '_X'),
        ('hedgerow', '_Y'),
        ('identity', '_Y_id'),
        ('cloud', '_X_cloud'),
        ('rpg', '_rpg'),
        ('aef', '_X_aef'),
    ]
    if dest_config.destination_type == 'gcs':
        return {
            dt: get_existing_ids_from_gcs(
                    dest_config.resolve_bucket(dt), suffix, logger,
                    folder=dest_config.resolve_gcs_folder(dt),
                ) if dest_config.resolve_bucket(dt) else set()
            for dt, suffix in data_types
        }
    else:
        folder_ids = {
            'satellite': dest_config.satellite_folder_id,
            'hedgerow': dest_config.hedgerow_folder_id,
            'identity': dest_config.identity_folder_id,
            'cloud': dest_config.cloud_folder_id,
            'rpg': dest_config.rpg_folder_id,
            'aef': dest_config.aef_folder_id,
        }
        return {
            dt: get_existing_ids_from_drive(folder_ids[dt], suffix, logger)
                if folder_ids[dt] else set()
            for dt, suffix in data_types
        }


# ============================================================================
# Helper Functions
# ============================================================================

def _check_done(feature_id, presence_id, existing_ids, task_desc, pending_gee_tasks):
    """Check if a single export type is already done (file exists or task is pending)."""
    if not presence_id:
        return True  # not configured, treat as done/not needed
    return (feature_id in existing_ids) or (task_desc in pending_gee_tasks)


def get_pending_gee_tasks() -> Set[str]:
    """Returns description strings for all active (SUBMITTED/READY/RUNNING) GEE tasks."""
    try:
        tasks = ee.data.getTaskList()
        active_states = {'SUBMITTED', 'READY', 'RUNNING'}
        pending = {t['description'] for t in tasks if t['state'] in active_states}
        logger.info(f"Found {len(pending)} active tasks in GEE queue.")
        return pending
    except Exception as e:
        logger.warning(f"Could not retrieve GEE task list: {e}. Proceeding without task checking.")
        return set()


def _count_gee_tasks_per_feature(dest_config: DestinationConfig) -> int:
    """Count GEE tasks submitted per feature based on configured export types."""
    return sum(1 for t in _TASK_DESC_PREFIXES if dest_config.get_presence_id(t))


def compute_fill_queue_params(
    source: str,
    dest_config: DestinationConfig,
    gee_queue_capacity: int = 3000,
) -> Tuple[int, int]:
    """Compute (start_index, limit) to fill the GEE task queue to capacity.

    Queries the live queue, determines available slots, then scans the GDF to
    find the first feature that still needs at least one GEE export.  Returns
    (0, 0) when the queue is already full or all features are done.
    """
    pending_tasks = get_pending_gee_tasks()
    available_slots = gee_queue_capacity - len(pending_tasks)
    logger.info(f"GEE queue: {len(pending_tasks)} active, {available_slots} slots available")

    if available_slots <= 0:
        logger.info("GEE task queue is at capacity. Nothing to export.")
        return 0, 0

    tasks_per_feature = _count_gee_tasks_per_feature(dest_config)
    if tasks_per_feature == 0:
        logger.warning("No GEE export types configured; cannot fill queue.")
        return 0, 0

    max_new_features = available_slots // tasks_per_feature
    logger.info(f"{tasks_per_feature} GEE task(s)/feature → targeting up to {max_new_features} new features")

    logger.info(f"Scanning {source} to find resume index...")
    gdf = gpd.read_file(source)
    existing_ids = _get_existing_ids(dest_config)

    start_idx = len(gdf)  # sentinel: all done
    for i, (idx, row) in enumerate(gdf.iterrows()):
        feature_id = str(row.get('id', idx))
        all_gee_done = all(
            _check_done(
                feature_id,
                dest_config.get_presence_id(t),
                existing_ids[t],
                f'{prefix}_{feature_id}',
                pending_tasks,
            )
            for t, prefix in _TASK_DESC_PREFIXES.items()
        )
        if not all_gee_done:
            start_idx = i
            break

    if start_idx == len(gdf):
        logger.info("All features already processed or pending in GEE queue.")
        return 0, 0

    logger.info(f"fill_queue: start={start_idx}, limit={max_new_features}")
    return start_idx, max_new_features


# ============================================================================
# Export Functions
# ============================================================================

async def export_metadata_geojson(features_data: list, destination: str, dest_type: str = 'gcs',
                                   gcs_folder: Optional[str] = None):
    """Export a metadata GeoJSON to GCS or Google Drive."""
    try:
        logger.info("Saving metadata to local geojson...")
        gdf = gpd.GeoDataFrame.from_features([f.getInfo() for f in features_data])
        gdf.to_file("metadata_post_export.geojson")

        ee_fc = ee.FeatureCollection(features_data)
        if dest_type == 'gcs':
            task = ee.batch.Export.table.toCloudStorage(
                collection=ee_fc,
                description=f'metadata_export_{destination}',
                bucket=destination,
                fileNamePrefix=_gcs_file_name_prefix('metadata', gcs_folder),
                fileFormat='GeoJSON'
            )
        else:
            task = ee.batch.Export.table.toDrive(
                collection=ee_fc,
                description=f'metadata_export_{destination}',
                folder=destination,
                fileNamePrefix='metadata',
                fileFormat='GeoJSON'
            )
        task.start()
        logger.info(f"Started metadata export to {destination}")
        return task

    except Exception as e:
        logger.error(f"Failed to export metadata to {destination}: {e}")
        raise


async def export_satellite_image(feature_id, geom, img, destination, dest_type='gcs',
                                  gcs_folder=None, crs='EPSG:3857', nb_pixel=128):
    """Export satellite imagery for a feature."""
    try:
        common = dict(
            image=img.clip(geom),
            description=f'export_X_{feature_id}',
            fileNamePrefix=_gcs_file_name_prefix(f'{feature_id}_X', gcs_folder)
                            if dest_type == 'gcs' else f'{feature_id}_X',
            region=geom,
            dimensions=f'{nb_pixel}x{nb_pixel}',
            maxPixels=1e10,
            crs=crs,
            fileFormat='GeoTIFF'
        )
        if dest_type == 'gcs':
            task = ee.batch.Export.image.toCloudStorage(**common, bucket=destination)
        else:
            task = ee.batch.Export.image.toDrive(**common, folder=destination)
        task.start()
        logger.info(f"Started satellite export for feature {feature_id}")
        return {'feature_id': feature_id, 'task': task, 'status': 'started', 'type': 'satellite'}
    except Exception as e:
        logger.error(f"Failed to export satellite image for {feature_id}: {e}")
        raise


async def export_hedgerow_raster(feature, destination, shared_cfg, downsampling_cfg,
                                 dest_type='gcs', gcs_folder=None):
    """Export rasterized hedgerows for a feature."""
    try:
        await asyncio.sleep(0.1)

        img = ee.Image(rasterize_feature(
            feature,
            crs=shared_cfg.crs,
            width=downsampling_cfg.width,
            initial_scale=downsampling_cfg.initial_scale,
            final_scale=downsampling_cfg.final_scale,
            reduce_resolution=downsampling_cfg.reduce_resolution,
            nb_pixel=shared_cfg.nb_pixel,
        ))
        feature_id = ee.Feature(feature).get("id").getInfo()

        nb_pixel = shared_cfg.nb_pixel
        common = dict(
            image=img,
            description=f"export_Y_{feature_id}",
            fileNamePrefix=_gcs_file_name_prefix(f"{feature_id}_Y", gcs_folder)
                            if dest_type == 'gcs' else f"{feature_id}_Y",
            region=img.geometry(),
            dimensions=f'{nb_pixel}x{nb_pixel}',
            maxPixels=1e10,
            crs=shared_cfg.crs,
            fileFormat='GeoTIFF',
        )
        if dest_type == 'gcs':
            task = ee.batch.Export.image.toCloudStorage(**common, bucket=destination)
        else:
            task = ee.batch.Export.image.toDrive(**common, folder=destination)
        task.start()
        logger.info(f"Started hedgerow export for feature {feature_id}")
        return {'feature_id': feature_id, 'task': task, 'status': 'started', 'type': 'hedgerow'}

    except Exception as e:
        logger.error(f"Failed to export hedgerow raster for feature: {e}")
        raise


async def export_identity_raster(feature, destination, shared_cfg, downsampling_cfg,
                                 dest_type='gcs', gcs_folder=None):
    """Export per-linestring identity raster for a feature."""
    try:
        await asyncio.sleep(0.1)

        img = ee.Image(rasterize_feature_identity(
            feature,
            crs=shared_cfg.crs,
            width=downsampling_cfg.width,
            initial_scale=downsampling_cfg.initial_scale,
            final_scale=downsampling_cfg.final_scale,
            reduce_resolution=downsampling_cfg.reduce_resolution,
            nb_pixel=shared_cfg.nb_pixel,
        ))
        feature_id = ee.Feature(feature).get("id").getInfo()

        nb_pixel = shared_cfg.nb_pixel
        common = dict(
            image=img,
            description=f"export_Y_id_{feature_id}",
            fileNamePrefix=_gcs_file_name_prefix(f"{feature_id}_Y_id", gcs_folder)
                            if dest_type == 'gcs' else f"{feature_id}_Y_id",
            region=img.geometry(),
            dimensions=f'{nb_pixel}x{nb_pixel}',
            maxPixels=1e10,
            crs=shared_cfg.crs,
            fileFormat='GeoTIFF',
        )
        if dest_type == 'gcs':
            task = ee.batch.Export.image.toCloudStorage(**common, bucket=destination)
        else:
            task = ee.batch.Export.image.toDrive(**common, folder=destination)
        task.start()
        logger.info(f"Started identity raster export for feature {feature_id}")
        return {'feature_id': feature_id, 'task': task, 'status': 'started', 'type': 'identity'}

    except Exception as e:
        logger.error(f"Failed to export identity raster for feature: {e}")
        raise


async def export_cloud_mask(feature_id, geom, cloud_img, destination, dest_type='gcs',
                             gcs_folder=None, crs='EPSG:3857', nb_pixel=128):
    """Export cloud score mask imagery for a feature."""
    try:
        common = dict(
            image=cloud_img.clip(geom),
            description=f'export_X_cloud_{feature_id}',
            fileNamePrefix=_gcs_file_name_prefix(f'{feature_id}_X_cloud', gcs_folder)
                            if dest_type == 'gcs' else f'{feature_id}_X_cloud',
            region=geom,
            dimensions=f'{nb_pixel}x{nb_pixel}',
            maxPixels=1e10,
            crs=crs,
            fileFormat='GeoTIFF'
        )
        if dest_type == 'gcs':
            task = ee.batch.Export.image.toCloudStorage(**common, bucket=destination)
        else:
            task = ee.batch.Export.image.toDrive(**common, folder=destination)
        task.start()
        logger.info(f"Started cloud mask export for feature {feature_id}")
        return {'feature_id': feature_id, 'task': task, 'status': 'started', 'type': 'cloud'}
    except Exception as e:
        logger.error(f"Failed to export cloud mask for {feature_id}: {e}")
        raise


async def export_aef_embedding(feature_id, geom, aef_cfg, destination, dest_type='gcs',
                               gcs_folder=None, crs='EPSG:3857', nb_pixel=128):
    """Export AEF (Google Satellite Embedding) for a feature."""
    try:
        emb = get_embedding_for_geom(geom, aef_cfg)
        common = dict(
            image=emb.clip(geom),
            description=f'export_X_aef_{feature_id}',
            fileNamePrefix=_gcs_file_name_prefix(f'{feature_id}_X_aef', gcs_folder)
                            if dest_type == 'gcs' else f'{feature_id}_X_aef',
            region=geom,
            dimensions=f'{nb_pixel}x{nb_pixel}',
            maxPixels=1e13,
            crs=crs,
            fileFormat='GeoTIFF',
        )
        if dest_type == 'gcs':
            task = ee.batch.Export.image.toCloudStorage(**common, bucket=destination)
        else:
            task = ee.batch.Export.image.toDrive(**common, folder=destination)
        task.start()
        logger.info(f"Started AEF embedding export for feature {feature_id}")
        return {'feature_id': feature_id, 'task': task, 'status': 'started', 'type': 'aef'}
    except Exception as e:
        logger.error(f"Failed to export AEF embedding for {feature_id}: {e}")
        raise


async def export_rpg_mask(feature_id: str,
                          patch_geom,
                          patch_crs: str,
                          rpg_gdf: gpd.GeoDataFrame,
                          dest_config: DestinationConfig,
                          drive_service=None,
                          rpg_buffer_width: Optional[int] = None) -> dict:
    """Compute RPG agricultural mask locally and upload to GCS or Google Drive."""
    def _compute_and_upload():
        mask, rpg_coverage = compute_rpg_mask_for_patch(
            patch_geom, rpg_gdf, patch_crs=patch_crs, buffer_width=rpg_buffer_width
        )
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            save_rpg_mask(mask, save_path=tmp_path)
            blob_name = f"{feature_id}_rpg.npy"
            if dest_config.destination_type == 'gcs':
                bucket = dest_config.resolve_bucket('rpg')
                folder = dest_config.resolve_gcs_folder('rpg')
                full_blob_name = _gcs_file_name_prefix(blob_name, folder)
                upload_file_to_gcs(bucket, tmp_path, full_blob_name, logger)
            else:
                svc = drive_service or get_drive_service()
                upload_file_to_drive(svc, tmp_path, dest_config.rpg_folder, logger,
                                     new_filename=blob_name)
        finally:
            os.remove(tmp_path)
        return rpg_coverage

    rpg_coverage = await asyncio.to_thread(_compute_and_upload)
    logger.info(f"Exported RPG mask for feature {feature_id} (coverage={rpg_coverage:.3f})")
    return {'feature_id': feature_id, 'rpg_coverage': rpg_coverage, 'type': 'rpg'}


async def export_feature_pair(feature, feature_id, geom,
                               s2_collection, cloud_collection,
                               dest_config: DestinationConfig,
                               export_configs: ExportConfigs,
                               cloud_filter,
                               should_export_satellite=True,
                               should_export_raster=True,
                               should_export_identity=True,
                               should_export_cloud=True,
                               should_export_aef=True):
    """Export EE-based outputs (X, Y, Y_id, X_cloud, X_aef) for a single feature."""
    try:
        dest_type = dest_config.destination_type
        shared = export_configs.shared
        results = {}

        if should_export_satellite:
            stacked_img = get_stacked_img_for_patch(
                geom, s2_collection, filter_clouds=cloud_filter,
                cloud_threshold=export_configs.satellite.cloud_threshold,
            )
            results['satellite'] = await export_satellite_image(
                feature_id, geom, stacked_img,
                dest_config.get_export_dest('satellite'), dest_type,
                gcs_folder=dest_config.resolve_gcs_folder('satellite'),
                crs=shared.crs, nb_pixel=shared.nb_pixel
            )
        else:
            results['satellite'] = "Skipped"

        if should_export_raster:
            results['hedgerow'] = await export_hedgerow_raster(
                feature,
                dest_config.get_export_dest('hedgerow'),
                shared, export_configs.downsampling, dest_type,
                gcs_folder=dest_config.resolve_gcs_folder('hedgerow'),
            )
        else:
            results['hedgerow'] = "Skipped"

        if should_export_identity:
            results['identity'] = await export_identity_raster(
                feature,
                dest_config.get_export_dest('identity'),
                shared, export_configs.downsampling, dest_type,
                gcs_folder=dest_config.resolve_gcs_folder('identity'),
            )
        else:
            results['identity'] = "Skipped"

        if should_export_cloud:
            cloud_img = get_stacked_img_for_patch(geom, cloud_collection, filter_clouds=False)
            results['cloud'] = await export_cloud_mask(
                feature_id, geom, cloud_img,
                dest_config.get_export_dest('cloud'), dest_type,
                gcs_folder=dest_config.resolve_gcs_folder('cloud'),
                crs=shared.crs, nb_pixel=shared.nb_pixel
            )
        else:
            results['cloud'] = "Skipped"

        if should_export_aef:
            results['aef'] = await export_aef_embedding(
                feature_id, geom, export_configs.aef,
                dest_config.get_export_dest('aef'), dest_type,
                gcs_folder=dest_config.resolve_gcs_folder('aef'),
                crs=shared.crs, nb_pixel=shared.nb_pixel,
            )
        else:
            results['aef'] = "Skipped"

        return results

    except Exception as e:
        logger.error(f"Failed to export feature pair {feature_id}: {e}")
        raise


# ============================================================================
# Main Processing Functions
# ============================================================================

async def _process_loop(
    feature_iter: Iterator[Tuple[str, ee.Feature, ee.Geometry, Optional[object]]],
    total: int,
    s2: ee.ImageCollection,
    cloud_collection: Optional[ee.ImageCollection],
    dest_config: DestinationConfig,
    export_configs: ExportConfigs,
    existing_ids: Dict[str, Set[str]],
    rpg_gdf: Optional[gpd.GeoDataFrame] = None,
    gdf_crs: Optional[str] = None,
    drive_service=None,
    collect_all_metadata: bool = False,
    filter_cloudy: bool = True,
    rpg_buffer_width: Optional[int] = None,
    skip_tasks: Optional[Set[str]] = None,
    gee_queue_capacity: int = 3000,
):
    """
    Core export loop shared by both GDF and EE FeatureCollection paths.

    feature_iter yields (feature_id, ee_feature, ee_geom, shapely_geom) where
    shapely_geom is the patch geometry as a Shapely object (None for EE path —
    fetched lazily if RPG export is configured).

    gdf_crs is the CRS string for shapely_geom (None for EE path, which always
    returns WGS84 from .getInfo()).
    """
    skip_tasks = set(skip_tasks or [])
    pending_gee_tasks = get_pending_gee_tasks()
    available_gee_slots = gee_queue_capacity - len(pending_gee_tasks)
    logger.info(f"GEE queue: {len(pending_gee_tasks)} active, {available_gee_slots} slots available")
    gee_tasks_submitted = 0
    wq = WorkerQueue(logger=logger)
    metadata_features = []
    skipped_count = 0

    for i, (feature_id, ee_feature, feature_geom, shapely_geom) in enumerate(feature_iter):
        in_sat = 'satellite' in skip_tasks or _check_done(
            feature_id, dest_config.get_presence_id('satellite'),
            existing_ids['satellite'], f'export_X_{feature_id}', pending_gee_tasks)
        in_hedge = 'hedgerow' in skip_tasks or _check_done(
            feature_id, dest_config.get_presence_id('hedgerow'),
            existing_ids['hedgerow'], f'export_Y_{feature_id}', pending_gee_tasks)
        in_identity = 'identity' in skip_tasks or _check_done(
            feature_id, dest_config.get_presence_id('identity'),
            existing_ids['identity'], f'export_Y_id_{feature_id}', pending_gee_tasks)
        in_cloud = 'cloud' in skip_tasks or _check_done(
            feature_id, dest_config.get_presence_id('cloud'),
            existing_ids['cloud'], f'export_X_cloud_{feature_id}', pending_gee_tasks)
        in_rpg = 'rpg' in skip_tasks or _check_done(
            feature_id, dest_config.get_presence_id('rpg'),
            existing_ids['rpg'], f'export_rpg_{feature_id}', pending_gee_tasks)
        in_aef = 'aef' in skip_tasks or _check_done(
            feature_id, dest_config.get_presence_id('aef'),
            existing_ids['aef'], f'export_X_aef_{feature_id}', pending_gee_tasks)
        should_skip = in_sat and in_hedge and in_identity and in_cloud and in_rpg and in_aef

        if collect_all_metadata or not should_skip:
            dates_str = get_s2_date_string(feature_geom, s2, filter_cloudy,
                                           export_configs.satellite.cloud_threshold)
            ee_feature = ee_feature.set("dates-s2", dates_str)
            metadata_features.append(ee_feature)

        if should_skip:
            skipped_count += 1
            if skipped_count % 100 == 0:
                logger.info(f"Skipping feature {feature_id} (already exists). Total skipped: {skipped_count}")
            continue

        # Check GEE queue capacity before submitting
        gee_tasks_for_feature = sum([
            int(not in_sat and bool(dest_config.get_presence_id('satellite'))),
            int(not in_hedge and bool(dest_config.get_presence_id('hedgerow'))),
            int(not in_identity and bool(dest_config.get_presence_id('identity'))),
            int(not in_cloud and bool(dest_config.get_presence_id('cloud'))),
            int(not in_aef and bool(dest_config.get_presence_id('aef'))),
        ])
        if gee_tasks_for_feature > 0 and gee_tasks_submitted + gee_tasks_for_feature > available_gee_slots:
            logger.info(
                f"GEE queue capacity reached ({gee_tasks_submitted} tasks submitted this run). "
                f"Stopping at feature {i}/{total}."
            )
            break

        # Queue EE-based exports
        await wq.add_task(
            export_feature_pair(
                feature=ee_feature,
                feature_id=feature_id,
                geom=feature_geom,
                s2_collection=s2,
                cloud_collection=cloud_collection,
                dest_config=dest_config,
                export_configs=export_configs,
                should_export_satellite=not in_sat,
                should_export_raster=not in_hedge,
                should_export_identity=not in_identity,
                should_export_cloud=not in_cloud,
                should_export_aef=not in_aef,
                cloud_filter=filter_cloudy
            )
        )
        gee_tasks_submitted += gee_tasks_for_feature

        # Queue RPG mask export (local compute + upload)
        if rpg_gdf is not None and not in_rpg:
            if shapely_geom is None:
                # EE path: fetch geometry as shapely on demand
                shapely_geom = shapely.geometry.shape(feature_geom.getInfo())
                patch_crs = "EPSG:4326"
            else:
                patch_crs = gdf_crs
            await wq.add_task(
                export_rpg_mask(feature_id, shapely_geom, patch_crs,
                                rpg_gdf, dest_config, drive_service,
                                rpg_buffer_width=rpg_buffer_width)
            )

        if i % 10 == 0:
            logger.info(f"Progress: {i}/{total}, Skipped: {skipped_count}, Queue stats: {wq.stats()}")

    await wq.wait_all()
    logger.info(f"Done. Processed: {total - skipped_count}, Skipped: {skipped_count}")

    if metadata_features and 'metadata' not in skip_tasks:
        logger.info("Exporting metadata...")
        dest_type = dest_config.destination_type
        sat_dest = dest_config.get_export_dest('satellite')
        hedge_dest = dest_config.get_export_dest('hedgerow')
        if sat_dest:
            await export_metadata_geojson(
                metadata_features, sat_dest, dest_type,
                gcs_folder=dest_config.resolve_gcs_folder('satellite'),
            )
        if hedge_dest:
            await export_metadata_geojson(
                metadata_features, hedge_dest, dest_type,
                gcs_folder=dest_config.resolve_gcs_folder('hedgerow'),
            )
    logger.info("Metadata export completed")


def _gdf_feature_iter(gdf):
    """Yield (feature_id, ee_feature, ee_geom, shapely_geom) for each GDF row."""
    for i, (idx, row) in enumerate(gdf.iterrows()):
        feature_id = str(row.get('id', idx))
        shapely_geom = gdf.geometry.iloc[i]
        ee_geom = shapely_to_ee(shapely_geom)

        if 'hedgerows' in row and row['hedgerows'] is not None:
            hedgerows_geom = shapely_to_ee(shapely.wkt.loads(row['hedgerows']))
        else:
            hedgerows_geom = ee_geom

        feature_dict = {
            'type': 'Feature',
            'id': feature_id,
            'geometry': ee_geom.getInfo(),
            'properties': {
                'id': feature_id,
                'hedgerows': hedgerows_geom.getInfo(),
                'ID_PATCH': feature_id,
            }
        }
        dictrow = dict(row)
        del dictrow["geometry"]
        del dictrow["hedgerows"]
        feature_dict["properties"].update(dictrow)

        yield feature_id, ee.Feature(feature_dict), ee_geom, shapely_geom


def _ee_feature_iter(feature_collection, length):
    """Yield (feature_id, ee_feature, feature_geom, None) for each EE feature.

    shapely_geom is None; it is fetched lazily in _process_loop if RPG export is needed.
    """
    for i in range(length):
        feature = ee.Feature(feature_collection.toList(1, i).get(0))
        feature_id = str(feature.get("id").getInfo())
        yield feature_id, feature, feature.geometry(), None


def _load_collections(export_configs: ExportConfigs):
    """Load S2 and cloud score EE image collections."""
    s2 = get_collection(
        collection_name=export_configs.satellite.collection,
        start_date=export_configs.shared.start_date,
        end_date=export_configs.shared.end_date,
        selected_bands=export_configs.satellite.selected_bands
    )
    cloud = get_collection(
        collection_name=export_configs.cloud_score.collection,
        start_date=export_configs.cloud_score.start_date,
        end_date=export_configs.cloud_score.end_date,
        selected_bands=export_configs.cloud_score.selected_bands
    )
    return s2, cloud


async def process_gdf(gdf_path: str,
                      dest_config: DestinationConfig,
                      export_configs: ExportConfigs,
                      rpg_path: Optional[str] = None,
                      rpg_buffer_width: Optional[int] = None,
                      start: Optional[int] = None,
                      limit: Optional[int] = None,
                      cloud_filter: bool = True,
                      skip_tasks: Optional[Set[str]] = None,
                      gee_queue_capacity: int = 3000):
    """Export features from a local GeoDataFrame file."""
    existing_ids = _get_existing_ids(dest_config)

    rpg_gdf = None
    drive_service = None
    if dest_config.get_export_dest('rpg') and rpg_path:
        logger.info(f"Loading RPG data from {rpg_path}...")
        rpg_gdf = gpd.read_file(rpg_path, use_arrow=True, engine='pyogrio').to_crs("EPSG:3857")
        if dest_config.destination_type == 'drive':
            drive_service = get_drive_service()

    s2, cloud_collection = _load_collections(export_configs)

    logger.info(f"Loading GeoDataFrame from {gdf_path}")
    gdf = gpd.read_file(gdf_path)
    if start:
        gdf = gdf[start:]
        logger.info(f"Starting from index {start}")
    if limit:
        gdf = gdf[:limit]
        logger.info(f"Limited to {limit} features")

    logger.info(f"Processing {len(gdf)} features from GeoDataFrame")
    await _process_loop(
        _gdf_feature_iter(gdf), len(gdf), s2, cloud_collection,
        dest_config, export_configs, existing_ids,
        rpg_gdf=rpg_gdf, gdf_crs=str(gdf.crs), drive_service=drive_service,
        collect_all_metadata=True, filter_cloudy=cloud_filter,
        rpg_buffer_width=rpg_buffer_width, skip_tasks=skip_tasks,
        gee_queue_capacity=gee_queue_capacity,
    )


async def process_ee_featurecollection(asset_name: str,
                                       dest_config: DestinationConfig,
                                       export_configs: ExportConfigs,
                                       rpg_path: Optional[str] = None,
                                       rpg_buffer_width: Optional[int] = None,
                                       cloud_filter: bool = True,
                                       skip_tasks: Optional[Set[str]] = None,
                                       gee_queue_capacity: int = 3000):
    """Export features from an Earth Engine FeatureCollection asset."""
    existing_ids = _get_existing_ids(dest_config)

    rpg_gdf = None
    drive_service = None
    if dest_config.get_export_dest('rpg') and rpg_path:
        logger.info(f"Loading RPG data from {rpg_path}...")
        rpg_gdf = gpd.read_file(rpg_path, use_arrow=True, engine='pyogrio').to_crs("EPSG:3857")
        if dest_config.destination_type == 'drive':
            drive_service = get_drive_service()

    ee.Initialize(project=PROJECT_ID)
    s2, cloud_collection = _load_collections(export_configs)

    fc = ee.FeatureCollection(f"projects/{PROJECT_ID}/assets/{asset_name}")
    length = fc.size().getInfo()
    logger.info(f"Processing {length} features from EE FeatureCollection")

    await _process_loop(
        _ee_feature_iter(fc, length), length, s2, cloud_collection,
        dest_config, export_configs, existing_ids,
        rpg_gdf=rpg_gdf, gdf_crs=None, drive_service=drive_service,
        filter_cloudy=cloud_filter, rpg_buffer_width=rpg_buffer_width,
        skip_tasks=skip_tasks, gee_queue_capacity=gee_queue_capacity,
    )


def setup_ee(client_secrets_file: str = 'client_secrets.json',
             scopes: list[str] = [
                 'https://www.googleapis.com/auth/cloud-platform',
                 'https://www.googleapis.com/auth/drive',
                 'https://www.googleapis.com/auth/earthengine'
             ],
             project_id=PROJECT_ID):

    flow = InstalledAppFlow.from_client_secrets_file(
        client_secrets_file=client_secrets_file,
        scopes=scopes
    )

    credentials = flow.run_local_server()
    ee.Initialize(credentials=credentials, project='hedgementation')

    try:
        ee.Initialize(credentials=credentials, project=project_id)
    except:
        ee.Authenticate()
        ee.Initialize(credentials=credentials, project=project_id)


def setup_ee_service_account(service_account: str = "hedgementation@hedgementation.iam.gserviceaccount.com",
                              private_key_file: str = 'secrets/service_account_key.json',
                              scopes: list[str] = [
                                  'https://www.googleapis.com/auth/cloud-platform',
                                  'https://www.googleapis.com/auth/drive',
                                  'https://www.googleapis.com/auth/earthengine'
                              ],
                              project_id=PROJECT_ID):

    credentials = ee.ServiceAccountCredentials(
        email=service_account,
        key_file=private_key_file
    )

    try:
        ee.Initialize(credentials=credentials, project=project_id)
    except:
        ee.Authenticate()
        ee.Initialize(credentials=credentials, project=project_id)


async def main(source: str,
               dest_config: DestinationConfig,
               export_configs: ExportConfigs,
               rpg_path: Optional[str] = None,
               rpg_buffer_width: Optional[int] = None,
               source_type: str = 'auto',
               start: Optional[int] = None,
               limit: Optional[int] = None,
               cloud_filter: bool = True,
               skip_tasks: Optional[Set[str]] = None,
               gee_queue_capacity: int = 3000):
    """Export satellite imagery, hedgerow rasters, identity rasters, cloud masks, and RPG masks."""

    setup_ee_service_account()
    if source_type == 'auto':
        source_type = 'gdf' if source.endswith(('.gpkg', '.shp', '.geojson')) else 'ee'

    logger.info(f"Processing source type: {source_type}, destination: {dest_config.destination_type}")

    if source_type == 'gdf':
        await process_gdf(source, dest_config, export_configs, rpg_path, rpg_buffer_width,
                          start, limit, cloud_filter,
                          skip_tasks=skip_tasks, gee_queue_capacity=gee_queue_capacity)
    elif source_type == 'ee':
        await process_ee_featurecollection(source, dest_config, export_configs, rpg_path,
                                           rpg_buffer_width, cloud_filter,
                                           skip_tasks=skip_tasks, gee_queue_capacity=gee_queue_capacity)
    else:
        raise ValueError(f"Unknown source type: {source_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export satellite imagery, hedgerow rasters, identity rasters, "
                    "cloud masks, and RPG agricultural masks to GCS (default) or Google Drive."
    )
    parser.add_argument("--config", default=None,
                        help="Path to a JSON config file. CLI args override config values.")
    parser.add_argument("--source",
                        help="Path to local GeoDataFrame file (e.g., .gpkg, .shp) or EE asset name")
    parser.add_argument("--destination_type", default=None, choices=['gcs', 'drive'],
                        help="Export destination: 'gcs' (default) or 'drive'")

    # GCS arguments
    gcs_group = parser.add_argument_group("Google Cloud Storage")
    gcs_group.add_argument("--unified_bucket", default=None,
                           help="GCS bucket for all data types (fallback when no type-specific bucket is given)")
    gcs_group.add_argument("--satellite_bucket", default=None,
                           help="GCS bucket for satellite images (X)")
    gcs_group.add_argument("--hedgerow_bucket", default=None,
                           help="GCS bucket for hedgerow rasters (Y)")
    gcs_group.add_argument("--identity_bucket", default=None,
                           help="GCS bucket for identity rasters (Y_id)")
    gcs_group.add_argument("--cloud_bucket", default=None,
                           help="GCS bucket for cloud score masks (X_cloud)")
    gcs_group.add_argument("--rpg_bucket", default=None,
                           help="GCS bucket for RPG agricultural masks")
    gcs_group.add_argument("--aef_bucket", default=None,
                           help="GCS bucket for AEF satellite embeddings (X_aef)")
    gcs_group.add_argument("--unified_gcs_folder", default=None,
                           help="Path prefix (folder) inside the GCS bucket for all data types "
                                "(fallback when no type-specific GCS folder is given)")
    gcs_group.add_argument("--satellite_gcs_folder", default=None,
                           help="Path prefix (folder) inside the GCS bucket for satellite images (X)")
    gcs_group.add_argument("--hedgerow_gcs_folder", default=None,
                           help="Path prefix (folder) inside the GCS bucket for hedgerow rasters (Y)")
    gcs_group.add_argument("--identity_gcs_folder", default=None,
                           help="Path prefix (folder) inside the GCS bucket for identity rasters (Y_id)")
    gcs_group.add_argument("--cloud_gcs_folder", default=None,
                           help="Path prefix (folder) inside the GCS bucket for cloud score masks (X_cloud)")
    gcs_group.add_argument("--rpg_gcs_folder", default=None,
                           help="Path prefix (folder) inside the GCS bucket for RPG agricultural masks")
    gcs_group.add_argument("--aef_gcs_folder", default=None,
                           help="Path prefix (folder) inside the GCS bucket for AEF satellite embeddings (X_aef)")

    # Drive arguments
    drive_group = parser.add_argument_group("Google Drive")
    drive_group.add_argument("--satellite_folder", default=None,
                             help="Drive folder NAME for satellite images (X)")
    drive_group.add_argument("--hedgerow_folder", default=None,
                             help="Drive folder NAME for hedgerow rasters (Y)")
    drive_group.add_argument("--identity_folder", default=None,
                             help="Drive folder NAME for identity rasters (Y_id)")
    drive_group.add_argument("--cloud_folder", default=None,
                             help="Drive folder NAME for cloud score masks (X_cloud)")
    drive_group.add_argument("--rpg_folder", default=None,
                             help="Drive folder NAME for RPG agricultural masks")
    drive_group.add_argument("--aef_folder", default=None,
                             help="Drive folder NAME for AEF satellite embeddings (X_aef)")
    drive_group.add_argument("--satellite_folder_id", default=None,
                             help="Drive folder ID for satellite images (for dedup checks)")
    drive_group.add_argument("--hedgerow_folder_id", default=None,
                             help="Drive folder ID for hedgerow rasters (for dedup checks)")
    drive_group.add_argument("--identity_folder_id", default=None,
                             help="Drive folder ID for identity rasters (for dedup checks)")
    drive_group.add_argument("--cloud_folder_id", default=None,
                             help="Drive folder ID for cloud score masks (for dedup checks)")
    drive_group.add_argument("--rpg_folder_id", default=None,
                             help="Drive folder ID for RPG masks (for dedup checks)")
    drive_group.add_argument("--aef_folder_id", default=None,
                             help="Drive folder ID for AEF satellite embeddings (for dedup checks)")

    # Export config files (each defaults to its file under configs/gee_export/)
    cfg_group = parser.add_argument_group(
        "Export configs",
        "Paths to JSON config files. Defaults live in configs/gee_export/. "
        "These keys can also be set inside the main --config file."
    )
    cfg_group.add_argument("--shared_export_config", default=None,
                           help="Shared time-series/grid config: date range, crs, scale, nb_pixel "
                                "(default: configs/gee_export/shared_export.json)")
    cfg_group.add_argument("--satellite_config", default=None,
                           help="Sentinel-2 config: collection, bands, cloud threshold "
                                "(default: configs/gee_export/sentinel2.json)")
    cfg_group.add_argument("--cloud_score_config", default=None,
                           help="Cloud score config: collection, bands, optional date overrides "
                                "(default: configs/gee_export/cloud_score.json)")
    cfg_group.add_argument("--aef_config", default=None,
                           help="AEF embedding config: collection, averaged date windows "
                                "(default: configs/gee_export/aef.json)")
    cfg_group.add_argument("--downsampling_config", default=None,
                           help="y/y_id rasterization config: buffer width, initial/final scale, "
                                "reduce_resolution (default: configs/gee_export/downsampling.json)")

    parser.add_argument("--rpg_path", default=os.getenv("RPG_ASSET", None),
                        help="Path to the RPG .gpkg file (required when RPG export is configured)")
    parser.add_argument("--rpg_buffer_width", default=None, type=int,
                        help="Number of meters to buffer RPG geometry by(defaults to not buffering)")
    parser.add_argument("--skip_tasks", nargs="+",
                        choices=['satellite', 'hedgerow', 'identity', 'cloud', 'rpg', 'aef', 'metadata'],
                        default=None,
                        help="Task types to skip entirely (e.g. --skip_tasks satellite hedgerow identity cloud)")
    parser.add_argument("--source-type", choices=['gdf', 'ee', 'auto'], default='auto',
                        help="Source type: 'gdf' for local file, 'ee' for EE asset, 'auto' to detect")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of features to process (useful for testing)")
    parser.add_argument("--start", type=int, default=0,
                        help="Index to start from (useful for interrupted runs)")
    parser.add_argument("--no_cloud_filter", action="store_true",
                        help="Flag to disable filtering based on cloud pixel percentage.")
    parser.add_argument("--fill_queue", action="store_true",
                        help="Automatically compute start index and limit to fill the GEE "
                             "task queue to capacity. Ignores --start and --limit. "
                             "Only supported for GDF sources.")
    parser.add_argument("--gee_queue_capacity", type=int, default=3000,
                        help="Maximum GEE task queue size used for capacity enforcement (default: 3000)")

    args = parser.parse_args()

    # Load config file defaults, then let explicit CLI args override them
    config = {}
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)

    # Identify which args were explicitly passed on the CLI (not just defaulted)
    explicitly_set = {action.dest for action in parser._actions if action.dest != "help"}
    cli_provided = {
        dest for dest in explicitly_set
        if any(f"--{dest}" in sys.argv or f"--{dest.replace('_', '-')}" in sys.argv
               for dest in [dest])
    }

    def resolve(dest, default=None):
        """Return CLI value if explicitly provided, else config file value, else default."""
        if dest in cli_provided:
            return getattr(args, dest)
        return config.get(dest, getattr(args, dest) if getattr(args, dest) != default else default)

    # Determine destination type; auto-detect Drive when Drive folder params are present and
    # destination_type was not explicitly specified (preserves backward compat with Drive configs).
    destination_type = resolve("destination_type")
    if destination_type is None:
        has_drive_params = any(resolve(k) for k in (
            'satellite_folder', 'hedgerow_folder', 'satellite_folder_id', 'hedgerow_folder_id'
        ))
        destination_type = 'drive' if has_drive_params else 'gcs'

    dest_config = DestinationConfig(
        destination_type=destination_type,
        unified_bucket=resolve("unified_bucket"),
        satellite_bucket=resolve("satellite_bucket"),
        hedgerow_bucket=resolve("hedgerow_bucket"),
        identity_bucket=resolve("identity_bucket"),
        cloud_bucket=resolve("cloud_bucket"),
        rpg_bucket=resolve("rpg_bucket"),
        unified_gcs_folder=resolve("unified_gcs_folder"),
        satellite_gcs_folder=resolve("satellite_gcs_folder"),
        hedgerow_gcs_folder=resolve("hedgerow_gcs_folder"),
        identity_gcs_folder=resolve("identity_gcs_folder"),
        cloud_gcs_folder=resolve("cloud_gcs_folder"),
        rpg_gcs_folder=resolve("rpg_gcs_folder"),
        satellite_folder=resolve("satellite_folder"),
        hedgerow_folder=resolve("hedgerow_folder"),
        identity_folder=resolve("identity_folder"),
        cloud_folder=resolve("cloud_folder"),
        rpg_folder=resolve("rpg_folder"),
        satellite_folder_id=resolve("satellite_folder_id"),
        hedgerow_folder_id=resolve("hedgerow_folder_id"),
        identity_folder_id=resolve("identity_folder_id"),
        cloud_folder_id=resolve("cloud_folder_id"),
        rpg_folder_id=resolve("rpg_folder_id"),
        aef_bucket=resolve("aef_bucket"),
        aef_gcs_folder=resolve("aef_gcs_folder"),
        aef_folder=resolve("aef_folder"),
        aef_folder_id=resolve("aef_folder_id"),
    )

    # Validate that required destination params are present
    if destination_type == 'gcs':
        if not dest_config.resolve_bucket('satellite') or not dest_config.resolve_bucket('hedgerow'):
            parser.error(
                "GCS mode requires satellite and hedgerow buckets. "
                "Provide --unified_bucket or both --satellite_bucket and --hedgerow_bucket."
            )
    else:
        if not dest_config.satellite_folder:
            parser.error("Drive mode requires --satellite_folder (via CLI or --config)")
        if not dest_config.hedgerow_folder:
            parser.error("Drive mode requires --hedgerow_folder (via CLI or --config)")

    export_configs = load_export_configs(
        shared_path=resolve("shared_export_config"),
        satellite_path=resolve("satellite_config"),
        cloud_score_path=resolve("cloud_score_config"),
        aef_path=resolve("aef_config"),
        downsampling_path=resolve("downsampling_config"),
    )

    fill_queue = args.fill_queue or bool(config.get("fill_queue"))

    if fill_queue:
        source_type_resolved = resolve("source_type")
        if source_type_resolved == 'auto':
            src = resolve("source") or ""
            source_type_resolved = 'gdf' if src.endswith(('.gpkg', '.shp', '.geojson')) else 'ee'
        if source_type_resolved != 'gdf':
            parser.error("--fill_queue is only supported for GDF sources.")
        setup_ee_service_account()
        gee_queue_cap = resolve("gee_queue_capacity")
        start_val, limit_val = compute_fill_queue_params(resolve("source"), dest_config, gee_queue_cap)
        if limit_val == 0:
            sys.exit(0)
    else:
        start_val = resolve("start")
        limit_val = resolve("limit")
        gee_queue_cap = resolve("gee_queue_capacity")

    asyncio.run(main(
        source=resolve("source"),
        dest_config=dest_config,
        export_configs=export_configs,
        rpg_path=resolve("rpg_path"),
        rpg_buffer_width=resolve("rpg_buffer_width"),
        source_type=resolve("source_type"),
        start=start_val,
        limit=limit_val,
        cloud_filter=(not resolve("no_cloud_filter")),
        skip_tasks=set(resolve("skip_tasks") or []),
        gee_queue_capacity=gee_queue_cap,
    ))
