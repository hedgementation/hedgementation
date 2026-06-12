"""Export AEF 1.3 Google Satellite Embedding X tiles from Earth Engine.

Run from the repository root after authenticating Earth Engine:

    python -m src.gee_export.aef1_3_export hedgementation_1_3_small
    python -m src.gee_export.aef1_3_export hedgementation_1_3_medium
    python -m src.gee_export.aef1_3_export hedgementation_1_3_large

Useful partial runs:

    python -m src.gee_export.aef1_3_export hedgementation_1_3_large --start 0 --end 1000
    python -m src.gee_export.aef1_3_export hedgementation_1_3_large --ids-file large_x_missing_ids.txt

The source argument is the Earth Engine FeatureCollection asset name under
projects/hedgementation-473422/assets/.
"""

import argparse
import asyncio
import logging

import ee

from src.gee_export.export_configs import AEFConfig, load_export_configs
from src.worker_queue import WorkerQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
TASKS = {}
PROJECT_ID = "hedgementation-473422"


def get_patch_polygon_from_feature(feature):
    return feature.geometry()


def _annual_embedding(geom, collection, start, end):
    return (
        ee.ImageCollection(collection)
        .filterDate(start, end)
        .filterBounds(geom)
        .mosaic()
    )


def get_embedding_for_geom(geom, aef_config: AEFConfig):
    """Average the embeddings over each [start, end] window in the config."""
    embeddings = [
        _annual_embedding(geom, aef_config.collection, start, end)
        for start, end in aef_config.windows
    ]
    total = embeddings[0]
    for emb in embeddings[1:]:
        total = total.add(emb)
    return total.divide(len(embeddings)).toFloat()


def register_task(feature_id, task):
    TASKS[f"X:{feature_id}"] = task


async def export_x(feature_id, fold, polygon, emb, folder, scale, crs):
    try:
        task = ee.batch.Export.image.toDrive(
            image=emb.clip(polygon),
            description=f"export_X_{feature_id}",
            folder=folder,
            fileNamePrefix=f"{feature_id}_fold{fold}_X",
            region=polygon,
            scale=scale,
            crs=crs,
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )
        task.start()
        register_task(feature_id, task)
        logger.info(f"[X] Started export for {feature_id} (fold={fold})")
        return task
    except Exception as e:
        logger.error(f"[X] Failed: {feature_id}: {e}")
        raise


async def monitor_tasks(poll_s=10):
    last_state = {}

    while True:
        if not TASKS:
            await asyncio.sleep(poll_s)
            continue

        done_count = 0
        total = len(TASKS)

        for key, task in list(TASKS.items()):
            try:
                st = task.status()
                state = st.get("state", "UNKNOWN")
                err = st.get("error_message", "")

                if last_state.get(key) != state:
                    if err:
                        logger.info(f"[TASK] {key} -> {state} | {err}")
                    else:
                        logger.info(f"[TASK] {key} -> {state}")
                    last_state[key] = state

                if state in ("COMPLETED", "FAILED", "CANCELLED"):
                    done_count += 1

            except Exception as e:
                logger.warning(f"[TASK] {key} status check failed: {e}")

        logger.info(f"[TASK] summary: done {done_count}/{total}")

        if total > 0 and done_count == total:
            logger.info("[TASK] all tasks finished.")
            return

        await asyncio.sleep(poll_s)


async def export_feature(feature, feature_id, x_folder, aef_config, scale, crs):
    fold = feature.get("fold").getInfo()
    polygon = get_patch_polygon_from_feature(feature)
    emb = get_embedding_for_geom(polygon, aef_config)
    return await export_x(feature_id, fold, polygon, emb, x_folder, scale, crs)


async def process_fc(asset_name, x_folder, start, end, aef_config, scale, crs):
    ee.Initialize(project=PROJECT_ID)

    fc = ee.FeatureCollection(f"projects/{PROJECT_ID}/assets/{asset_name}")
    n = fc.size().getInfo()
    logger.info(f"FC size = {n}")

    if end is None:
        end = n

    logger.info(f"Processing index range [{start}, {end})")

    wq = WorkerQueue(logger, max_concurrent=5)
    monitor = asyncio.create_task(monitor_tasks(poll_s=10))

    for i in range(start, end):
        f = ee.Feature(fc.toList(1, i).get(0))
        feature_id = str(f.get("ID_PATCH").getInfo())
        await wq.add_task(export_feature(f, feature_id, x_folder, aef_config, scale, crs))

    await wq.wait_all()
    await monitor


async def process_ids(asset_name, x_folder, ids, aef_config, scale, crs):
    ee.Initialize(project=PROJECT_ID)

    fc = ee.FeatureCollection(f"projects/{PROJECT_ID}/assets/{asset_name}")
    logger.info(f"Processing explicit ID list with {len(ids)} IDs")

    wq = WorkerQueue(logger, max_concurrent=5)
    monitor = asyncio.create_task(monitor_tasks(poll_s=10))

    for feature_id in ids:
        f = ee.Feature(fc.filter(ee.Filter.eq("ID_PATCH", int(feature_id))).first())
        await wq.add_task(export_feature(f, str(feature_id), x_folder, aef_config, scale, crs))

    await wq.wait_all()
    await monitor


async def main(source, x_folder, start, end, ids_file, aef_config, scale, crs):
    if ids_file:
        with open(ids_file, "r", encoding="utf-8") as f:
            ids = [line.strip() for line in f if line.strip()]
        await process_ids(source, x_folder, ids, aef_config, scale, crs)
    else:
        await process_fc(source, x_folder, start, end, aef_config, scale, crs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("--X_folder", default="Dataset_X")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--ids-file", default=None)
    parser.add_argument("--aef_config", default=None,
                        help="Path to an AEF config JSON (default: configs/gee_export/aef.json)")
    parser.add_argument("--shared_config", default=None,
                        help="Path to a shared export config JSON defining dates/crs/scale "
                             "(default: configs/gee_export/shared_export.json)")
    parser.add_argument("--only", choices=["X"], default="X", help=argparse.SUPPRESS)

    args = parser.parse_args()

    configs = load_export_configs(shared_path=args.shared_config, aef_path=args.aef_config)

    asyncio.run(main(
        source=args.source,
        x_folder=args.X_folder,
        start=args.start,
        end=args.end,
        ids_file=args.ids_file,
        aef_config=configs.aef,
        scale=configs.shared.scale,
        crs=configs.shared.crs,
    ))
