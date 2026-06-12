#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage
import webdataset as wds

import os
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = os.getenv("DATASET_ROOT")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

BUCKET      = "hedgementation_1_3"
CLOUD_PREFIX = ""
PREFIX      = f"{DATASET_ROOT}/"
WDS_PREFIX  = "webdataset/"
TMP_PATTERN = "/tmp/dataset-%06d.tar"
SHARD_SIZE  = 500
MAX_WORKERS = 16

client = storage.Client()
bucket = client.bucket(BUCKET)

# Each entry is (wds_key, gcs_path_template).
# Use {id} as the placeholder for the sample ID in path templates.
DEFAULT_PAIRS: list[tuple[str, str]] = [
    ("X.tif",                f"{PREFIX}X/X_{{id}}.tif"),
    ("X_cloud.tif",          f"{PREFIX}X_cloud/X_cloud_{{id}}.tif"),
    ("y.tif",                f"{PREFIX}y/y_{{id}}.tif"),
    ("y_id.tif",             f"{PREFIX}y_id/y_id_{{id}}.tif"),
    ("rpg_unbuffered.npy",   f"{PREFIX}rpg_unbuffered/rpg_{{id}}.npy"),
    ("rpg_buffered_10m.npy", f"{PREFIX}rpg_buffered_10m/rpg_{{id}}.npy"),
    #("aef.npy", f"{PREFIX}AEF_NPY/AEF_{{id}}.npy"),
]


def blob_bytes(path: str, is_local: bool = True) -> bytes:

    if is_local:
        with open(path, "rb") as infile:
            buf = io.BytesIO(infile.read())
    else:
        buf = io.BytesIO()
        bucket.blob(path).download_to_file(buf)
    return buf.getvalue()


def upload_and_delete(fname: str) -> None:
    dest = f"gs://{BUCKET}/{WDS_PREFIX}{os.path.basename(fname)}"
    log.info("uploading %s → %s", fname, dest)
    subprocess.run(["gsutil", "cp", fname, dest], check=True)
    os.remove(fname)


def list_source_ids() -> list[str]:
    """Discover all sample IDs from the X/ prefix in GCS."""
    return sorted(
        m.group(1)
        for blob in client.list_blobs(BUCKET, prefix=f"{CLOUD_PREFIX}X/")
        if (m := re.match(r".*/(.+)_X\.tif$", blob.name))
    )


def list_existing_shards() -> list[str]:
    """Return sorted list of .tar shard filenames already uploaded to GCS."""
    return sorted(
        os.path.basename(blob.name)
        for blob in client.list_blobs(BUCKET, prefix=WDS_PREFIX)
        if blob.name.endswith(".tar")
    )


def build_sample(
    sample_id: str,
    pairs: list[tuple[str, str]],
    is_local: bool
) -> dict | None:
    """Download all blobs for a sample and return a wds-ready dict, or None on failure."""
    sample: dict = {"__key__": sample_id}
    for key, path_tpl in pairs:
        path = path_tpl.format(id=sample_id)
        try:
            sample[key] = blob_bytes(path, is_local=is_local)
        except Exception as exc:
            log.warning("skipping %s — failed to fetch %s: %s", sample_id, path, exc)
            return None
    return sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pack GCS blobs into webdataset shards.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="cap the number of IDs processed (useful for test runs)",
    )
    p.add_argument(
        "--skip-existing", action=argparse.BooleanOptionalAction, default=True,
        help="skip IDs already covered by shards present in GCS",
    )
    p.add_argument(
        "--redo-last", action="store_true", default=False,
        help=(
            "delete and redo the last existing shard before resuming; "
            "use this when a previous run may have written a partial final shard"
        ),
    )
    p.add_argument("--bucket", default=BUCKET,
                   help="GCS bucket name")
    p.add_argument("--cloud-prefix", default=CLOUD_PREFIX,
                   help="prefix within bucket for source data")
    p.add_argument("--prefix", default=PREFIX,
                   help="local dataset root prefix")
    p.add_argument("--wds-prefix", default=WDS_PREFIX,
                   help="GCS prefix for webdataset shards")
    p.add_argument("--tmp-pattern", default=TMP_PATTERN,
                   help="local temp file pattern for shards")
    p.add_argument("--workers", type=int, default=MAX_WORKERS,
                   help="number of parallel download threads")
    p.add_argument("--shard-size", type=int, default=SHARD_SIZE,
                   help="max samples per shard")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--use_gcloud_data", action="store_true", help="Signals to build the webdataset with the data in google cloud rather than the one stored locally.")
    return p.parse_args()


def main(pairs: list[tuple[str, str]] | None = None) -> None:
    """
    Build webdataset shards from GCS blobs.

    Args:
        pairs: list of (wds_key, gcs_path_template) pairs to use instead of DEFAULT_PAIRS.
    """
    global BUCKET, CLOUD_PREFIX, PREFIX, WDS_PREFIX, TMP_PATTERN, bucket
    args = parse_args()
    logging.getLogger().setLevel(args.log_level)

    BUCKET = args.bucket
    CLOUD_PREFIX = args.cloud_prefix
    PREFIX = args.prefix
    WDS_PREFIX = args.wds_prefix
    TMP_PATTERN = args.tmp_pattern
    bucket = client.bucket(BUCKET)

    if pairs is None:
        pairs = [
            ("X.tif",                f"{PREFIX}X/X_{{id}}.tif"),
            ("X_cloud.tif",          f"{PREFIX}X_cloud/X_cloud_{{id}}.tif"),
            ("y.tif",                f"{PREFIX}y/y_{{id}}.tif"),
            ("y_id.tif",             f"{PREFIX}y_id/y_id_{{id}}.tif"),
            ("rpg_unbuffered.npy",   f"{PREFIX}rpg_unbuffered/rpg_{{id}}.npy"),
            ("rpg_buffered_10m.npy", f"{PREFIX}rpg_buffered_10m/rpg_{{id}}.npy"),
        ]

    all_ids = list_source_ids()
    log.info("found %d source IDs", len(all_ids))

    is_local = not args.use_gcloud_data

    start_shard = 0
    ids = all_ids

    if args.skip_existing:
        existing = list_existing_shards()
        if existing:
            n = len(existing)
            if args.redo_last:
                last_blob_path = f"{WDS_PREFIX}{existing[-1]}"
                log.info("deleting gs://%s/%s to redo last shard", BUCKET, last_blob_path)
                bucket.blob(last_blob_path).delete()
                n -= 1
            start_shard = n
            skip_count = n * args.shard_size
            ids = all_ids[skip_count:]
            log.info(
                "%d existing shard(s); skipping first %d IDs, resuming at shard %d",
                n, skip_count, start_shard,
            )
        else:
            log.info("no existing shards found, building from scratch")

    if args.limit is not None:
        ids = ids[: args.limit]
        log.info("--limit: capped to %d IDs", len(ids))

    if not ids:
        log.info("nothing to do — all IDs already covered")
        return

    log.info(
        "writing %d samples | shard_size=%d | start_shard=%d | workers=%d",
        len(ids), args.shard_size, start_shard, args.workers,
    )

    written = skipped = 0

    with wds.ShardWriter(
        TMP_PATTERN,
        maxcount=args.shard_size,
        post=upload_and_delete,
        start_shard=start_shard,
    ) as sink:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for sample in pool.map(build_sample, ids, [pairs] * len(ids), [is_local] * len(ids)):
                if sample is None:
                    skipped += 1
                    continue
                sink.write(sample)
                written += 1
                if written % args.shard_size == 0:
                    log.info("progress: %d written, %d skipped", written, skipped)

    log.info("finished: %d written, %d skipped", written, skipped)


if __name__ == "__main__":
    main()
