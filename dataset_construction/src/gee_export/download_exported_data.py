import argparse
import asyncio
import io
import json
import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse
from google.oauth2.credentials import Credentials

from google.cloud import storage as gcs
from googleapiclient.http import MediaIoBaseDownload

from src.utils.drive_utils import (
    get_creds,
    get_drive_service,
    get_drive_folder_name,
    get_http,
    list_drive_subfolders,
)
from src.worker_queue import WorkerQueue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_source(source: str):
    """Classify a source string.

    Returns ('gcs', bucket, blob_prefix) for paths starting with 'gs://',
    otherwise ('drive', folder_id, None).
    """
    if source.startswith("gs://"):
        parsed = urlparse(source)
        return "gcs", parsed.netloc, parsed.path.lstrip("/")
    return "drive", source, None


def _infer_folder_and_prefix(segment: str) -> tuple[str, str]:
    """Derive a local folder name and filename prefix from a source's last segment.

    The local folder keeps the full segment name; the prefix collapses to 'rpg' for
    any segment starting with 'rpg' (matching the '_rpg' export convention).
    """
    folder = segment.rstrip("/").rsplit("/", 1)[-1]
    prefix = "rpg" if folder.startswith("rpg") else folder
    return folder, prefix


def _resolve_job(source: str, creds_path: Optional[str] = None) -> tuple[str, str]:
    """Infer (folder, prefix) for a single source via its last path segment.

    GCS sources use the trailing path component (or the bucket name when the path is
    empty); Drive sources are folder IDs whose name is fetched from the Drive API.
    """
    kind, identifier, blob_prefix = parse_source(source)
    if kind == "gcs":
        segment = blob_prefix.rstrip("/").rsplit("/", 1)[-1] if blob_prefix else identifier
    else:
        segment = get_drive_folder_name(identifier)
    return _infer_folder_and_prefix(segment)


def _expand_source_root(source_root: str,
                        creds_path: Optional[str] = None) -> list[tuple[str, str, str]]:
    """Expand a single root into (source, folder, prefix) triples for each subfolder.

    GCS roots list immediate "directory" prefixes via a delimited blob listing; Drive
    roots list immediate subfolders via the Drive API.
    """
    kind, identifier, blob_prefix = parse_source(source_root)
    jobs: list[tuple[str, str, str]] = []

    if kind == "gcs":
        if creds_path:
            creds = Credentials.from_authorized_user_file(filename=creds_path)
        else:
            creds = None
        client = gcs.Client(credentials=creds)
        root_prefix = f"{blob_prefix.rstrip('/')}/" if blob_prefix else ""
        iterator = client.list_blobs(identifier, prefix=root_prefix, delimiter="/")
        # Consume the iterator so .prefixes is populated with the child "directories".
        for _ in iterator:
            pass
        for child in iterator.prefixes:
            folder, prefix = _infer_folder_and_prefix(child)
            jobs.append((f"gs://{identifier}/{child.rstrip('/')}", folder, prefix))
    else:
        for sub_id, name in list_drive_subfolders(identifier):
            folder, prefix = _infer_folder_and_prefix(name)
            jobs.append((sub_id, folder, prefix))

    return jobs


def _destination_path(destination_folder: str, filename: str, prefix: str) -> str:
    id_part = "".join(c for c in filename if c.isdigit())
    ext = filename.rsplit(".", 1)[-1] if "." in filename else ""
    suffix = f".{ext}" if ext else ""
    return os.path.join(destination_folder, f"{prefix}_{id_part}{suffix}")


async def download_drive_file(file, creds, http, destination_folder: str, prefix: str):
    try:
        service = get_drive_service(creds=creds, http=http)
        filename = file["name"]
        gdrive_id = file["id"]
        request = service.files().get_media(fileId=gdrive_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        destination_file = _destination_path(destination_folder, filename, prefix)
        with open(destination_file, "wb") as outfile:
            outfile.write(buf.getvalue())
    except Exception as e:
        logger.error(f"Failed to download Drive file {file.get('name')}: {e}")
        raise


async def download_gcs_blob(client, bucket_name: str, blob_name: str,
                            destination_folder: str, prefix: str, overwrite_existing:bool=False):
    try:
        filename = blob_name.rsplit("/", 1)[-1]
        destination_file = _destination_path(destination_folder, filename, prefix)
        if os.path.exists(destination_file) and not overwrite_existing:
            logger.info(f"File {destination_file} already exists. Returning...")
            return
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        await asyncio.to_thread(blob.download_to_filename, destination_file)
    except Exception as e:
        logger.error(f"Failed to download gs://{bucket_name}/{blob_name}: {e}")
        raise


async def queue_drive_folder_download(wq: WorkerQueue,
                                       folder_id: str,
                                       destination_folder: str,
                                       prefix: str,
                                       limit: Optional[int],
                                       names: Optional[list] = None):
    creds = get_creds()
    service = get_drive_service(creds=creds)

    file_list = []
    page_token = None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            pageSize=1000,
            fields="nextPageToken, files(id, name)",
            pageToken=page_token,
        ).execute()
        curr = response.get("files", [])
        page_token = response.get("nextPageToken")
        if limit:
            file_list.extend(curr[: limit - len(file_list)])
        else:
            file_list.extend(curr)
        if page_token is None or (limit is not None and len(file_list) >= limit):
            break

    total = len(file_list)
    for i, current_file in enumerate(file_list):
        if names and current_file["name"] not in names:
            continue
        if current_file["name"].endswith(".geojson"):
            continue
        http = get_http(creds=creds)
        await wq.add_task(
            download_drive_file(
                current_file,
                creds=creds,
                http=http,
                destination_folder=destination_folder,
                prefix=prefix,
            )
        )
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{total}, Queue stats: {wq.stats()}")


async def queue_gcs_folder_download(wq: WorkerQueue,
                                     bucket_name: str,
                                     blob_prefix: str,
                                     destination_folder: str,
                                     prefix: str,
                                     limit: Optional[int],
                                     names: Optional[list] = None,
                                     creds_path: Optional[str] = None,
                                     overwrite_existing:bool = False):
    if creds_path:
        creds = Credentials.from_authorized_user_file(filename=creds_path)
    else:
        creds = None
    client = gcs.Client(credentials=creds)
    list_kwargs = {"prefix": blob_prefix} if blob_prefix else {}
    
    existing_ids = set()
    for filename in os.listdir(destination_folder):
        digits = re.findall(r'\d+', filename.split(".")[0])
        if digits:
            existing_ids.add(digits[0])

    blob_list = []
    for blob in client.list_blobs(bucket_name, **list_kwargs):
        if blob.name.endswith("/"):
            continue
        blob_list.append(blob)
        if limit is not None and len(blob_list) >= limit:
            break

    total = len(blob_list)
    for i, blob in enumerate(blob_list):
        filename = blob.name.rsplit("/", 1)[-1]
        digits = re.findall(r'\d+', filename.split(".")[0])
        if not digits:
            logger.warning(f"Could not extract ID from blob filename '{filename}', skipping.")
            continue
        file_id = digits[0]
        if file_id in existing_ids and not overwrite_existing:
            print(f"ID {file_id} already present in {destination_folder}. Skipping...")
            continue
        if not filename:
            continue
        if names and filename not in names:
            continue
        if filename.endswith(".geojson"):
            continue
        await wq.add_task(
            download_gcs_blob(client, bucket_name, blob.name, destination_folder, prefix, overwrite_existing)
        )
        if i % 10 == 0:
            logger.info(f"Progress: {i}/{total}, Queue stats: {wq.stats()}")


def _build_jobs(
    sources: Optional[list[str]],
    destination_folders: Optional[list[str]],
    prefixes: Optional[list[str]],
    source_root: Optional[str],
    creds_path: Optional[str],
) -> list[tuple[str, str, str]]:
    """Resolve inputs into a list of (source, folder, prefix) jobs.

    With `source_root`, mirror every subfolder of the root. Otherwise iterate
    `sources`, using explicit `destination_folders`/`prefixes` when provided and
    inferring whichever of the two is omitted.
    """
    if source_root:
        if sources:
            raise ValueError("Pass either --source_root or --sources, not both.")
        return _expand_source_root(source_root, creds_path)

    if not sources:
        raise ValueError("Provide --sources or --source_root.")
    if destination_folders is not None and len(destination_folders) != len(sources):
        raise ValueError("--destination_folders must match --sources in length.")
    if prefixes is not None and len(prefixes) != len(sources):
        raise ValueError("--prefixes must match --sources in length.")

    jobs: list[tuple[str, str, str]] = []
    for i, source in enumerate(sources):
        folder = destination_folders[i] if destination_folders is not None else None
        prefix = prefixes[i] if prefixes is not None else None
        if folder is None or prefix is None:
            inferred_folder, inferred_prefix = _resolve_job(source, creds_path)
            folder = folder if folder is not None else inferred_folder
            prefix = prefix if prefix is not None else inferred_prefix
        jobs.append((source, folder, prefix))
    return jobs


async def main(
    destination_root: str,
    sources: Optional[list[str]] = None,
    destination_folders: Optional[list[str]] = None,
    prefixes: Optional[list[str]] = None,
    logger: logging.Logger = logger,
    source_root: Optional[str] = None,
    overwrite_existing: bool = False,
    limit: Optional[int] = None,
    names: Optional[list] = None,
    creds_path: Optional[str] = None,
):
    jobs = _build_jobs(sources, destination_folders, prefixes, source_root, creds_path)

    wq = WorkerQueue(logger=logger, max_concurrent=15)
    for source, folder, prefix in jobs:
        destination_folder = os.path.join(destination_root, folder)
        os.makedirs(destination_folder, exist_ok=True)

        kind, identifier, blob_prefix = parse_source(source)
        logger.info(f"Source '{source}' resolved as {kind} -> {destination_folder}")

        if kind == "gcs":
            await queue_gcs_folder_download(
                wq=wq,
                bucket_name=identifier,
                blob_prefix=blob_prefix,
                destination_folder=destination_folder,
                prefix=prefix,
                limit=limit,
                names=names,
                creds_path=creds_path,
                overwrite_existing=overwrite_existing
            )
        else:
            await queue_drive_folder_download(
                wq=wq,
                folder_id=identifier,
                destination_folder=destination_folder,
                prefix=prefix,
                limit=limit,
                names=names,
            )
    await wq.wait_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download exported data from a Google Drive folder or a GCS bucket path. "
                    "Sources starting with 'gs://' are treated as GCS; anything else is treated "
                    "as a Google Drive folder ID."
    )
    parser.add_argument("--destination_root", help="The base directory to save downloaded data")
    parser.add_argument("--destination_folders", nargs="+", default=None,
                        help="The folders within the root to save each source's contents into. "
                             "If omitted, inferred from each source's last path segment.")
    parser.add_argument("--sources", nargs="+", default=None,
                        help="One or more sources. Each is either a GCS path (gs://bucket/path) "
                             "or a Google Drive folder ID.")
    parser.add_argument("--source_root", default=None,
                        help="A single GCS path (gs://bucket/path) or Drive folder ID whose "
                             "immediate subfolders are each mirrored into destination_root with "
                             "inferred folders and prefixes. Mutually exclusive with --sources.")
    parser.add_argument("--prefixes", nargs="+", default=None,
                        help="Filename prefix to prepend to each downloaded file (e.g. 'X' or 'y'). "
                             "If omitted, inferred from each source's last path segment "
                             "(segments starting with 'rpg' use the prefix 'rpg').")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of files to download per source")
    parser.add_argument("--names", nargs="*", default=None,
                        help="Optional whitelist of exact filenames to download")
    parser.add_argument("--creds_path", default=None, help="Optional filepath holding credentials information for authenticating to google cloud")
    parser.add_argument("--overwrite_existing", action="store_true", help="Whether or not to overwrite files that already exist when downloading.")
    args = parser.parse_args()

    asyncio.run(
        main(
            destination_root=args.destination_root,
            destination_folders=args.destination_folders,
            sources=args.sources,
            source_root=args.source_root,
            prefixes=args.prefixes,
            limit=args.limit,
            names=args.names,
            overwrite_existing=args.overwrite_existing,
            logger=logging.getLogger(__name__),
        )
    )
