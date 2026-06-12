
from typing import Optional
from hedgementation_utils.training.metadata_library import MetadataLibrary, SizeGroup

from dotenv import load_dotenv
import os

import json

import geopandas as gpd

load_dotenv()

DATASET_ROOT = os.environ["DATASET_ROOT"]


testing_paths = {
        "valid": "{}/pastis/one_batch_metadata.csv",
        "train": "{}/pastis/one_batch_metadata.csv"
    }

exclude_classes = [
    "Temperate, cool; wet, no soil/terrain limitations",
    "Temperate, cool; moist, no soil/terrain limitations",
]

metadata = gpd.read_file(f"{DATASET_ROOT}/metadata.geojson")

TRAIN_FOLDS = json.loads(os.environ["TRAIN_FOLDS"] if os.environ["TRAIN_FOLDS"] else [0,1,2]) 
VALID_FOLDS = json.loads(os.environ["VALID_FOLDS"] if os.environ["VALID_FOLDS"] else [3])
TEST_FOLDS = json.loads(os.environ["TEST_FOLDS"] if os.environ["TEST_FOLDS"] else [4])

NUM_TILEGROUPS = int(os.environ.get("NUM_TILEGROUPS",5))

def get_frames_per_tile_group(frames, num_groups=NUM_TILEGROUPS):
    far_group_to_frames = {}
    train_frame = frames["train"]
    valid_frame = frames["valid"]
    test_frame = frames["test"]

    for far_group in range(num_groups):
        frames = {
            "train": train_frame[train_frame["tile_group"] != far_group],
            "valid": valid_frame[valid_frame["tile_group"] != far_group],
            "test": test_frame[test_frame["tile_group"] != far_group],
            "test_far": test_frame[test_frame["tile_group"] == far_group], 
            "train_far": train_frame[train_frame["tile_group"] == far_group],
            "valid_far": valid_frame[valid_frame["tile_group"] == far_group]
        }
        far_group_to_frames[far_group] = frames
    
    return far_group_to_frames

def get_default_frames(metadata):
    frames = {
        "train": metadata[metadata["fold"].isin(TRAIN_FOLDS)],
        "valid": metadata[metadata["fold"].isin(VALID_FOLDS)],
        "test": metadata[metadata["fold"].isin(TEST_FOLDS)]
    }

    return frames

def get_semantic_distance_frames(metadata, exclude_classes):

    semantic_distant_frames = {
        "train": metadata[metadata["fold"].isin(TRAIN_FOLDS) & ~metadata["aez_class"].isin(exclude_classes)],
        "valid": metadata[metadata["fold"].isin(VALID_FOLDS) & ~metadata["aez_class"].isin(exclude_classes)],
        "test": metadata[metadata["fold"].isin(TEST_FOLDS) & metadata["aez_class"].isin(exclude_classes)]
    }

    return semantic_distant_frames

def get_dense_hedgerow_frames(metadata, quantile):
    dense_hedgerows_frames = {
        "train": metadata[(metadata["fold"].isin(TRAIN_FOLDS)) & (metadata["hedge_pixel_count"] > metadata["hedge_pixel_count"].quantile(quantile))],
        "valid": metadata[metadata["fold"].isin(VALID_FOLDS) & metadata["aez_class"].isin(exclude_classes)],
        "test": metadata[metadata["fold"].isin(TEST_FOLDS) & metadata["aez_class"].isin(exclude_classes)]
    }

    return dense_hedgerows_frames


def get_queried_frames(metadata, queries):
    frames = {
        "train": metadata[metadata["fold"].isin(TRAIN_FOLDS)].query(queries["train"]),
        "valid": metadata[metadata["fold"].isin(VALID_FOLDS)].query(queries["valid"]),
        "test": metadata[metadata["fold"].isin(TEST_FOLDS)].query(queries["test"])
    }

    return frames

def downsample_to_queried_frames(metadata, queries):
    queried_frames = get_queried_frames(metadata, queries)
    frames = {
        "train": metadata[metadata["fold"].isin(TRAIN_FOLDS)].sample(len(queried_frames["train"])),
        "valid": metadata[metadata["fold"].isin(VALID_FOLDS)].sample(len(queried_frames["valid"])),
        "test": metadata[metadata["fold"].isin(TEST_FOLDS)].sample(len(queried_frames["test"])),
    }

    return frames

def get_full_dataset_frame(size_group: Optional[SizeGroup] = None):
    metadata = gpd.read_file(f"{DATASET_ROOT}/metadata.geojson")
    return metadata

