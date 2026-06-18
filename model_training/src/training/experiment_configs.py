"""
Experiment configuration generator for parallel SLURM jobs.

Each experiment corresponds to a keyword in TrainerConfig.predefined_config_keywords;
TrainerConfig.get_predefined_config(keyword) builds the actual TrainerConfig.

To add a new experiment:
1. Add a keyword to TrainerConfig.predefined_config_keywords
2. Add a matching `case` arm in TrainerConfig.get_predefined_config()
"""

import json
import os
from collections import OrderedDict

import geopandas as gpd
from dotenv import load_dotenv

from src.training.get_metadata_frames import (
    downsample_to_queried_frames,
    get_default_frames,
    get_frames_per_tile_group,
    get_queried_frames,
)
from src.training.trainer_config import TrainerConfig


load_dotenv()
DATASET_ROOT = os.environ.get("DATASET_ROOT", "")
NUM_TILEGROUPS = int(os.environ.get("NUM_TILEGROUPS", 5))


def get_experiment_configs(
    return_ordered_dict: bool = False,
) -> "dict[str, TrainerConfig] | OrderedDict[str, TrainerConfig]":
    """Build a TrainerConfig for each predefined keyword."""
    keywords = TrainerConfig.predefined_config_keywords
    experiments = {}
    for k in keywords:
        try:
            experiments[k] = TrainerConfig.get_predefined_config(k)
        except:
            continue
    return OrderedDict(experiments) if return_ordered_dict else experiments


def get_metadata_frames_mapping() -> dict:
    """
    Mapping of named metadata-frame splits used by analysis notebooks.

    Loaded lazily so importing this module stays cheap (these queries hit the
    full metadata.geojson).
    """
    metadata = gpd.read_file(f"{DATASET_ROOT}/metadata.geojson")

    return_all_query = "ilevel_0 in ilevel_0"
    subtropics_query = (
        'thz_class in ["TRC5: Subtropics, cool", "TRC4: Subtropics, moderately cool"]'
    )
    temperate_query = (
        'thz_class in ["TRC7: Temperate, cool", "TRC6: Temperate, moderately cool"]'
    )

    default_frames = get_default_frames(metadata)
    default_frames_split_by_tilegroup = get_frames_per_tile_group(default_frames)

    crossval_dict = {
        f"default_frames_far_group_{i}": default_frames_split_by_tilegroup[i]
        for i in range(NUM_TILEGROUPS)
    }

    non_crossval_dict = {
        "default_frames": default_frames,
        "far_frames": default_frames,
        "test_on_subtropic_frames": get_queried_frames(
            metadata,
            queries={
                "train": return_all_query,
                "valid": return_all_query,
                "test": subtropics_query,
            },
        ),
        "test_on_temperate_frames": get_queried_frames(
            metadata,
            queries={
                "train": return_all_query,
                "valid": return_all_query,
                "test": temperate_query,
            },
        ),
        "train_subtropic_test_temperate_frames": get_queried_frames(
            metadata,
            queries={
                "train": subtropics_query,
                "valid": subtropics_query,
                "test": temperate_query,
            },
        ),
        "train_temperate_test_subtropic_frames": get_queried_frames(
            metadata,
            queries={
                "train": temperate_query,
                "valid": temperate_query,
                "test": subtropics_query,
            },
        ),
        "train_same_size_as_subtropic_test": downsample_to_queried_frames(
            metadata,
            queries={
                "train": subtropics_query,
                "valid": subtropics_query,
                "test": temperate_query,
            },
        ),
        "train_same_size_as_temperate_test": downsample_to_queried_frames(
            metadata,
            queries={
                "train": temperate_query,
                "valid": temperate_query,
                "test": subtropics_query,
            },
        ),
    }
    return crossval_dict | non_crossval_dict


def save_experiment_configs(output_dir: str = "experiments") -> int:
    """
    Save experiment configurations as JSON files plus per-experiment metadata CSVs.

    Layout:
        {output_dir}/experiment_{idx}_config.json
        {output_dir}/experiment_{idx}/metadata/{split}.csv
        {output_dir}/predefined_experiments_matching.json

    Returns:
        Number of experiments written.
    """
    os.makedirs(output_dir, exist_ok=True)
    experiments = get_experiment_configs(return_ordered_dict=True)

    for keyword, config in experiments.items():
        metadata_dir = os.path.join(output_dir, f"{keyword}", "metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        for split, frame in config.metadata_frames.items():
            frame.to_csv(os.path.join(metadata_dir, f"{split}.csv"))

        config_path = os.path.join(output_dir, f"{keyword}_config.json")
        with open(config_path, "w") as f:
            json.dump(config.to_dict(), f, indent=4)
    
    num_exp = len(experiments)

    print(f"Saved {num_exp} experiment configurations to {output_dir}")
    return num_exp


if __name__ == "__main__":
    save_experiment_configs()