#!/usr/bin/env python3
"""
Run a single experiment, either from a JSON config file (the SLURM array path)
or from a predefined keyword (manual / smoke-test path).

Called by scripts/submit_experiments.py via the SLURM array job. Each array
task receives:
  --config_file      Path to JSON config produced by save_experiment_configs()
  --metadata_dir     Directory containing per-split metadata CSVs
  --data_path        Override for TrainerConfig.data_path
  --save_path        Override for TrainerConfig.save_path
  --cache_dir        Override for TrainerConfig.cache_dir
  --config_overrides JSON object of additional TrainerConfig field overrides

For ad-hoc local runs you can instead pass:
  --predefined_keyword <name>

Where <name> is one of TrainerConfig.predefined_config_keywords.
"""

import argparse
import inspect
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

import geopandas as gpd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import Trainer
from src.training.trainer_config import TrainerConfig


def load_metadata_dict_from_dir(metadata_dir: str) -> dict:
    """Load per-split metadata frames from CSVs written by save_experiment_configs()."""
    metadata_frames = {}
    for file in os.listdir(metadata_dir):
        split_name = os.path.splitext(file)[0]
        metadata_frames[split_name] = gpd.read_file(os.path.join(metadata_dir, file))
    return metadata_frames


def parse_config_overrides(overrides_str: str) -> dict:
    """Parse and validate a JSON string of TrainerConfig field overrides."""
    try:
        overrides = json.loads(overrides_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"--config_overrides must be valid JSON: {e}")
    if not isinstance(overrides, dict):
        raise ValueError("--config_overrides must be a JSON object, e.g. '{\"lr\": 0.0001}'")
    valid_keys = set(inspect.signature(TrainerConfig).parameters)
    invalid = set(overrides) - valid_keys
    if invalid:
        raise ValueError(
            f"Unknown TrainerConfig fields in --config_overrides: {sorted(invalid)}. "
            f"Valid fields: {sorted(valid_keys)}"
        )
    return overrides


def build_trainer_config(
    config_file: Optional[str],
    predefined_keyword: Optional[str],
    data_path: Optional[str],
    save_path: Optional[str],
    metadata_dir: Optional[str],
    cache_dir: Optional[str] = None,
    config_overrides: Optional[dict] = None,
) -> TrainerConfig:
    """Build a TrainerConfig from either a JSON snapshot or a predefined keyword.

    Override precedence (highest to lowest):
      env-specific args (data_path, save_path, cache_dir, metadata_dir)
      > --config_overrides
      > image_count=None default
      > base config / JSON snapshot
    """
    if not config_file and not predefined_keyword:
        raise ValueError("Pass either --config_file or --predefined_keyword")

    metadata_frames = load_metadata_dict_from_dir(metadata_dir) if metadata_dir else None

    # Build the layered override dict. image_count=None is the baseline default
    # so jobs use all available images unless explicitly restricted.
    runtime_overrides: dict = {"image_count": None}
    if config_overrides:
        runtime_overrides.update(config_overrides)
    # Env-specific args always win — they reflect actual node paths.
    if metadata_frames is not None:
        runtime_overrides["metadata_frames"] = metadata_frames
    if data_path:
        runtime_overrides["data_path"] = data_path
    if save_path:
        runtime_overrides["save_path"] = save_path
    if cache_dir:
        runtime_overrides["cache_dir"] = cache_dir

    if config_file:
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        config_dict.update(runtime_overrides)
        return TrainerConfig.from_dict(config_dict)

    return TrainerConfig.get_predefined_config(
        keyword=predefined_keyword, override=runtime_overrides
    )


def run_single_experiment(
    config_file: Optional[str] = None,
    predefined_keyword: Optional[str] = None,
    data_path: Optional[str] = None,
    save_path: Optional[str] = None,
    metadata_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    config_overrides: Optional[dict] = None,
) -> int:
    print("=" * 40)
    print("Running Single Experiment")
    print("=" * 40)
    print(f"Config file: {config_file}")
    print(f"Predefined keyword: {predefined_keyword}")
    print(f"Data path override: {data_path}")
    print(f"Save path override: {save_path}")
    print(f"Metadata directory: {metadata_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Config overrides: {config_overrides}")

    try:
        trainer_config = build_trainer_config(
            config_file=config_file,
            predefined_keyword=predefined_keyword,
            data_path=data_path,
            save_path=save_path,
            metadata_dir=metadata_dir,
            cache_dir=cache_dir,
            config_overrides=config_overrides,
        )
    except Exception as e:
        print(f"ERROR: Failed to build TrainerConfig: {e}")
        traceback.print_exc()
        return 1

    print("\n" + "=" * 40)
    print("Starting Training")
    print("=" * 40)

    try:
        trainer = Trainer(trainer_config)
        trainer.setup()
        trainer.train()
    except Exception:
        print("\n" + "=" * 40)
        print("ERROR: Training failed")
        print("=" * 40)
        traceback.print_exc()
        return 1

    print("\n" + "=" * 40)
    print("Training Completed Successfully")
    print("=" * 40)
    print(f"Best validation IoU: {trainer.metrics.best_iou:.4f}")
    print(f"Best epoch: {trainer.metrics.best_epoch}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a single training experiment from a config file or predefined keyword"
    )
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to JSON config produced by save_experiment_configs()")
    parser.add_argument("--predefined_keyword", type=str, default=None,
                        help="Keyword from TrainerConfig.predefined_config_keywords")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override TrainerConfig.data_path")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Override TrainerConfig.save_path")
    parser.add_argument("--metadata_dir", type=str, default=None,
                        help="Directory of per-split metadata CSVs")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Override TrainerConfig.cache_dir for dataset caching")
    parser.add_argument(
        "--config_overrides",
        type=str,
        default=None,
        help=(
            "JSON object of TrainerConfig field overrides, "
            "e.g. '{\"lr\": 0.0001, \"batch_size\": 8}'"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_overrides = None
    if args.config_overrides:
        try:
            config_overrides = parse_config_overrides(args.config_overrides)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    sys.exit(
        run_single_experiment(
            config_file=args.config_file,
            predefined_keyword=args.predefined_keyword,
            data_path=args.data_path,
            save_path=args.save_path,
            metadata_dir=args.metadata_dir,
            cache_dir=args.cache_dir,
            config_overrides=config_overrides,
        )
    )
