import logging
import os
import json
import glob
import time 
import datetime
import geopandas as gpd
import pandas as pd
import traceback

from src.training.trainer_config import TrainerConfig
from src.training.trainer import Trainer

from src.training.experiments_registry import REGISTRY, deserialize_label_fn

from src.logging_utils import setup_logging

import time
import datetime

setup_logging()
logger = logging.getLogger(__name__)

def timer(function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = function(*args, **kwargs)
        end = time.perf_counter()
        dt = datetime.timedelta(seconds=(end - start))
        days = dt.days
        hours, remainder = divmod(dt.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        total_seconds = seconds + (dt.microseconds / 1_000_000)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{total_seconds:.6f}s")
        logger.info(f"[{function.__name__}] Runtime: " + " ".join(parts))
        return result
    return wrapper

def load_experiments(root_dir):
    result = {}

    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        if not os.path.isdir(item_path):
            continue

        metadata_path = os.path.join(item_path, "metadata")
        data_dict = {}
        if os.path.exists(metadata_path):
            for csv_file in glob.glob(os.path.join(metadata_path, "*.csv")):
                split = os.path.splitext(os.path.basename(csv_file))[0]
                try:
                    data_dict[split] = pd.read_csv(csv_file)
                    logger.info(f"✓ Successfully read {csv_file}")
                except Exception as e:
                    logger.error(f"  Error: {e}")
                    logger.error(f"✗ Failed reading {csv_file}")
                    return 1
        else:
            logger.warning(f"No folder 'metadata' in {item_path}")

        config_file = os.path.join(root_dir, f"{item}_config.json")
        config_dict = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                logger.info(f"✓ Successfully read {config_file}")
            except Exception as e:
                logger.error(f"  Error: {e}")
                logger.error(f"✗ Failed reading {config_file}")
                return 1
        else:
            logger.warning(f"No file {config_file}")

        # Migrate configs generated before data_path was renamed to dataset_root
        if "data_path" in config_dict:
            config_dict["dataset_root"] = config_dict.pop("data_path")
            logger.info(f"Note: migrated legacy 'data_path' key to 'dataset_root' in {config_file}")

        if config_dict.get("class_weights") == "autocalculated":
            config_dict["class_weights"] = None
        if config_dict.get("transfer"):
            config_dict["transfer"] = deserialize_label_fn(config_dict["transfer"])

        config_dict["metadata_frames"] = data_dict
        result[item] = config_dict

    return result


def save_experiments(root_dir, configs):
    used_exp_keywords = []
    used_config_keywords = []

    for exp_keyword, config in configs.items():
        metadata_dir = os.path.join(root_dir, f"{exp_keyword}", "metadata")
        try:
            os.makedirs(metadata_dir, exist_ok=True)
            logger.info(f"✓ Successfully generated directory {metadata_dir}")
        except Exception as e:
            logger.error(f"  Error: {e}")
            logger.error(f"✗ Failed generating directory {metadata_dir}")
            return 1

        for split, frame in config.metadata_frames.items():
            file_path = os.path.join(metadata_dir, f"{split}.csv")
            try:
                frame.to_csv(file_path, index=False)
                logger.info(f"✓ Successfully generated file {file_path}")
            except Exception as e:
                logger.error(f"  Error: {e}")
                logger.error(f"✗ Failed generating file {file_path}")
                return 1

        config_path = os.path.join(root_dir, f"{exp_keyword}_config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=4)
            logger.info(f"✓ Successfully generated file {config_path}")
        except Exception as e:
            logger.error(f"  Error: {e}")
            logger.error(f"✗ Failed generating file {config_path}")
            return 1

        if exp_keyword in used_exp_keywords:
            logger.error(f"✗ exp_keyword '{exp_keyword}' used twice. Keywords must be unique")
            return 1
        
        if config.keyword in used_config_keywords:
            logger.warning(f"config_keyword '{config.keyword}' used twice. Results might be override")

        used_exp_keywords.append(exp_keyword)
        used_config_keywords.append(config.keyword)
    return 0

def train(
    config_dict,
    output_dir=None,
    data_path=None,
    cache_dir=None,
    num_workers=None,
    batch_size=None,
    smoke_test=False,
):
    for exp, cfg in config_dict.items():
        exp_keyword = exp

        logger.info("=" * 10)
        logger.info(f"Training experiment '{exp_keyword}'...")
        logger.info("=" * 10)

        if output_dir is not None:
            cfg["save_path"]    = output_dir
            cfg["save_results"] = True
        else:
            cfg["save_results"] = False

        if data_path is not None:
            cfg["dataset_root"] = data_path

        if cache_dir is not None:
            cfg["cache_dir"] = cache_dir

        if num_workers is not None:
            cfg["num_workers"] = num_workers

        if batch_size is not None:
            cfg["batch_size"] = batch_size

        if smoke_test:
            logger.info("Smoke test: 1 epoch, truncated dataset, evals skipped")
            cfg["num_epochs"] = 1
            cfg["datapoint_limit"] = 2 * (batch_size or cfg.get("batch_size", 4))
            cfg["skip_untrained_eval"] = True
            cfg["skip_final_eval"] = True
            cfg["skip_full_dataset_inference"] = True

        result = run_single_experiment(cfg)
        if result != 0:
            return 1

    return 0

@timer
def run_single_experiment(config_dict):
    try:
        trainer_config = TrainerConfig(**config_dict)
        trainer = Trainer(trainer_config)
        trainer.setup()
        trainer.train(
            save_results=config_dict["save_results"],
            skip_untrained_eval=trainer_config.skip_untrained_eval,
            skip_final_eval=trainer_config.skip_final_eval,
            skip_full_dataset_inference=trainer_config.skip_full_dataset_inference,
        )
        logger.info("✓ Training Completed Successfully")
        logger.info(f"Best validation {config_dict['validation_metric']}: {trainer.metrics.best_metric:.4f}")
        logger.info(f"Best epoch: {trainer.metrics.best_epoch}")
        return 0
    except Exception as e:
        logger.error(f"  Error: {e}")
        traceback.print_exc()
        logger.error("✗ Training failed")
        return 1


def get_multi_experiment_overrides(keyword, nb_exp):
    return REGISTRY[keyword].multi_experiment_factory(nb_exp)