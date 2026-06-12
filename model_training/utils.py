import os
import json
import re
import glob
import geopandas as gpd
import pandas as pd
import traceback

from src.training.trainer_config import TrainerConfig
from src.training.trainer import Trainer

from src.training.experiments_registry import REGISTRY, deserialize_label_fn

def load_experiments(root_dir):
    result = {}
    pattern = re.compile(r"experiment_(\d+)$")

    for item in sorted(os.listdir(root_dir)):
        item_path = os.path.join(root_dir, item)
        match = pattern.match(item)
        if not match:
            print(f"Warning: {item_path} is not named 'experiment_{{i}}'. Skipping.")
            continue
        i = int(match.group(1))

        metadata_path = os.path.join(item_path, "metadata")
        data_dict = {}
        if os.path.exists(metadata_path):
            for csv_file in glob.glob(os.path.join(metadata_path, "*.csv")):
                split = os.path.splitext(os.path.basename(csv_file))[0]
                try:
                    data_dict[split] = pd.read_csv(csv_file)
                    print(f"✓ Successfully read {csv_file}")
                except Exception as e:
                    print(f"  Error: {e}")
                    print(f"✗ Failed reading {csv_file}")
                    return 1
        else:
            print(f"Warning: No folder 'metadata' in {item_path}")

        config_file = os.path.join(root_dir, f"experiment_{i}_config.json")
        config_dict = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config_dict = json.load(f)
                print(f"✓ Successfully read {config_file}")
            except Exception as e:
                print(f"  Error: {e}")
                print(f"✗ Failed reading {config_file}")
                return 1
        else:
            print(f"Warning: No file {config_file}")

        if config_dict.get("class_weights") == "autocalculated":
            config_dict["class_weights"] = None
        if config_dict.get("transfer"):
            config_dict["transfer"] = deserialize_label_fn(config_dict["transfer"])

        config_dict["metadata_frames"] = data_dict
        result[i] = config_dict

    return result


def save_experiments(root_dir, configs, matching_path):
    matching = {}

    for i, (exp_keyword, config) in enumerate(configs.items()):
        metadata_dir = os.path.join(root_dir, f"experiment_{i}", "metadata")
        try:
            os.makedirs(metadata_dir, exist_ok=True)
            print(f"✓ Successfully generated directory {metadata_dir}")
        except Exception as e:
            print(f"  Error: {e}")
            print(f"✗ Failed generating directory {metadata_dir}")
            return 1

        for split, frame in config.metadata_frames.items():
            file_path = os.path.join(metadata_dir, f"{split}.csv")
            try:
                frame.to_csv(file_path, index=False)
                print(f"✓ Successfully generated file {file_path}")
            except Exception as e:
                print(f"  Error: {e}")
                print(f"✗ Failed generating file {file_path}")
                return 1

        config_path = os.path.join(root_dir, f"experiment_{i}_config.json")
        try:
            with open(config_path, "w") as f:
                json.dump(config.to_dict(), f, indent=4)
            print(f"✓ Successfully generated file {config_path}")
        except Exception as e:
            print(f"  Error: {e}")
            print(f"✗ Failed generating file {config_path}")
            return 1

        if exp_keyword in [v["exp_keyword"] for v in matching.values()]:
            print(f"Error: exp_keyword '{exp_keyword}' used twice. Keywords must be unique")
            return 1

        matching[i] = {
            "exp_keyword": exp_keyword,
            "config_keyword": config.keyword,
        }

    matching_file = os.path.join(root_dir, matching_path)
    try:
        with open(matching_file, "w") as f:
            json.dump(matching, f, indent=4)
        print(f"✓ Successfully generated matching file {matching_file}")
    except Exception as e:
        print(f"  Error: {e}")
        print(f"✗ Failed generating matching file {matching_file}")
        return 1

    return 0

def train(
    experiments_dir,
    matching_path,
    config_dict,
    output_dir=None,
    data_path=None,
    cache_dir=None,
    num_workers=None,
    batch_size=None,
):
    matching_file = os.path.join(experiments_dir, matching_path)
    try:
        with open(matching_file, "r") as f:
            matching = json.load(f)
        print(f"✓ Successfully read {matching_file}")
    except Exception as e:
        print(f"  Error: {e}")
        print(f"✗ Failed reading {matching_file}")
        return 1

    for num_exp, cfg in config_dict.items():
        entry = matching.get(str(num_exp), {})
        exp_keyword    = entry.get("exp_keyword",    f"experiment_{num_exp}")
        config_keyword = entry.get("config_keyword", None)

        print("\n")
        print("=" * 10)
        print(f"Training experiment '{exp_keyword}'...")
        print("=" * 10)

        # Restore config_keyword so Trainer names the save dir correctly.
        if config_keyword:
            cfg["keyword"] = config_keyword

        if output_dir is not None:
            cfg["save_path"]    = output_dir
            cfg["save_results"] = True
        else:
            cfg["save_results"] = False

        if data_path is not None:
            cfg["data_path"] = data_path

        if cache_dir is not None:
            cfg["cache_dir"] = cache_dir

        if num_workers is not None:
            cfg["num_workers"] = num_workers

        if batch_size is not None:
            cfg["batch_size"] = batch_size

        result = run_single_experiment(cfg)
        if result != 0:
            return 1

    return 0


def run_single_experiment(config_dict):
    try:
        trainer_config = TrainerConfig(**config_dict)
        trainer = Trainer(trainer_config)
        trainer.setup()
        model, metrics = trainer.train(save_results=config_dict["save_results"])
        print("✓ Training Completed Successfully")
        print(f"Best validation {config_dict['validation_metric']}: {trainer.metrics.best_metric:.4f}")
        print(f"Best epoch: {trainer.metrics.best_epoch}")
        return 0
    except Exception as e:
        print(f"  Error: {e}")
        traceback.print_exc()
        print("✗ Training failed")
        return 1


def get_multi_experiment_overrides(keyword, nb_exp):
    return REGISTRY[keyword].multi_experiment_factory(nb_exp)