import tracemalloc

from scripts.run_single_experiment import run_single_experiment
from src.training.train_utils import LRSchedule
import os
from dotenv import load_dotenv

from src.training.transforms import YTransform

load_dotenv()
MODEL_DIR = os.environ.get("MODEL_DIR", "models")

tracemalloc.start()

run_single_experiment(predefined_keyword="default",
                      config_overrides={
                        "keyword": "UTAE_hedge_1.3_test",
                        "num_epochs": 2,
                        "save_path": os.path.join(MODEL_DIR, "hedgementation_1.3"),
                        "save_results": True,
                        "cache_dir": "/scratch-ssd/nathan/data/hedgementation_1.3/cache",
                        "datapoint_limit": 2995,
                        "overwrite": True,
                        "batch_size": 4,
                        "image_count": 10,
                        "load_y_id": False,
                        "num_workers": 4,
                        "eval_batch_size": None
                      })
