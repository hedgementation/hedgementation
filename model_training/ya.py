from scripts.run_single_experiment import run_single_experiment
from src.training.train_utils import LRSchedule
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_DIR = os.environ.get("MODEL_DIR", "models")


run_single_experiment(keyword="test",
                      override={
                        "keyword": "UTAE_hedge_1.3_test",
                        "save_path": os.path.join(MODEL_DIR, "hedgementation_1.3"),
                        "cache_dir": "/scratch-ssd/nathan/data/hedgementation_1.3/cache"
                      })
