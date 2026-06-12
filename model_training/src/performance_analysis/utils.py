import json

from src.performance_analysis.evaluate_model import get_model_predictions
from src.training.train_utils import ModelFactory
from src.training.trainer_config import TrainerConfig
import os
from dotenv import load_dotenv
import torch
import pandas as pd

load_dotenv()
NUM_CHANNELS = int(os.getenv("NUM_CHANNELS", 10))

def setup_model_from_path_and_config(model_root:str,
                          config: TrainerConfig,
                          checkpoint:str = "best_model_params.pt"):
    
    
    model = ModelFactory.construct_model(
        backbone=config.backbone,
        num_buckets=config.num_buckets,
        num_channels=NUM_CHANNELS,
        additional_model_params=config.additional_model_params
    )

    state_dict = torch.load(
        os.path.join(model_root, "checkpoints", checkpoint),
        weights_only=True
    )
    model.load_state_dict(state_dict)
    return model

def get_model_prediction_from_model_path(model_dir: str,
                         ids: list[int],
                         checkpoint: str = "best_model_params.pt"):
    

    model_root = os.path.join("models", model_dir)

    with open(os.path.join(model_root, "params.json"), "r+") as infile:
        params = json.load(infile)
        config = TrainerConfig.from_dict(params)

    model = setup_model_from_path_and_config(
        model_root=model_root,
        config=config,
        checkpoint=checkpoint

    )

    metadata_frame = pd.DataFrame({"id":[id for id in ids], "ID_PATCH":[id for id in ids]})
    
    dataloader = config.setup_dataloader_from_config(metadata_frame)

    preds, labels = get_model_predictions(model, dataloader, y_transform=config.y_transform)

    return preds,labels