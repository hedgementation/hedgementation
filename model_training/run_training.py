import json
import pandas as pd
import geopandas as gpd
import torch
from src.training.get_metadata_frames import get_default_frames, downsample_to_queried_frames, get_queried_frames, get_semantic_distance_frames
from src.training.train_utils import Backbones
from train import train_model
from torchvision.transforms import v2
from train import YTransform, Normalization
from src.training.losses import CombinedCrossEntropyMSELoss

from dotenv import load_dotenv
import os

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

RETURN_ALL_QUERY = 'ilevel_0 in ilevel_0'
SUBTROPICS_QUERY = 'thz_class in ["TRC5: Subtropics, cool", "TRC4: Subtropics, moderately cool"]'
TEMPERATE_QUERY = 'thz_class in ["TRC7: Temperate, cool", "TRC6: Temperate, moderately cool"]'

default_frames = get_default_frames(metadata)

far_frames = get_default_frames(metadata, near=False)

test_on_subtropic_frames = get_queried_frames(metadata,
                                              queries={
                                                  "train": RETURN_ALL_QUERY,
                                                  "valid": RETURN_ALL_QUERY,
                                                  "test": SUBTROPICS_QUERY
                                              })

test_on_temperate_frames = get_queried_frames(metadata,
                                              queries={
                                                  "train": RETURN_ALL_QUERY,
                                                  "valid": RETURN_ALL_QUERY,
                                                  "test": TEMPERATE_QUERY
                                              })

train_subtropic_test_temperate_frames = get_queried_frames(metadata,
                                              queries={
                                                  "train": SUBTROPICS_QUERY,
                                                  "valid": SUBTROPICS_QUERY,
                                                  "test": TEMPERATE_QUERY
                                              })

train_temperate_test_subtropic_frames = get_queried_frames(metadata,
                                              queries={
                                                  "train": TEMPERATE_QUERY,
                                                  "valid": TEMPERATE_QUERY,
                                                  "test": SUBTROPICS_QUERY
                                              })

train_same_size_as_subtropic_test = downsample_to_queried_frames(metadata,
                                              queries={
                                                  "train": SUBTROPICS_QUERY,
                                                  "valid": SUBTROPICS_QUERY,
                                                  "test": TEMPERATE_QUERY
                                              })

train_same_size_as_temperate_test = downsample_to_queried_frames(metadata,
                                              queries={
                                                  "train": TEMPERATE_QUERY,
                                                  "valid": TEMPERATE_QUERY,
                                                  "test": SUBTROPICS_QUERY
                                              })                                           

default_train_params = {
    "metadata_frames": default_frames,
    "backbone": Backbones.UTAE,
    "normalization": Normalization.MINMAX,
    "y_transform": YTransform.UNIFORM,
    "loss_function": torch.nn.CrossEntropyLoss,
    "num_buckets": 1,
    "class_weighted": True,
    "class_weights": [1.0, 3.0],
    "num_epochs": 100,
    "image_count": 10,
    "lr_scheduler_patience": 15,
    "early_stopping_patience": 100,
    "early_stopping_min_delta": 0.001,
    "batch_size": 8,
    "weight_decay": 1e-4,
    "augmentation": ["hflip", "vflip", "rotation"],
    "save_results": True,
    "keyword": "UTAE_hedge_1.1_near",
    "overwrite": False,
}

far_params = {
    **default_train_params,
    "keyword": "UTAE_hedge_1.1_far",
    "metadata_frames": far_frames
}

train_temperate_test_subtropic_params = {
    **default_train_params,
    "keyword": "UTAE_hedge_1.1_train_temperate_test_subtropic",
    "metadata_frames": train_temperate_test_subtropic_frames,
    "overwrite": True,
}

train_subtropic_test_temperate_params = {
    **default_train_params,
    "keyword": "UTAE_hedge_1.1_train_subtropic_test_temperate",
    "metadata_frames": train_subtropic_test_temperate_frames
}

#Randomly downsample to same size as these datasets to ensure the drop in performance is not from
#smaller dataset size
downsample_to_train_subtropic_test_temperate_params = {
    **default_train_params,
    "keyword": "UTAE_hedge_1.1_downsample_to_train_subtropic_test_temperate",
    "metadata_frames": train_same_size_as_subtropic_test
}

downsample_to_train_temperate_test_subtropic_params = {
    **default_train_params,
    "keyword": "UTAE_hedge_1.1_downsample_to_train_temperate_test_subtropic",
    "metadata_frames": train_same_size_as_temperate_test
}


larger_model_params =  {
    **default_train_params,
    "additional_model_params": {
        "encoder_widths": [64, 128, 256, 512],
        "decoder_widths": [64, 128, 256, 512]
    },
    "batch_size": 8,
    "keyword": "UTAE_hedge_1.0_increase_encoder_decoder_size",
}


test_params = {
        **default_train_params,
        "num_epochs":10,
        "keyword": "test",
        "overwrite": True,
    }




train_model(**downsample_to_train_subtropic_test_temperate_params)