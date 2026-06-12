"""
Configuration dataclass for the Trainer.

This module provides TrainerConfig, a dataclass that encapsulates all training
parameters, making it easy to serialize, validate, and pass configurations.
"""

import copy
import inspect
import ast
import os
from dataclasses import dataclass, field
import re
from typing import Any, ClassVar, List, Optional, Tuple

from hedgementation_utils.training.metadata_library import MetadataLibrary, SizeGroup
import torch

from src.training.dataset_dataloader import setup_dataloader
from src.training.train_utils import Backbones, LRSchedule
from src.training.transforms import Normalization, YTransform
import geopandas as gpd

from dotenv import load_dotenv

from src.training.losses import MaskedCrossEntropyLoss
from src.training.train_utils import Backbones
from src.training.transforms import Normalization, YTransform
from src.training.experiments_registry import REGISTRY
from src.performance_analysis.reg_meter import REGRESSION_METRICS, CLASSIFICATION_METRICS


load_dotenv()
DATASET_ROOT = os.environ.get("DATASET_ROOT", "")
NUM_TILEGROUPS = int(os.environ.get("NUM_TILEGROUPS",5))
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
VERSION = os.environ.get("VERSION", "1.2")

metadata_library = MetadataLibrary()

@dataclass
class TrainerConfig:
    """
    Configuration for the Trainer.

    This dataclass encapsulates all training parameters, providing validation
    and serialization capabilities.

    Attributes:
        backbone: Model architecture to use
        num_buckets: Number of classification buckets
        additional_model_params: Optional additional model parameters
        checkpoint_path: Path to checkpoint for resuming training
        data_path: Root directory of dataset
        image_count: Number of images per sample
        batch_size: Batch size for training
        use_memmap: Whether to use memory-mapped files
        metadata_frames: Optional pre-loaded metadata frames
        num_epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        device: Device for training ('cuda' or 'cpu')
        normalization: Normalization strategy
        y_transform: Target transform strategy
        y_threshold: Optional threshold for target values
        augmentation: List of augmentation names to apply
        inclusion_intervals: Optional intervals for included loss calculation
        loss_function: Loss function class
        class_weighted: Whether to use class weighting
        class_weights: Optional pre-computed class weights
        regression: Whether task is regression (vs classification)
        lr_scheduler_patience: Patience for learning rate scheduler
        early_stopping_patience: Patience for early stopping
        early_stopping_min_delta: Minimum delta for early stopping
        provide_agriculture_mask: Whether or not to mask out non-agricultural pixels before calculating loss
        validation_metric: Metric used to select the best model and drive early stopping.
            Classification options : "iou" (default), "f1", "precision", "recall".
            Regression options     : "mae", "mse", "rmse", "r2".
            Must be consistent with the `regression` flag.
        save_path: Directory to save results
        keyword: Keyword for naming the results directory
        save_results: Whether to save results permanently
        overwrite: Whether to overwrite existing results directory
    """
    transfer: Optional[dict] = None

    # Model configuration
    backbone: Backbones = Backbones.UNet3D
    num_buckets: int = 1
    additional_model_params: Optional[dict] = None
    checkpoint_path: Optional[str] = None

    # Data configuration
    data_path: str = None  # Will be set from environment
    size_group: SizeGroup = SizeGroup.SMALL
    image_count: int = 10
    batch_size: int = 16
    use_memmap: bool = False
    metadata_frames: Optional[dict[str,gpd.GeoDataFrame]] = None
    load_X_cloud: bool = True
    load_y_id: bool = False
    cloud_threshold: Optional[float] = 0.2
    cloud_band: int = 0
    cache_dir: Optional[str] = None
    io_manager_kwargs: Optional[dict] = None
    num_workers: int = 8

    # Training configuration
    num_epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 1e-4
    device: Optional[str] = None
    trainable_params: Optional[list[str]] = None

    # Preprocessing configuration
    normalization: Normalization = Normalization.MINMAX
    y_transform: YTransform = YTransform.UNIFORM
    y_threshold: Optional[float] = None
    augmentation: Optional[List[str]] = None
    inclusion_intervals: Optional[List[Tuple[float, float]]] = None

    # Loss configuration
    loss_function: type = torch.nn.CrossEntropyLoss
    class_weighted: bool = False
    class_weights: Optional[List[float]] = None
    regression: bool = False
    provide_agriculture_mask: bool = False
    agriculture_mask_path: Optional[str] = None
    downsample_loss: bool = False
    downsample_mask_path: Optional[str] = None
    validation_metric: str = "iou"

    # Scheduler and early stopping
    lr_scheduler: LRSchedule = LRSchedule.REDUCE_ON_PLATEAU
    lr_scheduler_args: dict[str,Any] = field(default_factory=dict)
    lr_scheduler_patience: int = 10
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0

    # Persistence configuration
    save_path: str = MODEL_DIR
    keyword: Optional[str] = None
    save_results: bool = True
    overwrite: bool = True

    def __post_init__(self):
        """Validate configuration parameters."""
        # Set default data_path from environment if not provided
        if self.data_path is None:
            self.data_path = os.environ.get("DATASET_ROOT", "")

        if isinstance(self.y_transform, str):
            self.y_transform = YTransform[self.y_transform.upper()]

        # Validate regression and y_transform compatibility
        if self.regression and self.y_transform != YTransform.NONE:
            raise ValueError("Cannot bucketize data for regression!")

        metric = self.validation_metric.lower()
        if self.regression and metric not in REGRESSION_METRICS:
            raise ValueError(
                f"validation_metric='{self.validation_metric}' is not valid for regression. "
                f"Choose one of: {sorted(REGRESSION_METRICS)}"
            )
        if not self.regression and metric not in CLASSIFICATION_METRICS:
            raise ValueError(
                f"validation_metric='{self.validation_metric}' is not valid for classification. "
                f"Choose one of: {sorted(CLASSIFICATION_METRICS)}"
            )
        self.validation_metric = metric  # normalise to lowercase

        # Check if save directory exists and handle overwrite
        if self.keyword and self.save_results:
            full_save_path = os.path.join(self.save_path, self.keyword)
            
        if isinstance(self.augmentation, str):
            self.augmentation = ast.literal_eval(self.augmentation)
            
        if isinstance(self.loss_function, str):
            if self.loss_function == "CrossEntropyLoss":
                self.loss_function = torch.nn.CrossEntropyLoss
            elif self.loss_function == "MSELoss":
                self.loss_function = torch.nn.MSELoss
            elif self.loss_function == "CombinedCrossEntropyMSELoss":
                from src.training.losses import CombinedCrossEntropyMSELoss

                self.loss_function = CombinedCrossEntropyMSELoss
            elif self.loss_function == "MaskedCrossEntropyLoss":
                self.loss_function = MaskedCrossEntropyLoss
            else:
                raise ValueError(f"Unknown loss function: {self.loss_function}")
        
        if isinstance(self.backbone, str):
            self.backbone = Backbones[self.backbone]

        if isinstance(self.normalization, str):
            self.normalization = Normalization[self.normalization.upper()]
        
        if isinstance(self.lr_scheduler, str):
            try:
                self.lr_scheduler = LRSchedule(self.lr_scheduler)
            except:
                self.lr_scheduler = LRSchedule[self.lr_scheduler]
        
        if self.metadata_frames:
            for k in self.metadata_frames:
                if isinstance(self.metadata_frames[k], str):
                    self.metadata_frames[k] = gpd.read_file(self.metadata_frames[k])

    @classmethod
    def from_dict(cls, data: dict):
        valid_keys = inspect.signature(cls).parameters
        return cls(**{k: v for k, v in data.items() if k in valid_keys})
    
    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary for serialization.

        Returns:
            dict: Configuration as a dictionary
        """
        if self.transfer is not None and not self.transfer.get("transfer"):
            new = copy.deepcopy(self.transfer)
            new["function"] = new["function"].__name__

        elif self.transfer is not None and self.transfer.get("transfer"):
            new = copy.deepcopy(self.transfer)
            new["params"]["f0_weights"] = "torch.Tensor"

        else:
            new = self.transfer 

        return {
            "backbone": self.backbone.value,
            "loss_function": self.loss_function.__name__,
            "normalization": self.normalization.value,
            "y_transform": self.y_transform.value,
            "num_buckets": self.num_buckets,
            "y_threshold": self.y_threshold,
            "class_weighted": self.class_weighted,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "class_weights": (
                "autocalculated"
                if self.class_weighted and not self.class_weights
                else self.class_weights
            ),
            "regression": self.regression,
            "num_epochs": self.num_epochs,
            "checkpoint_path": self.checkpoint_path,
            "augmentation": str(self.augmentation),
            "image_count": self.image_count,
            "lr": self.lr,
            "lr_scheduler": self.lr_scheduler.name,
            "lr_scheduler_args": self.lr_scheduler_args,
            "weight_decay": self.weight_decay,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "inclusion_intervals": self.inclusion_intervals,
            "additional_model_params": self.additional_model_params,
            "provide_agriculture_mask": self.provide_agriculture_mask,
            "agriculture_mask_path": self.agriculture_mask_path,
            "downsample_loss": self.downsample_loss,
            "downsample_mask_path": self.downsample_mask_path,
            "load_X_cloud": self.load_X_cloud,
            "load_y_id": self.load_y_id,
            "cloud_threshold": self.cloud_threshold,
            "cloud_band": self.cloud_band,
            "cache_dir": self.cache_dir,
            "validation_metric": self.validation_metric,
            "transfer": new,
        }
    
    def setup_dataloader_from_config(self,
                                 metadata_frame: Optional[gpd.GeoDataFrame] = None):
        if metadata_frame is None:
            metadata_frame = gpd.read_file(os.path.join(DATASET_ROOT, "metadata.geojson"))

        return setup_dataloader(
            metadata_frame=metadata_frame,
            data_path=DATASET_ROOT,
            image_count=self.image_count,
            num_buckets=self.num_buckets,
            normalization=self.normalization,
            y_transform=self.y_transform,
            inclusion_intervals=self.inclusion_intervals,
            batch_size=self.batch_size,
            provide_agriculture_masks=self.provide_agriculture_mask,
            agriculture_mask_path=self.agriculture_mask_path,
            downsample_loss=self.downsample_loss,
            downsample_mask_path=self.downsample_mask_path,
            load_X_cloud=self.load_X_cloud,
            load_y_id=self.load_y_id,
            cloud_threshold=self.cloud_threshold,
            cloud_band=self.cloud_band,
            cache_dir=self.cache_dir,
            transfer=self.transfer,
            is_train=False,
            shuffle=False,
            num_workers=self.num_workers,
        )

    predefined_config_keywords: ClassVar[list[str]] = list(REGISTRY.keys())

    @classmethod
    def get_predefined_config(cls, keyword: str, override: Optional[dict] = None):
        if keyword not in REGISTRY:
            raise ValueError(
                f"Unknown keyword '{keyword}'. "
                f"Available: {list(REGISTRY.keys())}"
            )
        config_dict = REGISTRY[keyword].config_factory(override)
        return cls(**config_dict)