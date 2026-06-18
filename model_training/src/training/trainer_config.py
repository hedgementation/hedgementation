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
        dataset_root: Root directory of dataset
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
        provide_rpg_masks: Whether or not to provide RPG masks, using them to mask out pixels before loss calculation
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
    dataset_root: str = os.environ.get("DATASET_ROOT", None)
    reference_date: str = "2021-09-17"
    target_size: int = 128
    size_group: SizeGroup = SizeGroup.SMALL
    datapoint_limit: Optional[int] = None #Optional cap for datapoints to use, useful for testing 
    image_count: int = None
    batch_size: int = 16
    eval_batch_size: Optional[int] = 32
    use_memmap: bool = False
    metadata_frames: Optional[dict[str,gpd.GeoDataFrame]] = None
    load_X_cloud: bool = True
    load_y_id: bool = True
    cloud_threshold: Optional[float] = 0.2
    cloud_band: int = 0
    cache_dir: Optional[str] = os.environ.get("CACHE_DIR", None)
    io_manager_kwargs: Optional[dict] = None
    num_workers: int = 8
    datapoint_limit: int = None

    # Training configuration
    num_epochs: int = 100
    lr: float = 0.001
    weight_decay: float = 1e-4
    device: Optional[str] = None
    trainable_params: Optional[list[str]] = None
    skip_untrained_eval: bool = False
    skip_final_eval: bool = False
    skip_full_dataset_inference: bool = False

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
    provide_rpg_masks: bool = False
    rpg_mask_subdir: Optional[str] = None
    downsample_loss: bool = False
    downsample_mask_dir: Optional[str] = None
    validation_metric: str = "iou"

    # Scheduler and early stopping
    lr_scheduler: LRSchedule = LRSchedule.REDUCE_ON_PLATEAU
    lr_scheduler_args: dict[str,Any] = field(default_factory=dict)
    lr_scheduler_patience: int = 10
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0

    # Persistence configuration
    save_path: str = os.environ.get("MODEL_DIR", "models")
    keyword: Optional[str] = None
    save_results: bool = True
    overwrite: bool = True

    def __post_init__(self):
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
        
        if isinstance(self.size_group, int):
            self.size_group = SizeGroup(self.size_group)
        
        if self.metadata_frames:
            for k in self.metadata_frames:
                if isinstance(self.metadata_frames[k], str):
                    self.metadata_frames[k] = gpd.read_file(self.metadata_frames[k])
            if self.datapoint_limit:
                for k in self.metadata_frames:
                    self.metadata_frames[k] = self.metadata_frames[k][:min(len(self.metadata_frames[k]), self.datapoint_limit)]

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
            "keyword": self.keyword,
            "reference_date": self.reference_date,
            "backbone": self.backbone.value,
            "loss_function": self.loss_function.__name__,
            "normalization": self.normalization.value,
            "y_transform": self.y_transform.value,
            "num_buckets": self.num_buckets,
            "y_threshold": self.y_threshold,
            "class_weighted": self.class_weighted,
            "trainable_params": self.trainable_params,
            "skip_untrained_eval": self.skip_untrained_eval,
            "skip_final_eval": self.skip_final_eval,
            "skip_full_dataset_inference": self.skip_full_dataset_inference, 
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "target_size": self.target_size,
            "size_group": self.size_group.value,
            "num_workers": self.num_workers,
            "class_weights": (
                "autocalculated"
                if self.class_weighted and not self.class_weights
                else self.class_weights
            ),
            "use_memmap": self.use_memmap,
            "regression": self.regression,
            "num_epochs": self.num_epochs,
            "checkpoint_path": self.checkpoint_path,
            "augmentation": str(self.augmentation),
            "image_count": self.image_count,
            "device": self.device,
            "lr": self.lr,
            "lr_scheduler": self.lr_scheduler.name,
            "lr_scheduler_args": self.lr_scheduler_args,
            "weight_decay": self.weight_decay,
            "lr_scheduler_patience": self.lr_scheduler_patience,
            "early_stopping_patience": self.early_stopping_patience,
            "early_stopping_min_delta": self.early_stopping_min_delta,
            "inclusion_intervals": self.inclusion_intervals,
            "additional_model_params": self.additional_model_params,
            "datapoint_limit": self.datapoint_limit,
            "dataset_root": self.dataset_root,
            "save_results": self.save_results,
            "save_path": self.save_path,
            "provide_rpg_masks": self.provide_rpg_masks,
            "rpg_mask_subdir": self.rpg_mask_subdir,
            "downsample_loss": self.downsample_loss,
            "downsample_mask_dir": self.downsample_mask_dir,
            "load_X_cloud": self.load_X_cloud,
            "load_y_id": self.load_y_id,
            "cloud_threshold": self.cloud_threshold,
            "cloud_band": self.cloud_band,
            "cache_dir": self.cache_dir,
            "io_manager_kwargs": self.io_manager_kwargs,
            "validation_metric": self.validation_metric,
            "transfer": new,
        }
    
    def setup_dataloader_from_config(self,
                                 metadata_frame: Optional[gpd.GeoDataFrame] = None):
        from src.training.dataloader_utils import setup_dataloader
        if metadata_frame is None:
            metadata_frame = gpd.read_file(os.path.join(self.dataset_root, "metadata.geojson"))

        return setup_dataloader(
            metadata_frame=metadata_frame,
            data_path=self.dataset_root,
            image_count=self.image_count,
            num_buckets=self.num_buckets,
            normalization=self.normalization,
            y_transform=self.y_transform,
            inclusion_intervals=self.inclusion_intervals,
            batch_size=self.batch_size,
            provide_agriculture_masks=self.provide_rpg_masks,
            agriculture_mask_path=self.rpg_mask_subdir,
            downsample_loss=self.downsample_loss,
            downsample_mask_path=self.downsample_mask_dir,
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