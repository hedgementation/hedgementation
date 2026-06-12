"""
Main Trainer class for model training.

This module provides the Trainer class, which orchestrates the complete training
lifecycle including setup, training loop, evaluation, and results persistence.
"""

import json
import os
import random
import shutil
import time
from functools import partial
from tempfile import TemporaryDirectory
from typing import Tuple

import geopandas as gpd
import numpy as np
import torch
from dotenv import load_dotenv
from git import Repo
from torchvision.transforms import v2
from src.training.train_utils import LRSchedule

from src.performance_analysis.evaluate_model import (
    evaluate_model,
    evaluate_per_patch_performance,
    visualize_model_predictions,
)
from src.performance_analysis.reg_meter import RegMeter, higher_is_better
from hedgementation_utils.metrics.seg_meter import SegMeter
from src.training.checkpointer import Checkpointer
from src.training.datastore_utils import DataStore
from src.training.dataset_dataloader import setup_dataloaders
from src.training.early_stopping import EarlyStopping
from src.training.metrics_tracker import MetricsTracker
from src.training.train_utils import LRSchedulerFactory, ModelFactory, adapt_state_dict, calculate_class_weights, calculate_excluded_loss
from src.training.trainer_config import TrainerConfig
from hedgementation_utils.training.metadata_library import MetadataLibrary
from src.training.transforms import AUGMENTATION_TRANSFORMS, YTransform, transform_y


load_dotenv()
NUM_CHANNELS = int(os.environ["NUM_CHANNELS"])
DATASET_ROOT = os.environ["DATASET_ROOT"]

TRAIN_FOLDS = json.loads(os.environ.get("TRAIN_FOLDS", "[0,1,2]"))
VALID_FOLDS = json.loads(os.environ.get("VALID_FOLDS", "[3]"))
TEST_FOLDS = json.loads(os.environ.get("TEST_FOLDS", "[4]"))


class Trainer:
    """
    Main training orchestrator.

    Manages the complete training lifecycle including setup, training loop,
    evaluation, and results persistence.

    Attributes:
        config: TrainerConfig object with all training parameters
        model: The model being trained
        dataloaders: Train/valid/test dataloaders
        eval_dataloaders: Evaluation dataloaders (without inclusion_intervals)
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function
        device: Device for training ('cuda' or 'cpu')
        early_stopping: EarlyStopping instance
        metrics: MetricsTracker instance
        checkpointer: Checkpointer instance
        evaluate_model_func: Function for model evaluation
    """

    def __init__(self, config: TrainerConfig):
        """
        Initialize trainer with configuration.

        Args:
            config: TrainerConfig object with all training parameters
        """
        self.config = config
        self.size_group = config.size_group

        # These will be initialized in setup()
        self.model = None
        self.dataloaders = None
        self.eval_dataloaders = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = None
        self.early_stopping = None

        # Helper objects
        self.metrics = MetricsTracker(validation_metric=config.validation_metric)
        self.checkpointer = None  # Created during training
        self.evaluate_model_func = None
        self.epoch_callback = None  # Optional[Callable[[int, float], None]] — called with (epoch, val_iou)

        self._print_config()
        self._validate_config()

    def setup(self):
        """
        Initialize all training components.

        This method:
        1. Sets random seeds
        2. Initializes model
        3. Creates dataloaders
        4. Sets up optimizer and scheduler
        5. Configures loss function
        6. Initializes early stopping
        """
        self._set_random_seeds()
        self._setup_device()
        self._setup_model()
        self._setup_dataloaders()
        self._setup_optimizer_and_scheduler()
        self._setup_loss_function()
        self._setup_early_stopping()
        self._setup_evaluation()

    def train(self,
              save_results: bool = None,
              skip_untrained_eval: bool = False,
              skip_per_patch_eval: bool = False) -> Tuple[torch.nn.Module, dict]:
        """
        Execute the complete training process.

        This method orchestrates:
        1. Training loop execution
        2. Checkpoint management
        3. Model evaluation (untrained, final, best)
        4. Results persistence

        Args:
            save_results: Override config.save_results if provided

        Returns:
            tuple: (trained_model, evaluation_metrics)
        """
        save_results = (
            save_results if save_results is not None else self.config.save_results
        )

        permanent_path = self._get_permanent_save_path()

        if os.path.isdir(permanent_path) and self.config.save_results and not self.config.overwrite:
                raise FileExistsError(
                    f"Directory {permanent_path} already exists and overwrite=False"
                )
        with TemporaryDirectory() as tempdir:
            try:
                # Setup temporary directory structure
                self._setup_temp_directory(tempdir)
                self.checkpointer = Checkpointer(
                    os.path.join(tempdir, "checkpoints"), save_interval=20
                )

                # Save configuration
                self._save_config(tempdir)

                # Evaluate untrained model
                if not skip_untrained_eval:
                    self._evaluate_untrained(tempdir)

                # Main training loop
                start_time = time.time()
                self._run_training_loop()
                self._print_training_summary(time.time() - start_time)

                # Save metrics and graphs
                self._save_training_artifacts(tempdir)

                # Evaluate final and best models
                best_model_eval = self._evaluate_final_and_best(tempdir)

                # Evaluate per-patch performance
                if not skip_per_patch_eval:
                    self._evaluate_per_patch_performance(tempdir)

                # Conditionally move to permanent storage
                if save_results:
                    self._persist_results(tempdir)
            except Exception as e:
                print(f"Training interrupted by exception {e}")
                if save_results:
                    print("Saving existing results...")
                    self._persist_results(tempdir)
                raise e


        return self.model, best_model_eval

    # ========== Private Setup Methods ==========

    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        try:
            seed = int(os.environ["RANDOM_SEED"])
        except (KeyError, ValueError):
            seed = 15

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _setup_device(self):
        """Configure training device."""
        if self.config.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device
        print(f"Using {self.device} device")

    
    def _load_from_checkpoint(self,
                            model: torch.nn.Module,
                            checkpoint_path: str):
        state_dict = torch.load(checkpoint_path, weights_only=True)
        state_dict = adapt_state_dict(model, state_dict)
        if len(state_dict) == 0:
            print(f"WARNING: No overlapping keys found for checkpoint at {checkpoint_path}")
        model.load_state_dict(state_dict)

    def _setup_model(self):
        """Initialize and configure the model."""
        self.model = ModelFactory.construct_model(
            backbone=self.config.backbone,
            num_channels=NUM_CHANNELS,
            num_buckets=self.config.num_buckets,
            additional_model_params=self.config.additional_model_params,
        )

        if self.config.checkpoint_path:
            self._load_from_checkpoint(
                model=self.model,
                checkpoint_path=self.config.checkpoint_path
            )

        if self.config.trainable_params is not None:
            for (name,param) in self.model.named_parameters():
                if name not in self.config.trainable_params:
                    param.requires_grad = False

        self.model = self.model.to(self.device)

    def _setup_dataloaders(self):
        """Create training and validation dataloaders."""
        if not self.config.metadata_frames:
            metadata = gpd.read_file(f"{DATASET_ROOT}/metadata.geojson")
            self.config.metadata_frames = {
                "valid": metadata[metadata["fold"].isin(VALID_FOLDS)],
                "train": metadata[metadata["fold"].isin(TRAIN_FOLDS)],
            }

        # Store train metadata in DataStore
        DataStore.set_data("train_metadata", self.config.metadata_frames["train"])
        DataStore.set_data("data_path", self.config.data_path)

        # Setup augmentation
        augmentation_func = self._create_augmentation_func()

        # Create dataloaders
        self.dataloaders = setup_dataloaders(
            metadata_frames=self.config.metadata_frames,
            data_path=self.config.data_path,
            image_count=self.config.image_count,
            num_buckets=self.config.num_buckets,
            inclusion_intervals=self.config.inclusion_intervals,
            normalization=self.config.normalization,
            y_threshold=self.config.y_threshold,
            y_transform=self.config.y_transform,
            augmentation_func=augmentation_func,
            batch_size=self.config.batch_size,
            use_memmap=self.config.use_memmap,
            provide_agriculture_masks=self.config.provide_agriculture_mask,
            agriculture_mask_path=self.config.agriculture_mask_path,
            downsample_loss=self.config.downsample_loss,
            downsample_mask_path=self.config.downsample_mask_path,
            load_X_cloud=self.config.load_X_cloud,
            load_y_id=self.config.load_y_id,
            cloud_threshold=self.config.cloud_threshold,
            cloud_band=self.config.cloud_band,
            cache_dir=self.config.cache_dir,
            io_manager_kwargs=self.config.io_manager_kwargs,
            transfer=self.config.transfer,
            num_workers=self.config.num_workers,
        )

        # Setup evaluation dataloaders (without inclusion_intervals)
        if self.config.inclusion_intervals:
            metadata = gpd.read_file(f"{DATASET_ROOT}/metadata.geojson")
            self.eval_dataloaders = setup_dataloaders(
                metadata_frames=self.config.metadata_frames,
                data_path=self.config.data_path,
                image_count=self.config.image_count,
                num_buckets=self.config.num_buckets,
                inclusion_intervals=None,
                normalization=self.config.normalization,
                y_threshold=self.config.y_threshold,
                y_transform=self.config.y_transform,
                batch_size=self.config.batch_size,
                use_memmap=self.config.use_memmap,
                provide_agriculture_masks=self.config.provide_agriculture_mask,
                agriculture_mask_path=self.config.agriculture_mask_path,
                downsample_loss=self.config.downsample_loss,
                downsample_mask_path=self.config.downsample_mask_path,
                load_X_cloud=self.config.load_X_cloud,
                load_y_id=self.config.load_y_id,
                cloud_threshold=self.config.cloud_threshold,
                cloud_band=self.config.cloud_band,
                cache_dir=self.config.cache_dir,
                io_manager_kwargs=self.config.io_manager_kwargs,
                num_workers=self.config.num_workers,
                transfer=self.config.transfer,
            )
        else:
            self.eval_dataloaders = self.dataloaders

    def _create_augmentation_func(self):
        """Create augmentation function from config."""
        if not self.config.augmentation:
            return None

        transforms = [
            AUGMENTATION_TRANSFORMS[aug] for aug in self.config.augmentation
        ]
        return v2.Compose(transforms) if len(transforms) > 1 else transforms[0]

    def _setup_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = LRSchedulerFactory.construct_lr_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.config.lr_scheduler,
            scheduler_args=self.config.lr_scheduler_args
        )

    def _setup_loss_function(self):
        """Configure the loss function."""
        
        needs_pixel_loss = (
            self.config.inclusion_intervals is not None
            or self.config.provide_agriculture_mask
            or self.config.downsample_loss
            or (self.config.transfer is not None and self.config.transfer.get("transfer"))
        )

        if self.config.transfer is not None and self.config.transfer.get("transfer"):
            self.config.transfer["params"]["f0_max"] = self.config.transfer["params"]["f0_weights"].max().item()

        if self.config.class_weighted:
            if not self.config.class_weights:
                class_weights = calculate_class_weights(
                    self.dataloaders["train"],
                    num_classes=self.config.num_buckets + 1,
                    device=self.device,
                )
            else:
                class_weights = self.config.class_weights

            reduction = "none" if needs_pixel_loss else "mean"
            self.criterion = self.config.loss_function(
                weight=torch.tensor(class_weights).to(self.device), reduction=reduction
            )
        else:
            if needs_pixel_loss:
                self.criterion = self.config.loss_function(reduction="none")
            else:
                self.criterion = self.config.loss_function()

    def _setup_early_stopping(self):
        """Initialize early stopping."""
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            verbose=False,
        )

    def _setup_evaluation(self):
        """Configure evaluation function based on task type."""
        self.evaluate_model_func = partial(
            evaluate_model,
            regression=self.config.regression
        )

    def _make_meter(self):
        """
        Instantiate the appropriate meter for the current task.

        Returns:
            RegMeter  if regression=True
            SegMeter  if regression=False
        """
        if self.config.regression:
            return RegMeter()
        return SegMeter(num_classes=self.config.num_buckets + 1)

    # ========== Training Loop Methods ==========

    def _run_training_loop(self):
        """Execute the main training loop."""
        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch}/{self.config.num_epochs - 1}")
            print("-" * 10)

            # Train and validate
            for phase in ["train", "valid"]:
                epoch_loss, epoch_metric = self._run_epoch(phase)

                # Record metrics
                if phase == "train":
                    self.metrics.record_train_metrics(epoch_loss, epoch_metric)
                else:
                    is_best = self.metrics.record_val_metrics(epoch_loss, epoch_metric)

                    # Handle validation-specific logic
                    self._handle_validation_epoch(epoch, is_best, epoch_loss)

                    if self.epoch_callback is not None:
                        self.epoch_callback(epoch, epoch_metric)

            # Check early stopping
            if self.early_stopping.early_stop:
                print("Stopping early due to EarlyStopping count exceeding patience")
                break

    def _run_epoch(self, phase: str) -> Tuple[float, float]:
        """
        Run a single training or validation epoch.

        Args:
            phase: "train" or "valid"

        Returns:
            tuple: (average_loss, validation_metric_value)
        """
        self.model.train() if phase == "train" else self.model.eval()

        running_loss = 0.0
        batch_count = 0
        batch_metrics = []

        for batch_idx, item in enumerate(self.dataloaders[phase]):
            batch_count += 1
            loss, preds, labels = self._process_batch(item, phase)

            running_loss += loss
            
            if self.config.validation_metric != "loss":
                batch_meter = self._make_meter()
                if self.config.regression:
                    batch_meter.update(preds, labels)
                else:
                    batch_meter.update_batch(preds, labels)
                batch_metrics.append(batch_meter.ask_metrics()[self.config.validation_metric])

            if batch_idx % 10 == 0:
                print(
                    f"  Batch {batch_idx}/{len(self.dataloaders[phase])-1}, "
                    f"Loss: {loss:.4f}"
                )

        epoch_loss = running_loss / batch_count
        if self.config.validation_metric == "loss":
            epoch_metric = epoch_loss 
        else:
            epoch_metric = float(np.mean(batch_metrics))

        print(
            f"{phase.capitalize()} Loss: {epoch_loss:.4f}, "
            f"{self.config.validation_metric.upper()}: {epoch_metric:.4f}"
        )

        return epoch_loss, epoch_metric

    def _process_batch(self, item, phase: str) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Process a single batch.

        Args:
            item: Batch data from dataloader
            phase: "train" or "valid"

        Returns:
            tuple: (loss_value, predictions_cpu, labels_cpu)
                - predictions_cpu: argmax class indices for classification,
                                   raw output values for regression
                - labels_cpu: ground-truth labels
        """
        has_masks = (
            self.config.provide_agriculture_mask
            or self.config.downsample_loss
            or self.config.load_y_id
            or (self.config.transfer is not None and self.config.transfer.get("transfer"))
        )
        if has_masks:
            input_tuple, labels, masks_raw = item
            masks = {k: v.to(self.device) for k, v in masks_raw.items()}
        else:
            input_tuple, labels = item
            masks = {}

        if self.config.load_X_cloud:
            inputs, dates, X_cloud = input_tuple
            X_cloud = X_cloud.to(self.device)
        else:
            inputs, dates = input_tuple
    

        # Handle inclusion intervals
        labels_continuous = None
        if self.config.inclusion_intervals and not self.config.regression:
            labels_continuous = labels.clone().to(self.device)
            labels = transform_y(
                labels, self.config.y_transform, self.config.num_buckets
            ).to(self.device)

        inputs = inputs.to(self.device)
        dates = dates.to(self.device)
        labels = labels.to(self.device)

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == "train"):
            outputs = self.model(inputs, batch_positions=dates)

            # Calculate loss
            if self.config.inclusion_intervals:
                loss, _ = calculate_excluded_loss(
                    labels_continuous,
                    labels,
                    outputs,
                    self.config.inclusion_intervals,
                    self.criterion,
                )
            else:
                loss = self.criterion(outputs, labels)
                if masks:
                    loss = self._apply_masks_and_normalize(loss, masks)


            # Backward pass
            if phase == "train":
                loss.backward()
                self.optimizer.step()

        if self.config.regression:
            preds_cpu = outputs.detach().cpu()
            labels_cpu = labels.cpu()
        else:
            ag_mask = masks.get("ag") if masks else None
            if ag_mask is not None:
                preds_cpu = torch.argmax(outputs, dim=1)[ag_mask].detach().cpu()
                labels_cpu = labels[ag_mask].cpu()
            else:
                preds_cpu = torch.argmax(outputs, dim=1).detach().cpu()
                labels_cpu = labels.cpu()

        loss_val = loss.item()

        del outputs, inputs, dates, labels
        if labels_continuous is not None:
            del labels_continuous

        return loss_val, preds_cpu, labels_cpu

    def _apply_masks_and_normalize(self, loss, masks: dict):
        """
        Apply agriculture mask and/or downsample mask to loss, then normalize.

        Steps:
          1. Zero out pixels excluded by the agriculture mask (if present).
          2. Randomly zero out excess pixels per sample so the active count matches
             the downsample mask count (if present and active > target).
          3. Normalize by the total number of remaining active pixels.

        Args:
            loss:  Per-pixel loss tensor of shape (B, H, W).
            masks: Dict with optional keys:
                   "ag" — boolean agriculture mask (B, H, W), True = include
                   "ds" — boolean downsample reference mask (B, H, W)

        Returns:
            Scalar loss.
        """
        if loss.dim() == 0:
            return loss

        B, H, W = loss.shape
        ag_mask = masks.get("ag")
        ds_mask = masks.get("ds")
        pw = masks.get("pw")

        # Boolean tensor tracking which pixels are still active
        active = torch.ones(B, H, W, dtype=torch.bool, device=loss.device)

        if ag_mask is not None:
            active = active & ag_mask

        if ds_mask is not None:
            active = self._apply_downsample(active, ds_mask)

        if pw is not None:
            pw = pw.to(device=loss.device, dtype=loss.dtype)
            loss = loss * pw 
            max_f0 = self.config.transfer["params"]["f0_max"]
            denom = active.float().sum() * max_f0

        else:
            denom = active.float().sum()

        loss = loss * active.float()

        if denom > 0:
            return loss.sum() / denom
        else:
            return torch.tensor(0.0, device=loss.device, requires_grad=True)

    def _apply_downsample(self, active: torch.Tensor, ds_mask: torch.Tensor) -> torch.Tensor:
        """
        Randomly deactivate pixels per sample so the active count matches the
        downsample mask count.  If active count <= target count, nothing changes.

        Args:
            active:  Boolean tensor (B, H, W) of currently active pixels.
            ds_mask: Boolean tensor (B, H, W) defining the target count per sample.

        Returns:
            Updated active tensor (B, H, W).
        """
        result = active.clone()
        for i in range(active.shape[0]):
            target_count = int(ds_mask[i].sum().item())
            active_indices = result[i].nonzero(as_tuple=False)  # (N, 2)
            active_count = len(active_indices)

            if active_count > target_count:
                keep = torch.randperm(active_count, device=active.device)[:target_count]
                new_active = torch.zeros_like(result[i])
                kept_indices = active_indices[keep]
                new_active[kept_indices[:, 0], kept_indices[:, 1]] = True
                result[i] = new_active

        return result
    
    def _calculate_masked_metric(self, predictions, labels, mask, num_buckets) -> float:
            pred_flat = predictions[mask]
            label_flat = labels[mask]

            if pred_flat.numel() == 0:
                return 0.0

            meter = SegMeter(num_classes=num_buckets + 1)
            meter.update(pred_flat.unsqueeze(0), label_flat.unsqueeze(0))

            return meter.ask_metrics()[self.config.validation_metric]
    
    def _handle_validation_epoch(self, epoch: int, is_best: bool, epoch_loss: float):
        """Handle validation epoch logic (checkpointing, scheduling, early stopping)."""
        # Update scheduler
        if self.config.lr_scheduler == LRSchedule.REDUCE_ON_PLATEAU:
            self.scheduler.step(epoch_loss)
        else:
            self.scheduler.step()

        # Save best model
        if is_best:
            self.checkpointer.save_best(self.model)
            print(
                f"New best model saved to temp dir with validation "
                f"{self.config.validation_metric.upper()}: "
                f"{self.metrics.best_metric:.4f}"
            )

        # Save periodic checkpoint
        self.checkpointer.save_periodic(self.model, epoch, self.config.num_epochs)

        # Check early stopping
        self.early_stopping(epoch_loss, self.model)

    # ========== Evaluation Methods ==========

    def _evaluate_untrained(self, tempdir: str):
        """Evaluate model before training."""
        self.evaluate_model_func(
            self.model,
            self.eval_dataloaders,
            "untrained_metrics",
            tempdir,
            self.config.num_buckets,
            self.config.y_transform,
            save_extra_info=False,
            device=self.device,
        )

    def _evaluate_final_and_best(self, tempdir: str) -> dict:
        """
        Evaluate final and best models.

        Returns:
            dict: Best model evaluation metrics
        """
        # Evaluate final model
        self.evaluate_model_func(
            self.model,
            self.eval_dataloaders,
            "final_trained_model",
            tempdir,
            self.config.num_buckets,
            self.config.y_transform,
            save_extra_info=True,
            device=self.device,
        )
        self.checkpointer.save_most_recent(self.model)

        # Evaluate best model
        self.model = self.checkpointer.load_best(self.model)
        best_model_eval = self.evaluate_model_func(
            self.model,
            self.eval_dataloaders,
            "best_trained_model",
            tempdir,
            self.config.num_buckets,
            self.config.y_transform,
            save_extra_info=True,
            device=self.device,
        )

        if not self.config.regression:
            visualize_model_predictions(
                self.model,
                self.eval_dataloaders["valid"],
                self.device,
                self.config.y_transform,
                tempdir,
                num_samples=9,
            )

        return best_model_eval
    
    def _get_full_dataset_frame(self):
        """Return all metadata rows filtered to the trainer's size_group."""
        library = MetadataLibrary()
        return library._resolve_metadata(None, self.size_group)

    def _evaluate_per_patch_performance(self, tempdir: str):
        """Evaluate model performance on each patch in the full dataset."""
        full_dataset_frame = self._get_full_dataset_frame()
        augmentation_func = self._create_augmentation_func()

        full_dataset_dataloader = setup_dataloaders(
            metadata_frames={"test": full_dataset_frame},
            data_path=self.config.data_path,
            image_count=self.config.image_count,
            num_buckets=self.config.num_buckets,
            inclusion_intervals=self.config.inclusion_intervals,
            normalization=self.config.normalization,
            y_threshold=self.config.y_threshold,
            y_transform=self.config.y_transform,
            augmentation_func=augmentation_func,
            batch_size=self.config.batch_size,
            use_memmap=self.config.use_memmap,
            provide_agriculture_masks=self.config.provide_agriculture_mask,
            agriculture_mask_path=self.config.agriculture_mask_path,
            downsample_loss=self.config.downsample_loss,
            downsample_mask_path=self.config.downsample_mask_path,
            load_X_cloud=self.config.load_X_cloud,
            load_y_id=self.config.load_y_id,
            cloud_threshold=self.config.cloud_threshold,
            cloud_band=self.config.cloud_band,
            cache_dir=self.config.cache_dir,
            io_manager_kwargs=self.config.io_manager_kwargs,
            num_workers=self.config.num_workers,
            transfer=self.config.transfer,
        )

        evaluate_per_patch_performance(
            self.model,
            full_dataset_frame,
            dataloader=full_dataset_dataloader["test"],
            save_path=tempdir,
            regression=self.config.regression,
            num_buckets=self.config.num_buckets,
            file_name="best_model_per_patch_performance",
            device=self.device,
            save_data=True,
        )

    # ========== Persistence Methods ==========

    def _setup_temp_directory(self, tempdir: str):
        """Create temporary directory structure."""
        os.makedirs(os.path.join(tempdir, "imgs"))
        os.makedirs(os.path.join(tempdir, "checkpoints"))
        os.makedirs(os.path.join(tempdir, "npys"))

    def _save_config(self, tempdir: str):
        """Save training configuration to JSON."""
        now_timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())
        )
        most_recent_commit = Repo(".").head.commit

        params_path = os.path.join(tempdir, "params.json")
        with open(params_path, "w+") as outfile:
            json.dump(
                {
                    "date run": now_timestamp,
                    "commit": most_recent_commit.hexsha,
                    "commit_msg": most_recent_commit.message,
                    **self.config.to_dict(),
                },
                outfile,
                indent=4,
            )

    def _save_dataframes(self, tempdir: str):
        frame_directory = os.path.join(tempdir, "metadata_frames")
        os.makedirs(frame_directory, exist_ok=True)
        frames = self.config.metadata_frames
        for split in frames:
            frames[split].to_file(os.path.join(
                frame_directory,
                f"{split}_frame.geojson"
            )
        )

    def _save_training_artifacts(self, tempdir: str):
        """Save graphs and metrics CSV."""
        self.metrics.save_graphs(tempdir)
        self.metrics.save_csv(os.path.join(tempdir, "metrics_per_epoch.csv"))
    
    def _get_permanent_save_path(self):
        now_timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(time.time())
        )
        permanent_save_path = os.path.join(
            self.config.save_path,
            self.config.keyword if self.config.keyword else now_timestamp,
        )
        return permanent_save_path
        
    def _persist_results(self, tempdir: str):
        """Move results from temporary to permanent storage."""
        permanent_save_path = self._get_permanent_save_path()
        shutil.copytree(tempdir, permanent_save_path, dirs_exist_ok=True)
        print(f"Results successfully saved to {permanent_save_path}")

    # ========== Utility Methods ==========

    def _print_config(self):
        """Print training configuration."""
        print("Training with parameters:")
        print(f"  keyword: {self.config.keyword}")
        print(f"  df lengths: {[len(f) for f in self.config.metadata_frames.values()]}")
        print(f"  normalization: {self.config.normalization}")
        print(f"  y_transform: {self.config.y_transform}")
        print(f"  num_buckets: {self.config.num_buckets}")
        print(f"  num_epochs: {self.config.num_epochs}")
        print(f"  weighted: {self.config.class_weighted}")
        print(f"  regression: {self.config.regression}")
        print(f"  validation_metric: {self.config.validation_metric}")
        print(f"  checkpoint_path: {self.config.checkpoint_path}")
        print(f"  device: {self.config.device}")
        print(f"  additional model params: {self.config.additional_model_params}")

    def _validate_config(self):
        """Validate configuration (called in __init__)."""
        # Additional validation beyond __post_init__
        pass

    def _print_training_summary(self, time_elapsed: float):
        """Print training completion summary."""
        print(
            f"Training complete in {time_elapsed // 60:.0f}m "
            f"{time_elapsed % 60:.0f}s"
        )
