"""
Metrics tracking and visualization for model training.

This module provides MetricsTracker, a class that tracks training and validation
metrics throughout the training process and generates visualizations.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

from src.performance_analysis.reg_meter import higher_is_better


class MetricsTracker:
    """
    Tracks and manages training/validation metrics.

    This class provides a clean interface for recording metrics during training
    and generating visualizations (loss and secondary-metric graphs).

    Attributes:
        validation_metric: Name of the metric used for best-model selection
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        train_metrics: List of training secondary-metric scores per epoch
        val_metrics: List of validation secondary-metric scores per epoch
        best_metric: Best validation secondary-metric value achieved
        best_epoch: Epoch where best metric was achieved
    """

    def __init__(self, validation_metric: str = "iou"):
        """Initialize the metrics tracker."""
        self.validation_metric = validation_metric
        self._higher_is_better = higher_is_better(validation_metric)

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_metrics: List[float] = []
        self.val_metrics: List[float] = []

        # Initialise best to worst possible value depending on direction
        self.best_metric: float = float("-inf") if self._higher_is_better else float("inf")
        self.best_epoch: int = -1

    def record_train_metrics(self, loss: float, metric: float):
        """
        Record training metrics for an epoch.

        Args:
            loss:   Training loss value
            metric: Training secondary-metric value (IoU, F1, MAE, …)
        """
        self.train_losses.append(loss)
        self.train_metrics.append(metric)

    def record_val_metrics(self, loss: float, metric: float) -> bool:
        """
        Record validation metrics for an epoch.

        Args:
            loss:   Validation loss value
            metric: Validation secondary-metric value

        Returns:
            bool: True if this is a new best model
        """
        self.val_losses.append(loss)
        self.val_metrics.append(metric)

        if self._higher_is_better:
            is_best = metric > self.best_metric
        else:
            is_best = metric < self.best_metric

        if is_best:
            self.best_metric = metric
            self.best_epoch = len(self.val_metrics) - 1

        return is_best

    def save_graphs(self, save_dir: str):
        """
        Generate and save loss and secondary-metric graphs.

        Args:
            save_dir: Base directory where 'imgs' subdirectory exists
        """
        imgs_path = os.path.join(save_dir, "imgs")
        self._save_graph(
            os.path.join(imgs_path, "loss_graph.png"),
            "Loss",
            self.val_losses,
            self.train_losses,
        )
        self._save_graph(
            os.path.join(imgs_path, f"{self.validation_metric}_graph.png"),
            self.validation_metric.upper(),
            self.val_metrics,
            self.train_metrics,
        )

    def save_csv(self, save_path: str):
        """
        Save metrics to CSV file.

        Args:
            save_path: Path where CSV file should be saved
        """
        df = pd.DataFrame(
            {
                "train_loss": self.train_losses,
                "val_loss": self.val_losses,
                f"train_{self.validation_metric}": self.train_metrics,
                f"val_{self.validation_metric}": self.val_metrics,
            }
        )
        df.to_csv(save_path)

    def _save_graph(
        self,
        path: str,
        metric: str,
        val_record: List[float],
        train_record: List[float],
    ):
        """
        Internal method to generate and save a single graph.

        Args:
            path: Path where graph should be saved
            metric: Name of the metric being plotted
            val_record: Validation metric values
            train_record: Training metric values
        """
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(len(val_record)), val_record, color="blue", label=f"Validation {metric}"
        )
        plt.plot(
            range(len(train_record)),
            train_record,
            color="orange",
            label=f"Train {metric}",
        )
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"Training and Validation {metric}")
        plt.savefig(path)
        plt.close()
