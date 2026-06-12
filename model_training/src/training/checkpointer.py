"""
Model checkpoint management for training.

This module provides Checkpointer, a class that manages saving and loading
model checkpoints during training.
"""

import os

import torch


class Checkpointer:
    """
    Manages model checkpointing during training.

    Handles saving best model, periodic checkpoints, and most recent model.

    Attributes:
        checkpoint_dir: Directory where checkpoints are saved
        save_interval: Epochs between periodic checkpoints
        best_model_path: Path to best model checkpoint
        most_recent_path: Path to most recent model checkpoint
        checkpoint_template: Template for periodic checkpoint paths
    """

    def __init__(self, checkpoint_dir: str, save_interval: int = 20):
        """
        Initialize checkpointer.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Epochs between periodic checkpoints (default: 20)
        """
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_model_path = os.path.join(checkpoint_dir, "best_model_params.pt")
        self.most_recent_path = os.path.join(
            checkpoint_dir, "most_recent_model_params.pt"
        )
        self.checkpoint_template = os.path.join(
            checkpoint_dir, "epoch_{}_model_params.pt"
        )

    def save_best(self, model: torch.nn.Module):
        """
        Save the best model checkpoint.

        Args:
            model: Model to save
        """
        torch.save(model.state_dict(), self.best_model_path)

    def save_periodic(self, model: torch.nn.Module, epoch: int, num_epochs: int):
        """
        Save periodic checkpoint if conditions are met.

        Checkpoints are saved at intervals determined by the dataset size
        and number of epochs.

        Args:
            model: Model to save
            epoch: Current epoch number
            num_epochs: Total number of epochs
        """
        interval = max(min(20, num_epochs // 10), 10)
        if epoch % interval == 0:
            torch.save(model.state_dict(), self.checkpoint_template.format(epoch))

    def save_most_recent(self, model: torch.nn.Module):
        """
        Save the most recent model state.

        Args:
            model: Model to save
        """
        torch.save(model.state_dict(), self.most_recent_path)

    def load_best(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Load the best model checkpoint.

        Args:
            model: Model to load checkpoint into

        Returns:
            torch.nn.Module: Model with loaded weights
        """
        model.load_state_dict(torch.load(self.best_model_path, weights_only=True))
        return model
