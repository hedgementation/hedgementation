"""
Regression metrics accumulator for model training.

Mirrors the SegMeter interface so that Trainer can use either interchangeably.
Accumulates sufficient statistics (sums) in a single pass — no need to store
raw predictions — and computes MAE, MSE, RMSE, R² at epoch end.
"""

import torch


_HIGHER_IS_BETTER = {"r2", "iou", "f1", "pre", "acc", "all_acc", "freq_iou"}

REGRESSION_METRICS = {"mae", "mse", "rmse", "r2", "loss"}

CLASSIFICATION_METRICS = {"all_acc", "acc", "pre", "f1", "iou", "freq_iou", "loss"}


def higher_is_better(metric: str) -> bool:
    return metric.lower() in _HIGHER_IS_BETTER


class RegMeter:
    """
    Accumulates regression metrics over an epoch without storing raw tensors.

    Tracks the following sufficient statistics per update:
        - n              : total number of predictions
        - sum_abs_err    : sum of |pred - label|
        - sum_sq_err     : sum of (pred - label)²
        - sum_labels     : sum of label values  (needed for R²)
        - sum_sq_labels  : sum of label² values (needed for R²)

    Usage mirrors SegMeter:
        meter = RegMeter()
        for batch in dataloader:
            meter.update(preds, labels)
        metrics = meter.ask_metrics()   # {"mae": ..., "mse": ..., ...}
    """

    def __init__(self):
        self.n: int = 0
        self.sum_abs_err: float = 0.0
        self.sum_sq_err: float = 0.0
        self.sum_labels: float = 0.0
        self.sum_sq_labels: float = 0.0

    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """
        Accumulate statistics from a single batch or sample.

        Args:
            preds:  Predicted values, any shape — will be flattened.
            labels: Ground-truth values, same shape as preds.
        """
        preds = preds.detach().float().view(-1)
        labels = labels.detach().float().view(-1)

        diff = preds - labels
        self.n += labels.numel()
        self.sum_abs_err += diff.abs().sum().item()
        self.sum_sq_err += (diff ** 2).sum().item()
        self.sum_labels += labels.sum().item()
        self.sum_sq_labels += (labels ** 2).sum().item()

    def update_batch(self, preds: torch.Tensor, labels: torch.Tensor):
        return self.update(preds=preds, labels=labels)

    def score(self) -> dict:
        """
        Compute all regression metrics from accumulated statistics.

        Returns:
            dict with keys: mae, mse, rmse, r2
        """
        if self.n == 0:
            return {"mae": float("nan"), "mse": float("nan"),
                    "rmse": float("nan"), "r2": float("nan")}

        mae = self.sum_abs_err / self.n
        mse = self.sum_sq_err / self.n
        rmse = mse ** 0.5

        # R² = 1 - SS_res / SS_tot
        # SS_tot = sum((y - y_mean)²) = sum_sq_labels - n * y_mean²
        y_mean = self.sum_labels / self.n
        ss_tot = self.sum_sq_labels - self.n * (y_mean ** 2)
        r2 = 1.0 - (self.sum_sq_err / ss_tot) if ss_tot > 0 else float("nan")

        return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    def ask_metrics(self) -> dict:
        """
        Return all metrics as a plain dict of Python floats.

        Mirrors SegMeter.ask_metrics() so callers are interchangeable.
        """
        return self.score()