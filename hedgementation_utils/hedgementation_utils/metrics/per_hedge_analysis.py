from typing import Optional

import torch

from hedgementation_utils.io.io_manager import HedgementationIOManager
import os
from dotenv import load_dotenv

from hedgementation_utils.metrics.seg_meter import SegMeter

load_dotenv()

DATASET_ROOT = os.getenv("DATASET_ROOT","")

_default_manager: Optional[HedgementationIOManager] = None
_default_meter: Optional[SegMeter] = None

def get_io_manager() -> HedgementationIOManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = HedgementationIOManager(dataset_root=DATASET_ROOT)
    return _default_manager

def get_seg_meter():
    global _default_meter
    if _default_meter is None:
        _default_meter = SegMeter(num_classes=2)
    return _default_meter


def calc_mean_hedge_accuracy(
        per_hedge_acc: torch.Tensor
):
    return torch.mean(per_hedge_acc)

def calc_hedge_detection_rate(
        per_hedge_acc: torch.Tensor,
        threshold: float = 0.2
):
    return torch.sum(per_hedge_acc > threshold) / len(per_hedge_acc)

def calc_patch_per_hedge_accuracy(
    predictions: torch.tensor,
    target_raster: Optional[torch.tensor] = None,
    target_id: Optional[int] = None,
    seg_meter: Optional[SegMeter] = None,
    io_manager: Optional[HedgementationIOManager] = None
):
    seg_meter = seg_meter or get_seg_meter()

    if target_id is None and target_raster is None:
        raise ValueError("Please provide either the target raster itself, or a patch ID to retrieve it with.")
    if target_raster is None:
        io_manager = io_manager or get_io_manager()
        target_raster = io_manager.load_y_id(identifier=target_id)

    num_hedgerows = int(torch.max(target_raster).item())

    score_per_hedgerow = []

    for i in range(num_hedgerows + 1):
        seg_meter.hist = None
        current_raster = target_raster == i
        seg_meter.update(predictions, current_raster)
        score_per_hedgerow.append(seg_meter.ask_metrics(1)["acc"])

    return torch.tensor(score_per_hedgerow)

def calc_dataset_per_hedge_metrics(
        predictions_and_y_ids: list[tuple[torch.Tensor, torch.Tensor]],
        detection_threshold: float = 0.2,
        seg_meter: Optional[SegMeter] = None,
) -> dict:
    """
    Pool per-hedge accuracies across all patches and return aggregate metrics.

    Each hedgerow across every patch contributes equally to the final statistics
    (not averaged per-patch first), so a dataset with many small patches is not
    dominated by per-patch averages.

    Args:
        predictions_and_y_ids: Iterable of (prediction, y_id) tensor pairs where
            prediction is (H, W) predicted class labels and y_id is (H, W) hedge ID raster.
        detection_threshold: Accuracy threshold above which a hedge counts as detected.
        seg_meter: Optional shared SegMeter; a fresh one is created if not provided.

    Returns:
        dict with 'mean_hedge_accuracy' and 'hedge_detection_rate' pooled across all hedgerows.
        Returns an empty dict if no patches are provided.
    """
    seg_meter = seg_meter or SegMeter(num_classes=2)
    all_per_hedge_acc = []

    for predictions, y_id in predictions_and_y_ids:
        per_hedge_acc = calc_patch_per_hedge_accuracy(
            predictions=predictions,
            target_raster=y_id,
            seg_meter=seg_meter,
        )
        all_per_hedge_acc.append(per_hedge_acc)

    if not all_per_hedge_acc:
        return {}

    pooled = torch.cat(all_per_hedge_acc)
    return {
        "mean_hedge_accuracy": calc_mean_hedge_accuracy(pooled).item(),
        "hedge_detection_rate": calc_hedge_detection_rate(pooled, threshold=detection_threshold).item(),
    }


def calc_per_hedge_metrics(
        predictions: torch.tensor,
    target_raster: Optional[torch.tensor] = None,
    target_id: Optional[int] = None,
    seg_meter: Optional[SegMeter] = None,
    io_manager: Optional[HedgementationIOManager] = None,
    detection_threshold: float = 0.2
):
    per_hedge_acc = calc_patch_per_hedge_accuracy(
        predictions = predictions,
        target_id=target_id,
        target_raster=target_raster,
        seg_meter=(seg_meter or get_seg_meter()),
        io_manager=(io_manager or get_io_manager())
    )

    return {
        "mean_hedge_accuracy": calc_mean_hedge_accuracy(per_hedge_acc=per_hedge_acc),
        "hedge_detection_rate": calc_hedge_detection_rate(per_hedge_acc=per_hedge_acc, threshold=detection_threshold)
    }

