
import torch

from src.performance_analysis.utils import get_model_prediction_from_model_path
from hedgementation_utils.visualization.visualization_utils import visualize_prediction


def get_and_display_prediction(
        model_dir: str,
        ax,
        id: int,
        prediction_cmaps=None
    ):
    pred,label = get_model_prediction_from_model_path(model_dir, [id])
    pred,label = torch.squeeze(pred), torch.squeeze(label)
    visualize_prediction(
        ax=ax,
        preds=pred,
        labels=label,
        patch_id=id,
        show_satellite_img_beneath=True,
        prediction_cmaps=prediction_cmaps
    )