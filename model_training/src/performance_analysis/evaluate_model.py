import json
import os
import sys
from typing import Union
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
import torch
import seaborn as sns
from torchmetrics import JaccardIndex
import geopandas as gpd

from src.training.train_utils import soft_labels_to_hard_labels
from src.training.transforms import YTransform

from src.performance_analysis.reg_meter import RegMeter
from hedgementation_utils.metrics.seg_meter import SegMeter



import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Optional, Tuple

def _progress_bar(current: int, total: int, prefix: str = "", bar_width: int = 30) -> None:
    filled = int(bar_width * current / total) if total > 0 else 0
    bar = "=" * filled + "-" * (bar_width - filled)
    pct = 100 * current / total if total > 0 else 0
    sys.stdout.write(f"\r{prefix}[{bar}] {pct:5.1f}% ({current}/{total} batches)")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()
        
def get_model_predictions(model, dataloader, y_transform, device=None, regression = False):
    """
    Get predictions and labels for a model on a dataloader.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader to iterate over
        y_transform: Label transformation type
        device: Device to run on
        
    Returns:
        tuple: (all_preds, all_labels) as torch tensors
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    model = model.to(device)
    
    all_preds = []
    all_labels = []
    total = len(dataloader)

    skipped = 0
    with torch.no_grad():
        for batch_idx, item in enumerate(dataloader):
            _progress_bar(batch_idx + 1, total, prefix="  Evaluating ")
            try:
                input_tuple, labels, *_ = item
                inputs, dates = input_tuple[0], input_tuple[1]
                inputs = inputs.to(device)
                dates = dates.to(device)
                labels = labels.to(device)

                outputs = model(inputs, batch_positions=dates)

                if regression:
                    preds = outputs 
                else:
                    preds = torch.argmax(outputs, dim=1)
                    if y_transform == YTransform.SOFT_TARGETS:
                        labels = soft_labels_to_hard_labels(labels)
                
                all_preds.append(preds)
                all_labels.append(labels)
            except RuntimeError as e:
                print(f"\n  WARNING: Skipping batch {batch_idx} due to RuntimeError: {e}")
                skipped += 1
                continue

    if skipped > 0:
        print(f"\n  WARNING: Skipped {skipped}/{total} batches due to RuntimeError.")

    all_preds = torch.cat(all_preds).cpu()
    all_labels = torch.cat(all_labels).cpu()

    return all_preds, all_labels

def calculate_all_metrics(labels: torch.Tensor,
                          preds: torch.Tensor,
                          num_buckets=1,
                          regression: bool = False,
                          return_as_tuple=False):

    if regression:
        meter = RegMeter()
        meter.update(preds, labels)
        metrics = meter.ask_metrics()

        if return_as_tuple:
            return (metrics["mae"], metrics["mse"], metrics["rmse"], metrics["r2"])
        return metrics
    
    else:
        seg_meter = SegMeter(num_classes=num_buckets+1)
        seg_meter.update_batch(preds, labels)    
        metrics = seg_meter.ask_metrics()
        metrics["precision"] = metrics.pop("pre")
        metrics["recall"] = metrics.pop("acc")
    
        if return_as_tuple:
            return (metrics["iou"], metrics["precision"], metrics["recall"], metrics["f1"])
        return metrics


    
def get_model_performance_on_dataset(model: torch.nn.Module,
                                     dataloader: gpd.GeoDataFrame,
                                     device: str="cuda",
                                     y_transform: YTransform = YTransform.UNIFORM,
                                     num_buckets=1,
                                     regression: bool = False,):
    preds, labels = get_model_predictions(
        model,
        dataloader,
        y_transform,
        device,
        regression=regression
    )

    stats = calculate_all_metrics(labels, preds, num_buckets=num_buckets, regression=regression)
    return stats, preds, labels

def plot_prediction_overlay(
    ax: plt.Axes,
    X: torch.Tensor,
    y_true: torch.Tensor,
    y_pred_probs: torch.Tensor,
    title: str = "",
    channel_idx: Tuple[int, int] = (1, 1)
) -> None:
    """
    Plot a single prediction overlay on a matplotlib axis.
    
    Args:
        ax: Matplotlib axis to plot on
        X: Input tensor of shape (time, channels, height, width)
        y_true: Ground truth labels of shape (height, width)
        y_pred_probs: Predicted probabilities of shape (num_classes, height, width)
        title: Title for the subplot
        channel_idx: Tuple of (time_idx, channel_idx) for background image
    """
    # Display background image (grayscale)
    time_idx, chan_idx = channel_idx
    ax.imshow(X[time_idx, chan_idx, :, :], cmap='gray', interpolation='nearest')
    
    # Compute predictions and overlaps
    y_pred_viz = y_pred_probs.argmax(dim=0)
    overlap_viz = (y_true == y_pred_viz) & (y_true > 0)
    
    # Create RGBA overlays
    y_true_rgba = np.zeros((*y_true.shape, 4))
    y_true_rgba[y_true > 0] = [0, 0, 1, 1]  # Blue for ground truth
    
    y_pred_rgba = np.zeros((*y_pred_viz.shape, 4))
    y_pred_rgba[y_pred_viz > 0] = [0.7, 0, 0, 1]  # Red for predictions
    
    overlap_viz_rgba = np.zeros((*overlap_viz.shape, 4))
    overlap_viz_rgba[overlap_viz.cpu().numpy()] = [0.8, 0, 0.8, 1]  # Purple for overlap
    
    # Layer the overlays
    ax.imshow(y_true_rgba, interpolation='nearest')
    ax.imshow(y_pred_rgba, interpolation='nearest')
    ax.imshow(overlap_viz_rgba, interpolation='nearest')
    
    ax.set_title(title)
    ax.axis('off')


def plot_rgb_image(
    ax: plt.Axes,
    X: torch.Tensor,
    title: str = "",
    time_idx: int = 0,
    rgb_channels: Tuple[int, int, int] = (2, 1, 0)
) -> None:
    """
    Plot an RGB image from a tensor on a matplotlib axis.
    
    Args:
        ax: Matplotlib axis to plot on
        X: Input tensor of shape (time, channels, height, width)
        title: Title for the subplot
        time_idx: Time step to visualize
        rgb_channels: Tuple of (R, G, B) channel indices
    """
    X_np = np.array(X.cpu())
    
    # Extract and normalize RGB channels
    R = np.squeeze(X_np[time_idx, rgb_channels[0], :, :])
    R = R / np.max(R) if np.max(R) > 0 else R
    
    G = np.squeeze(X_np[time_idx, rgb_channels[1], :, :])
    G = G / np.max(G) if np.max(G) > 0 else G
    
    B = np.squeeze(X_np[time_idx, rgb_channels[2], :, :])
    B = B / np.max(B) if np.max(B) > 0 else B
    
    img = np.stack([R, G, B], axis=-1)
    
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')


def predict_single_sample(
    model: torch.nn.Module,
    X: torch.Tensor,
    dates: torch.Tensor,
    device: Union[torch.device, str, int, None]
) -> torch.Tensor:
    """
    Get model predictions for a single sample.
    
    Args:
        model: The model to use for prediction
        X: Input tensor of shape (time, channels, height, width)
        dates: Date tensor
        device: Device to run prediction on
    
    Returns:
        Prediction probabilities of shape (num_classes, height, width)
    """
    X_batch = X.unsqueeze(0).to(device)
    dates_batch = dates.unsqueeze(0).to(device)
    
    outputs = model(X_batch, batch_positions=dates_batch)
    y_pred_logits = outputs.cpu().squeeze(0)
    y_pred_probs = torch.softmax(y_pred_logits, dim=0)
    
    return y_pred_probs


def prepare_ground_truth(
    batch_y: torch.Tensor,
    sample_idx: int,
    y_transform: 'YTransform'
) -> torch.Tensor:
    """
    Extract and prepare ground truth label for a single sample.
    
    Args:
        batch_y: Batch of ground truth labels
        sample_idx: Index of sample to extract
        y_transform: Transform type applied to labels
    
    Returns:
        Ground truth tensor of shape (height, width)
    """
    if y_transform != YTransform.SOFT_TARGETS:
        return batch_y[sample_idx, :, :].squeeze()
    else:
        return soft_labels_to_hard_labels(batch_y[sample_idx, :, :, :].unsqueeze(0))


def visualize_model_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Union[torch.device, str, int, None],
    y_transform: 'YTransform',
    save_path: str,
    num_samples: int = 9,
    title: str = "Hedgerows Prediction vs Ground Truth",
    show_without_preds: bool = False
) -> None:
    """
    Visualize model predictions vs ground truth for multiple samples.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the data
        device: Device to run model on
        y_transform: Transform applied to labels
        save_path: Path to save visualizations
        num_samples: Number of samples to visualize (default: 9)
        title: Title for the prediction plot
        show_without_preds: Whether to also show underlying RGB images
    """
    model.eval()
    model.to(device)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(title, fontsize=16)
    
    with torch.no_grad():
        # Get batch
        input_tuple, batch_y, *_ = next(iter(dataloader))
        batch_X, batch_dates = input_tuple[0], input_tuple[1]
        
        # Plot predictions
        for i in range(min(num_samples, batch_X.shape[0])):
            X = batch_X[i, :, :, :, :]
            dates = batch_dates[i]
            y_true = prepare_ground_truth(batch_y, i, y_transform)
            
            # Get predictions
            y_pred_probs = predict_single_sample(model, X, dates, device)
            
            # Plot on grid
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            plot_prediction_overlay(
                ax=ax,
                X=X,
                y_true=y_true,
                y_pred_probs=y_pred_probs,
                title=f"Item {i+1}"
            )
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/imgs/visualized_predictions.png")
        plt.show()
        
        # Optional: show underlying RGB images
        if show_without_preds:
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            fig.suptitle("Underlying RGB img", fontsize=16)
            
            for i in range(min(num_samples, batch_X.shape[0])):
                X = batch_X[i, :, :, :, :].squeeze()
                
                row = i // 3
                col = i % 3
                ax = axes[row, col]
                
                plot_rgb_image(
                    ax=ax,
                    X=X,
                    title=f"Item {i+1}"
                )
            
            plt.tight_layout()
            plt.savefig(f"{save_path}/imgs/visualized_underlying.png")
            plt.show()

def save_sample_predictions(model: torch.nn.Module,
                           dataloader: torch.utils.data.DataLoader,
                           save_path: str,
                           file_prefix: str,
                           device: torch.device):
    with torch.no_grad():
        item = next(iter(dataloader))
        input_tuple, labels, *_ = item
        inputs, dates = input_tuple[0], input_tuple[1]
        inputs = inputs.to(device)
        dates = dates.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs, batch_positions=dates)

    npy_dir = os.path.join(save_path, "npys")
    for name, tensor in [
        ("outputs", outputs),
        ("inputs", inputs),
        ("labels", labels),
        ("dates", dates),
    ]:
        path = os.path.join(npy_dir, f"{file_prefix}_{name}_sample.npy")
        with open(path, "wb+") as f:
            np.save(f, tensor.cpu().numpy())

def save_confusion_matrix(all_preds: torch.Tensor,
                          all_labels: torch.Tensor,
                          num_buckets: int,
                          save_path: str,
                          file_prefix: str,
                          split: str):
    all_preds_np = all_preds.numpy().astype(np.int64).flatten()
    all_labels_np = all_labels.numpy().astype(np.int64).flatten()
    
    cm = confusion_matrix(
        all_labels_np, all_preds_np, 
        labels=[i for i in range(num_buckets + 1)]
    )

    plt.clf()
    sns.set_theme(font_scale=2.8)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2%", cmap="Blues", cbar=False,
        xticklabels=["background", "hedgerow"], 
        yticklabels=["bkcgrnd", "hdgrw"]
    )
    sns.set_theme(font_scale=1.2)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f"{save_path}/imgs/{file_prefix}_{split}_conf_matrix.png")
    plt.clf()


def evaluate_model(model, 
                   dataloaders: torch.utils.data.DataLoader, 
                   file_prefix: str, 
                   save_path: str, 
                   num_buckets: int, 
                   y_transform: YTransform,
                   regression: bool = False, 
                   save_extra_info: bool=False, 
                   device=None, 
                   return_preds: bool=False):
    """
    Evaluate model classification performance across multiple dataloaders.
    """
    print("calculating model evaluations")
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dict = {}

    for k in dataloaders:
        dataloader = dataloaders[k]
        all_preds, all_labels = get_model_predictions(model, dataloader, y_transform, device, regression=regression)
        
        if save_extra_info:
            save_sample_predictions(
                model,
                dataloader,
                save_path,
                file_prefix,
                device

            )
        
        if regression:
            mae, mse, rmse, r2 = calculate_all_metrics(all_labels, all_preds, regression=True, return_as_tuple=True)
            print(f"{k} MAE:  {mae:.4f}")
            print(f"{k} MSE:  {mse:.4f}")
            print(f"{k} RMSE: {rmse:.4f}")
            print(f"{k} R²:   {r2:.4f}")
 
            data_dict[f"mae_{k}"]  = [float(mae)]
            data_dict[f"mse_{k}"]  = [float(mse)]
            data_dict[f"rmse_{k}"] = [float(rmse)]
            data_dict[f"r2_{k}"]   = [float(r2)]

        else:
            if save_extra_info:
                save_confusion_matrix(
                    all_preds,
                    all_labels,
                    num_buckets,
                    save_path,
                    file_prefix,
                    k
                )
            
            iou, precision, recall, f1 = calculate_all_metrics(all_labels, 
                                                            all_preds, 
                                                            num_buckets=num_buckets,
                                                            return_as_tuple=True)

            print(f"{k} Precision: {precision:.4f}")
            print(f"{k} Recall: {recall:.4f}")
            print(f"{k} F1-score: {f1:.4f}")
            print(f"{k} IoU: {iou:.4f}")

            data_dict[f"precision_{k}"] = [precision]
            data_dict[f"recall_{k}"] = [recall]
            data_dict[f"f1_{k}"] = [f1]
            data_dict[f"iou_{k}"] = [float(iou)]

    # Save metrics to JSON
    with open(f"{save_path}/{file_prefix}_metrics.json", "w+") as outfile:
        json.dump(data_dict, outfile, indent=4)
    
    if return_preds:
        all_preds_np = all_preds.numpy().astype(np.int64).flatten()
        all_labels_np = all_labels.numpy().astype(np.int64).flatten()
        return all_preds_np, all_labels_np
    return data_dict


def evaluate_per_patch_performance(model, 
                                   metadata: gpd.GeoDataFrame, 
                                   dataloader, 
                                   save_path,
                                   regression=False, 
                                   num_buckets=1,
                                   save_data=True,
                                   file_name=None,
                                   device=None):

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    model = model.to(device)

    patch_metrics = []
    preds = []


    with torch.no_grad():
        for item in dataloader:
            input_tuple, y, *_ = item
            X, dates = input_tuple[0], input_tuple[1]
            X = X.to(device)
            dates = dates.to(device)
            y = torch.squeeze(y.to(device))

            y_pred = model(X, dates)

            if not regression:
                y_pred = torch.squeeze(torch.argmax(y_pred, dim=1))

            for i in range(y.shape[0]):
                current_patch_labels = y[i]
                current_patch_preds = y_pred[i]
                
                metrics_dict = calculate_all_metrics(current_patch_labels, 
                                                     current_patch_preds,
                                                     num_buckets=num_buckets,
                                                     regression=regression)
                patch_metrics.append(metrics_dict)
                preds.append(current_patch_preds.cpu())

    if regression:
        for key in ["mae", "mse", "rmse", "r2"]:
            metadata[key] = [m[key] for m in patch_metrics]
    
    else:
        for key in ["iou", "precision", "recall", "f1"]:
            metadata[key] = [m[key] for m in patch_metrics]

    if save_data:
        name_to_use = file_name if file_name else "metadata_with_performance"
        metadata.to_file(f"{save_path}/{name_to_use}.geojson")
    return metadata