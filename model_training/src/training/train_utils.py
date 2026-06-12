import copy
import os
import random
from typing import Any
import numpy as np
import torch
from dotenv import load_dotenv
load_dotenv()
from enum import Enum
from backbones.convlstm import ConvLSTM
from backbones.unet3d import UNet3D
from backbones.utae import UTAE

from torch.optim import lr_scheduler

NUM_CHANNELS = int(os.environ["NUM_CHANNELS"])

class Backbones(Enum):
    ConvLSTM = "ConvLSTM"
    UNet3D = "UNet3D"
    UTAE = "UTAE"

class LRSchedule(Enum):
    COSINE_WARM_RESTARTS = "CosineAnnealingWarmRestarts"
    COSINE_ANNEALING = "CosineAnnealingLR"
    CONSTANT = "ConstantLR"
    REDUCE_ON_PLATEAU = "ReduceLROnPlateau"
    ONE_CYCLE = "OneCycle"
    STEP_LR = "StepLR"

    @classmethod
    def default_args(cls,type):
        if type == cls.COSINE_WARM_RESTARTS:
            return {
                "T_0": 20,
                "T_mult": 2
            }
        elif type == cls.COSINE_ANNEALING:
            return {
                "T_max": 90
            }
        elif type == cls.CONSTANT:
            return {}
        elif type == cls.REDUCE_ON_PLATEAU:
            return {
                "patience": 10
            }
        elif type == cls.ONE_CYCLE:
            return {
                "max_lr": 5e-3
            }
        elif type == cls.STEP_LR:
            return {
                "step_size": 10
            }
        else:
            raise ValueError(f"{type} is not a valid scheduler type.")

class LRSchedulerFactory():
    @classmethod
    def construct_lr_scheduler(cls,
                                optimizer: torch.optim.Optimizer,
                                scheduler_type: LRSchedule,
                                scheduler_args: dict[str,Any]):
        base_args = LRSchedule.default_args(scheduler_type)
        scheduler_args = base_args | scheduler_args
        match (scheduler_type):
            case LRSchedule.COSINE_WARM_RESTARTS:
                return lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer=optimizer,
                    **scheduler_args
                    )
            case LRSchedule.COSINE_ANNEALING:
                return lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    **scheduler_args
                    )
            case LRSchedule.CONSTANT:
                return lr_scheduler.ConstantLR(
                    optimizer=optimizer,
                    **scheduler_args
                    )
            case LRSchedule.REDUCE_ON_PLATEAU:
                return lr_scheduler.ReduceLROnPlateau(
                    optimizer=optimizer,
                    **scheduler_args
                    )
            case LRSchedule.ONE_CYCLE:
                return lr_scheduler.OneCycleLR(
                    optimizer=optimizer,
                    **scheduler_args
                    )
            case LRSchedule.STEP_LR:
                return lr_scheduler.StepLR(
                    optimizer=optimizer,
                    **scheduler_args
                    )
            case _:
                raise ValueError(f"{scheduler_type} is not a valid scheduler type.")
        
        
        

def soft_labels_to_hard_labels(labels):
    return torch.argmax(labels, dim=-3).squeeze()
        

class ModelFactory():
    @classmethod
    def construct_model(cls, 
                        backbone, 
                        num_buckets, 
                        num_channels=NUM_CHANNELS,
                        additional_model_params=None):
        print(backbone)
        if backbone == Backbones.UNet3D:
            if additional_model_params:
                return UNet3D(num_channels, n_classes=num_buckets+1, pad_value=0, **additional_model_params)
            return UNet3D(num_channels, n_classes=num_buckets+1, pad_value=0)
        elif backbone == Backbones.UTAE:
            if additional_model_params:
                return UTAE(num_channels, pad_value=0, **additional_model_params)
            return UTAE(num_channels, pad_value=0)
        else:
            raise NotImplementedError(f"The model type {backbone} is either invalid or not supported")
        

def calculate_excluded_loss(labels_continuous, 
                      labels, 
                      outputs, 
                      intervals,
                      criterion
                      ):
    device = labels_continuous.device
    intervals_tensor = torch.tensor(intervals, device=device)

    continuous_expanded = labels_continuous.unsqueeze(1).unsqueeze(-1)
    intervals_expanded = intervals_tensor.unsqueeze(0)

    in_intervals = ((continuous_expanded >= intervals_expanded[...,0]) & 
                    (continuous_expanded <= intervals_expanded[...,1]))
    mask = in_intervals.any(dim=-1)
    
    pixelwise_loss = criterion(input=outputs, target=labels)

    masked_loss = pixelwise_loss[mask.squeeze(1)]

    if masked_loss.numel() > 0:
        loss = masked_loss.mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss,mask

def calculate_class_weights(dataloader, num_classes, device):
    class_counts = torch.zeros(num_classes)
    total_pixels = 0

    for item in dataloader:
        _, labels, *_ = item
        if len(labels.shape) == 4:  # one-hot
            labels = labels.argmax(dim=1)
        class_counts += torch.bincount(labels.flatten(), minlength=num_classes)
        total_pixels += labels.numel()


    class_weights = total_pixels / (num_classes * class_counts)
    print(f"class_weights: {class_weights}")
    return class_weights.to(device)


def collate_fn(batch):
    # Check if masks are present
    has_masks = len(batch[0]) == 3

    if has_masks:
        inputs, targets, masks = zip(*batch)
    else:
        inputs, targets = zip(*batch)

    # Check if X_cloud is present (inputs are 3-tuples)
    has_cloud = len(inputs[0]) == 3

    if has_cloud:
        X_list, dates_list, cloud_list = zip(*inputs)
    else:
        X_list, dates_list = zip(*inputs)

    max_len = max(x.shape[0] for x in X_list)

    X_padded = []
    dates_padded = []
    cloud_padded = []

    for i, (X, dates) in enumerate(zip(X_list, dates_list)):
        seq_len = X.shape[0]
        dates_len = dates.shape[0]
        if seq_len < max_len:
            pad_size = max_len - seq_len
            X_pad = torch.full((pad_size, *X.shape[1:]), 0, dtype=X.dtype)
            X = torch.cat([X, X_pad], dim=0)
        if dates_len < max_len:
            pad_size = max_len - dates_len
            dates_pad = torch.full((pad_size,), -1, dtype=dates.dtype)
            dates = torch.cat([dates, dates_pad], dim=0)

        X_padded.append(X)
        dates_padded.append(dates)

        if has_cloud:
            X_cloud = cloud_list[i]
            cloud_len = X_cloud.shape[0]
            if cloud_len < max_len:
                pad_size = max_len - cloud_len
                cloud_pad = torch.full((pad_size, *X_cloud.shape[1:]), 0, dtype=X_cloud.dtype)
                X_cloud = torch.cat([X_cloud, cloud_pad], dim=0)
            cloud_padded.append(X_cloud)

    X_batch = torch.stack(X_padded)
    dates_batch = torch.stack(dates_padded)
    y_batch = torch.stack(targets)

    inputs_batch = (X_batch, dates_batch, torch.stack(cloud_padded)) if has_cloud else (X_batch, dates_batch)

    if has_masks:
        mask_keys = masks[0].keys()
        masks_batch = {k: torch.stack([m[k] for m in masks]) for k in mask_keys}
        return inputs_batch, y_batch, masks_batch
    else:
        return inputs_batch, y_batch
    

def find_layers_to_load(model_state_dict, pretrained_state_dict):
    matching_keys = []
    non_matching_keys = []
    absent_keys = []
    for k in model_state_dict:
        if k not in pretrained_state_dict:
            absent_keys.append(k)
        elif k in pretrained_state_dict and model_state_dict[k].shape != pretrained_state_dict[k].shape:
            non_matching_keys.append(k)
        else:
            matching_keys.append(k)
    return matching_keys, non_matching_keys, absent_keys

def adapt_state_dict(model: torch.nn.Module, 
                                  state_dict: dict[str,Any]):

    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    matching_keys, *_ = find_layers_to_load(model.state_dict(), state_dict)

    if matching_keys == model.state_dict().keys():
        return state_dict
    sub_state_dict = {k:i for (k,i) in state_dict.items() if k in matching_keys}

    model_state_dict = copy.deepcopy(model.state_dict())
    model_state_dict.update(sub_state_dict)
    
    return model_state_dict

    
