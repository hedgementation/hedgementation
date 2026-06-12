from enum import Enum
import random
import torch
import pandas as pd 
import numpy as np
import os

from torchvision.transforms import v2
from dotenv import load_dotenv
load_dotenv()
from torchvision.transforms.v2 import functional as F

import torchvision

from src.training.datastore_utils import DataStore, load_norm_vals, load_quantile_data, TRAIN_DATA_PATH, NORM_VALS_PATH, QUANTILE_DATA_PATTERN


NUM_CHANNELS = 10

def calculate_quantiles(loaded, num_buckets):
    return torch.from_numpy(np.quantile(loaded[loaded != 0], [i / num_buckets for i in range(num_buckets)]))

def PastisZScoreNormalize(X,y):
    X = X.float()

    norm_vals = DataStore.get_data("norm_vals",load_norm_vals, {
        "file_path": NORM_VALS_PATH,
        "num_channels": NUM_CHANNELS
    })
    
    mean_tensor = norm_vals["mean"].unsqueeze(-1).unsqueeze(-1).float()
    std_tensor = norm_vals["std"].unsqueeze(-1).unsqueeze(-1).float()
    
    mean_tensor = mean_tensor.to(X.device)
    std_tensor = std_tensor.to(X.device)
    
    X -= mean_tensor
    X /= std_tensor

    

    return X,y

def PastisMinMaxScale(X,y):
    X = X.float()

    norm_vals = DataStore.get_data("norm_vals",load_norm_vals, {
        "file_path": NORM_VALS_PATH,
        "num_channels": NUM_CHANNELS
    })
    mins = norm_vals["min"].unsqueeze(-1).unsqueeze(-1)
    maxs = norm_vals["max"].unsqueeze(-1).unsqueeze(-1)
    X = (X-mins) / (maxs-mins + 1e-8)

    return X,y


def bucketize(y,boundaries,one_hot=False):
    retval = torch.bucketize(y,boundaries)
    if one_hot:
        return torch.nn.functional.one_hot(retval).view(-1,128,128)
    return retval
    
def get_uniform_bucket_classification(y,num_buckets, one_hot=False):
    boundaries = torch.linspace(0,1,num_buckets+1)
    return bucketize(y, boundaries, one_hot=one_hot)

def get_quantile_bucket_classification(y,num_buckets,one_hot=False):
    loaded = DataStore.get_data("quantile_data", load_quantile_data, {"path": TRAIN_DATA_PATH, "pattern": QUANTILE_DATA_PATTERN})
    boundaries = DataStore.get_data(f"quantiles_{num_buckets}", calculate_quantiles, {"loaded": loaded, "num_buckets": num_buckets})
    return bucketize(y,boundaries,one_hot=one_hot)

def get_soft_targets(y, num_buckets, temperature=1.0, min_class_1_prob=0.51, max_class_1_prob=0.99):
    if num_buckets != 1:
        raise NotImplementedError("Soft targets for non-binary tasks are currently not supported")
    y = torch.clamp(y, 0, 1)
    
    prob_class_1 = torch.where(
        y == 0,
        torch.tensor(0.01),  
        min_class_1_prob + y * (max_class_1_prob - min_class_1_prob)  
    )
    
    prob_class_0 = 1 - prob_class_1
    
    soft_y = torch.stack([prob_class_0, prob_class_1], dim=-3)
    return soft_y

class Normalization(Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"
    NONE = "none"

class YTransform(Enum):
    QUANTILE = "quantile"
    UNIFORM = "uniform"
    SOFT_TARGETS = "soft_targets"
    NONE = "none"

class Random90DegreeRotation(torch.nn.Module):
    def __init__(self, p=1.0):
        super().__init__()
        self.p = p
        self.angles = [0, 90, 180, 270]
    
    def forward(self, img, mask):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            return F.rotate(img, angle, interpolation=v2.InterpolationMode.NEAREST), F.rotate(mask, angle, interpolation=v2.InterpolationMode.NEAREST)
        return img, mask

AUGMENTATION_TRANSFORMS = {
    "hflip": v2.RandomHorizontalFlip(0.5),
    "vflip": v2.RandomVerticalFlip(0.5),
    "rotation": Random90DegreeRotation()
}
    
RANDOM_SEED = int(os.environ["RANDOM_SEED"])

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def transform_y(y, transformation, num_buckets):
    if transformation == YTransform.QUANTILE:
        y = get_quantile_bucket_classification(y,num_buckets=num_buckets)
    elif transformation == YTransform.UNIFORM:
        y = get_uniform_bucket_classification(y,num_buckets=num_buckets)
    elif transformation == YTransform.SOFT_TARGETS:
        y = get_soft_targets(y,num_buckets=num_buckets)
    return y


def compound_transform(X,
                       y,
                       masks=None,
                       X_cloud=None,
                       normalization=None,
                       y_transform=None,
                       num_buckets=None,
                       augmentation=None,
                       y_threshold=None):
    """
    masks: dict mapping string keys to tensors (e.g. {"ag": bool, "ds": bool, "y_id": int64}),
           or None if no masks are needed. Spatial augmentations are applied to every
           tensor; original dtypes are preserved on return.
    X_cloud: optional (T, C_cloud, H, W) array aligned with X. Receives the same spatial
             augmentation as X but is NOT normalized.
    """

    X = torch.from_numpy(X.copy()).float()
    y = torch.from_numpy(y.copy())
    if X_cloud is not None:
        X_cloud = torch.from_numpy(X_cloud.copy()).float()

    if y_threshold:
        y[y < y_threshold] = 0

    if augmentation:
        # Concat X_cloud onto X channel-wise so a single augmentation call gives
        # both tensors identical spatial transforms.
        if X_cloud is not None:
            n_x_channels = X.shape[1]
            aug_input = torch.cat([X, X_cloud], dim=1)
        else:
            aug_input = X

        if masks:
            # Stack y and all masks into a single tv_tensors.Mask so every tensor
            # receives the same spatial augmentation in one call.
            mask_keys = list(masks.keys())
            mask_dtypes = {k: masks[k].dtype for k in mask_keys}
            mask_stack = [y] + [masks[k].float() for k in mask_keys]
            combined = torchvision.tv_tensors.Mask(torch.stack(mask_stack))
            aug_input, combined = augmentation(aug_input, combined)
            y = combined[0].contiguous()
            masks = {k: combined[i + 1].to(mask_dtypes[k]) for i, k in enumerate(mask_keys)}
        else:
            aug_input, y = augmentation(aug_input, torchvision.tv_tensors.Mask(y))
            y = y.contiguous()

        if X_cloud is not None:
            X = aug_input[:, :n_x_channels].contiguous()
            X_cloud = aug_input[:, n_x_channels:].contiguous()
        else:
            X = aug_input

    if normalization == Normalization.MINMAX:
        X, y = PastisMinMaxScale(X, y)
    elif normalization == Normalization.ZSCORE:
        X, y = PastisZScoreNormalize(X, y)

    y = transform_y(y, y_transform, num_buckets)

    out = (X,)
    if X_cloud is not None:
        out = out + (X_cloud,)
    out = out + (y,)
    if masks:
        out = out + (masks,)
    return out
