import datetime
from typing import Optional
from typing import Optional
import torch
import pathlib
import pandas as pd
import os
import numpy as np
import pickle
from dotenv import load_dotenv
from hedgementation_utils.io.io_manager import HedgementationIOManager

load_dotenv()

DATASET_ROOT = pathlib.Path(os.environ["DATASET_ROOT"])
RPG_MASK_SUBDIR = os.environ["RPG_MASK_SUBDIR"]



class HedgementationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 metadata_frame,
                 root_dir=DATASET_ROOT,
                 rpg_mask_subdir=RPG_MASK_SUBDIR,
                 constant_length=None,
                 use_memmap=False,
                 transform=None,
                 provide_agriculture_masks=False,
                 downsample_loss=False,
                 downsample_mask_path=None,
                 reference_date="2021-09-17",
                 cache_dir=None,
                 transform_vals = None,
                 target_size=128,
                 pad_value=0,
                 even_sample=True,
                 load_X_cloud=False,
                 load_y_id=False,
                 cloud_threshold=None,
                 cloud_band=0,
                 io_manager_kwargs=None, 
                 transfer=None):
        self.pad_value = pad_value
        self.metadata_frame = pd.read_csv(metadata_frame) if type(metadata_frame) == str else metadata_frame
        self.root_dir = root_dir
        self.transform = transform
        self.transform_vals = transform_vals
        self.constant_length = constant_length
        self.target_size = target_size
        self.even_sample = even_sample
        self.provide_agriculture_masks = provide_agriculture_masks
        self.downsample_loss = downsample_loss
        self.downsample_mask_path = downsample_mask_path
        self.load_X_cloud = load_X_cloud
        self.load_y_id = load_y_id
        self.cloud_threshold = cloud_threshold
        self.cloud_band = cloud_band
        self.transfer = transfer

        if rpg_mask_subdir is None and self.provide_agriculture_masks:
            rpg_mask_subdir = RPG_MASK_SUBDIR
        self.rpg_mask_subdir = rpg_mask_subdir

        self.cache_dir = cache_dir or os.path.join(root_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.use_memmap=use_memmap

        self.io_manager = HedgementationIOManager(
            dataset_root=str(root_dir),
            rpg_mask_dir=self.rpg_mask_subdir or RPG_MASK_SUBDIR,
            rpg_pattern="mask_{}",
            **(io_manager_kwargs or {}),
        )

        self.reference_date = datetime.datetime(*map(int, reference_date.split("-")))
        if self.use_memmap:
            self._build_memmap()

        if self.provide_agriculture_masks:
            self._load_masks()
        if self.downsample_loss:
            self._load_downsample_masks()

    def _get_cache_path(self, identifier):
        key = str(identifier)
        if self.load_X_cloud:
            key += f"_xcld_cb{self.cloud_band}"
            if self.cloud_threshold is not None:
                key += f"_ct{self.cloud_threshold}"
        return os.path.join(self.cache_dir, f"{key}.npz")

    def _load_raw_cached(self, identifier):
        cache_path = self._get_cache_path(identifier)
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path) as data:
                    X = data["X"].copy()
                    dates = torch.from_numpy(data["dates"].copy())
                    y = data["y"].copy()
                    if self.load_X_cloud:
                        return X, data["X_cloud"].copy(), dates, y
                    return X, dates, y
            except Exception:
                pass

        if not self.load_X_cloud:
            X, raw_dates = self.io_manager.load_X(identifier=identifier, return_dates=True)
        else:
            if self.cloud_threshold:
                X, X_cloud, raw_dates = self.io_manager.load_X_data_with_cloud_threshold(
                    identifier=identifier
                )
            else:
                X, X_cloud, raw_dates = self.io_manager.load_X_data_date_aligned(
                    identifier=identifier
                )
        y = self.io_manager.load_y(identifier=identifier)
        dates = self._dates_to_offsets(raw_dates)

        save_dict = {"X": X, "dates": dates.numpy(), "y": y}
        if self.load_X_cloud:
            save_dict["X_cloud"] = X_cloud
        np.savez(cache_path, **save_dict)

        if self.load_X_cloud:
            return X, X_cloud, dates, y
        return X, dates, y

    def _dates_to_offsets(self, dates):
        """Convert a pandas Series of 'YYYY-MM-DD' strings to a tensor of
        day offsets from ``self.reference_date``."""
        offsets = (pd.to_datetime(dates) - self.reference_date).dt.days
        return torch.from_numpy(offsets.to_numpy(dtype=np.int64).copy())
    
    def _load_masks(self):
        """Pre-load all agriculture masks into memory."""
        print("Loading agriculture masks...")
        self.agriculture_masks = {}

        for idx in range(len(self.metadata_frame)):
            current_id = self.metadata_frame["ID_PATCH"].iloc[idx]
            try:
                mask = self.io_manager.load_rpg_mask(identifier=current_id)
                self.agriculture_masks[current_id] = torch.from_numpy(mask.astype(bool))
            except FileNotFoundError:
                print(f"Warning: Mask not found for ID_PATCH {current_id}, using all-True mask")
                self.agriculture_masks[current_id] = torch.ones(
                    (self.target_size, self.target_size), dtype=torch.bool
                )

        print(f"Loaded {len(self.agriculture_masks)} agriculture masks")

    def _load_downsample_masks(self):
        """Pre-load all downsample reference masks into memory."""
        if self.downsample_mask_path is None:
            raise ValueError("downsample_loss=True but downsample_mask_path is not set")
        print("Loading downsample masks...")
        self.downsample_masks = {}

        for idx in range(len(self.metadata_frame)):
            current_id = self.metadata_frame["ID_PATCH"].iloc[idx]
            mask_path = os.path.join(self.root_dir, f"{self.downsample_mask_path}/mask_{current_id}.npy")

            try:
                mask = np.load(mask_path)
                self.downsample_masks[current_id] = torch.from_numpy(mask.astype(bool))
            except FileNotFoundError:
                print(f"Warning: Downsample mask not found for ID_PATCH {current_id}, using all-True mask")
                self.downsample_masks[current_id] = torch.ones(
                    (self.target_size, self.target_size), dtype=torch.bool
                )

        print(f"Loaded {len(self.downsample_masks)} downsample masks")

    def __len__(self):
        return len(self.metadata_frame)
    
    def __getitem__(self, idx):
        current_id = self.metadata_frame["ID_PATCH"].iloc[idx]
        if not self.load_X_cloud:
            X, dates, y = self._load_raw_cached(current_id)
        else:
            X, X_cloud, dates, y = self._load_raw_cached(current_id)

        if self.transfer is not None and not self.transfer.get("transfer"):
            y = self.transfer["function"](metadata_frame=self.metadata_frame, idx=idx, X=X, args=self.transfer["params"])
            
        if self.constant_length and X.shape[0] > self.constant_length:
            if self.even_sample:
                indices = np.linspace(0, X.shape[0] - 1, self.constant_length, dtype=np.int64)
                X = X[indices, :, :, :]
                dates = dates[indices]
                if self.load_X_cloud:
                    X_cloud = X_cloud[indices,:,:,:]
            else:
                X = X[:self.constant_length, :, :, :]
                dates = dates[:self.constant_length]
                if self.load_X_cloud:
                    X_cloud = X_cloud[:self.constant_length,:,:,:]

        # Build masks dict for any masks that are needed
        masks_in = {}
        if self.provide_agriculture_masks:
            masks_in["ag"] = self.agriculture_masks[current_id]
        if self.downsample_loss:
            masks_in["ds"] = self.downsample_masks[current_id]
        if self.load_y_id:
            y_id = self.io_manager.load_y_id(identifier=current_id)
            masks_in["y_id"] = torch.from_numpy(y_id.astype(np.int64))
        if self.transfer is not None and self.transfer.get("transfer"):
            patch_idx = self.transfer["params"]["patch_ids"].get(int(current_id))
            if patch_idx is not None:
                pixel_weight = self.transfer["params"]["f0_weights"][patch_idx].clone()
            else:
                pixel_weight = torch.ones(self.target_size, self.target_size)
            masks_in["pw"] = pixel_weight

        if self.transform:
            kwargs = {}
            if masks_in:
                kwargs["masks"] = masks_in
            if self.load_X_cloud:
                kwargs["X_cloud"] = X_cloud
            result = self.transform(X, y, **kwargs)

            it = iter(result)
            X = next(it)
            if self.load_X_cloud:
                X_cloud = next(it)
            y = next(it)
            masks_out = next(it) if masks_in else {}
        else:
            X = torch.from_numpy(X.astype(np.float32))
            y = torch.from_numpy(y.astype(np.float32))
            if self.load_X_cloud:
                X_cloud = torch.from_numpy(X_cloud.astype(np.float32))
            masks_out = masks_in

        inputs = (X, dates, X_cloud) if self.load_X_cloud else (X, dates)

        if masks_out:
            return inputs, y, masks_out
        else:
            return inputs, y



    


def setup_dataloader(
    metadata_frame,
    data_path,
    image_count,
    num_buckets,
    normalization,
    y_transform,
    inclusion_intervals,
    batch_size,
    augmentation_func=None,
    use_memmap=False,
    y_threshold=None,
    shuffle=True,
    is_train=False,
    provide_agriculture_masks=False,
    agriculture_mask_path=None,
    downsample_loss=False,
    downsample_mask_path=None,
    load_X_cloud=False,
    load_y_id=False,
    cloud_threshold=None,
    cloud_band=0,
    cache_dir=None,
    io_manager_kwargs=None,
    transfer=None,
    num_workers=8,
):
    """
    Create a single dataloader for a given phase.

    Args:
        metadata_frame: Metadata DataFrame for this phase
        data_path: Root directory of the dataset
        image_count: Number of images per sample
        num_buckets: Number of classification buckets
        normalization: Normalization strategy
        y_transform: Target transform strategy
        inclusion_intervals: Optional intervals for included loss
        augmentation_func: Augmentation function to apply (only used if is_train=True)
        y_threshold: Optional threshold for target values
        use_memmap: Whether to use memory-mapped files
        batch_size: Batch size for the dataloader
        is_train: Whether this dataloader is for the training phase

    Returns:
        DataLoader: A configured DataLoader instance
    """
    from functools import partial
    from torch.utils.data import DataLoader
    from src.training.train_utils import collate_fn
    from src.training.transforms import YTransform, compound_transform

    return DataLoader(
        HedgementationDataset(
            metadata_frame=metadata_frame,
            root_dir=data_path,
            constant_length=image_count,
            use_memmap=use_memmap,
            transform=partial(
                compound_transform,
                num_buckets=num_buckets,
                normalization=normalization,
                y_transform=(
                    y_transform if not inclusion_intervals else YTransform.NONE
                ),
                augmentation=(augmentation_func if is_train else None),
                y_threshold=y_threshold,
            ),
            provide_agriculture_masks=provide_agriculture_masks,
            rpg_mask_subdir=agriculture_mask_path,
            downsample_loss=downsample_loss,
            downsample_mask_path=downsample_mask_path,
            load_X_cloud=load_X_cloud,
            load_y_id=load_y_id,
            cloud_threshold=cloud_threshold,
            cloud_band=cloud_band,
            cache_dir=cache_dir,
            io_manager_kwargs=io_manager_kwargs,
            transfer=transfer
        ),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def setup_dataloaders(
    metadata_frames,
    data_path,
    image_count,
    num_buckets,
    normalization,
    y_transform,
    inclusion_intervals,
    augmentation_func,
    y_threshold,
    use_memmap,
    batch_size,
    provide_agriculture_masks,
    agriculture_mask_path,
    downsample_loss=False,
    downsample_mask_path=None,
    load_X_cloud=False,
    load_y_id=False,
    cloud_threshold=None,
    cloud_band=0,
    cache_dir=None,
    io_manager_kwargs=None,
    transfer=None,
    num_workers=8,
):
    """
    Create dataloaders for training, validation, and test sets.

    Args:
        metadata_frames: Dictionary mapping phase names to metadata DataFrames
        data_path: Root directory of the dataset
        image_count: Number of images per sample
        num_buckets: Number of classification buckets
        normalization: Normalization strategy
        y_transform: Target transform strategy
        inclusion_intervals: Optional intervals for included loss
        augmentation_func: Augmentation function to apply
        y_threshold: Optional threshold for target values
        use_memmap: Whether to use memory-mapped files
        batch_size: Batch size for dataloaders

    Returns:
        dict: Dictionary mapping phase names to DataLoaders
    """
    if transfer is not None and transfer.get("transfer"):
        ids_path = transfer["params"]["ids_path"]
        f0_path = transfer["params"]["f0_path"]
        f0_weights = torch.load(f0_path)[:, 1, :, :]
        val_max = transfer["params"]["clipping"]
        f0_weights = torch.clamp(f0_weights, max=val_max)
        patch_ids = torch.load(ids_path)
        patch_ids_dict = {int(pid): i for i, pid in enumerate(patch_ids)}
        transfer["params"]["patch_ids"] = patch_ids_dict
        transfer["params"]["f0_weights"] = f0_weights

    return {
        phase: setup_dataloader(
            metadata_frame=metadata_frames[phase],
            data_path=data_path,
            image_count=image_count,
            num_buckets=num_buckets,
            normalization=normalization,
            y_transform=y_transform,
            inclusion_intervals=inclusion_intervals,
            augmentation_func=augmentation_func,
            y_threshold=y_threshold,
            use_memmap=use_memmap,
            batch_size=batch_size,
            is_train=(phase == "train"),
            provide_agriculture_masks=provide_agriculture_masks,
            agriculture_mask_path=agriculture_mask_path,
            downsample_loss=downsample_loss,
            downsample_mask_path=downsample_mask_path,
            load_X_cloud=load_X_cloud,
            load_y_id=load_y_id,
            cloud_threshold=cloud_threshold,
            cloud_band=cloud_band,
            cache_dir=cache_dir,
            io_manager_kwargs=io_manager_kwargs,
            transfer=transfer,
            num_workers=num_workers,
        )
        for phase in metadata_frames
    }