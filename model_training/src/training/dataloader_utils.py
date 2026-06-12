  
from functools import partial

import torch
from torch.utils.data import DataLoader

from src.training.hedgementation_dataset import HedgementationDataset
from src.training.trainer_config import TrainerConfig
from src.training.transforms import YTransform, compound_transform


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
            provide_rpg_masks=provide_agriculture_masks,
            rpg_mask_subdir=agriculture_mask_path,
            downsample_loss=downsample_loss,
            downsample_mask_dir=downsample_mask_path,
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
        pin_memory=torch.cuda.is_available(),
        #persistent_workers=num_workers > 0,
        #prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )


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

def dataloader_from_config(
    config: TrainerConfig,
    split: str,
    shuffle: bool = True
):
    
    return DataLoader(
        dataset=HedgementationDataset._from_config(config=config, split=split),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        #persistent_workers=num_workers > 0,
        #prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )

def dataloaders_from_config(
    config: TrainerConfig,
    shuffle: bool = True
):
    return {
        split: dataloader_from_config(
            config=config,
            split=split,
            shuffle=shuffle
        ) for split in config.metadata_frames
    }