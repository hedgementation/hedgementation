import datetime
from typing import Optional
from typing import Optional
import torch
import pathlib
import pandas as pd
import geopandas as gpd
from torchvision.transforms import v2
import os
import numpy as np
from dotenv import load_dotenv
from hedgementation_utils.io.io_manager import HedgementationIOManager


from functools import partial
from src.training.transforms import AUGMENTATION_TRANSFORMS, YTransform, compound_transform

from src.training.trainer_config import TrainerConfig

load_dotenv()

DATASET_ROOT = pathlib.Path(os.environ["DATASET_ROOT"])
RPG_MASK_SUBDIR = os.environ["RPG_MASK_SUBDIR"]



class HedgementationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 metadata_frame: gpd.GeoDataFrame,
                 root_dir:str=DATASET_ROOT,
                 rpg_mask_subdir:str=RPG_MASK_SUBDIR,
                 constant_length:Optional[int]=None,
                 use_memmap:bool=False,
                 transform=None,
                 provide_rpg_masks=False,
                 downsample_loss=False,
                 downsample_mask_dir=None,
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
        self.provide_rpg_masks = provide_rpg_masks
        self.downsample_loss = downsample_loss
        self.downsample_mask_path = downsample_mask_dir
        self.load_X_cloud = load_X_cloud
        self.load_y_id = load_y_id
        self.cloud_threshold = cloud_threshold
        self.cloud_band = cloud_band
        self.transfer = transfer
        self.length = len(self.metadata_frame)

        if rpg_mask_subdir is None and self.provide_rpg_masks:
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

        self.patch_ids = self.metadata_frame["ID_PATCH"].tolist()

        self.reference_date = datetime.datetime(*map(int, reference_date.split("-")))
        if self.use_memmap:
            self._build_memmap()

        if self.provide_rpg_masks:
            self._load_masks()
        if self.downsample_loss:
            self._load_downsample_masks()
        if self.load_y_id:
            self._load_y_ids()

    @classmethod 
    def create_augmentation_func(cls, augmentation: Optional[list[str]]):
        """Create augmentation function from config."""
        if not augmentation:
            return None

        transforms = [
            AUGMENTATION_TRANSFORMS[aug] for aug in augmentation
        ]
        return v2.Compose(transforms) if len(transforms) > 1 else transforms[0]
    @classmethod
    def _from_config(cls,
                     config: TrainerConfig,
                     split: str = "train"):
        
        is_train = ("train" in split)
        augmentation_func = cls.create_augmentation_func(config.augmentation)
        return HedgementationDataset(
            metadata_frame=config.metadata_frames[split],
            root_dir=config.dataset_root,
            rpg_mask_subdir=config.rpg_mask_subdir,
            constant_length=config.image_count,
            provide_rpg_masks=config.provide_rpg_masks,
            transform=partial(
                compound_transform,
                num_buckets=config.num_buckets,
                normalization=config.normalization,
                y_transform=(
                    config.y_transform if not config.inclusion_intervals else YTransform.NONE
                ),
                augmentation=(augmentation_func if is_train else None),
                y_threshold=config.y_threshold,
            ),
            downsample_loss=config.downsample_loss,
            downsample_mask_dir=config.downsample_mask_dir,
            reference_date=config.reference_date,
            cache_dir = config.cache_dir,
            target_size=config.target_size,
            load_X_cloud=config.load_X_cloud,
            load_y_id=config.load_y_id,
            cloud_band=config.cloud_band,
            cloud_threshold=config.cloud_threshold,
            io_manager_kwargs=config.io_manager_kwargs,
            transfer=config.transfer
        )

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
            except Exception as e:
                print(f"Cache read failed for {cache_path}: {e}")
        if not self.load_X_cloud:
            X, raw_dates = self.io_manager.load_X(identifier=identifier, return_dates=True)
        else:
            if self.cloud_threshold:
                X, X_cloud, raw_dates = self.io_manager.load_X_data_with_cloud_threshold(
                    identifier=identifier,
                    cloud_threshold=self.cloud_threshold,
                    band=self.cloud_band
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

    def _load_y_ids(self):
        """Pre-load all y_id arrays into memory."""
        print("Loading y_id arrays...")
        self.y_ids = {}
        for current_id in self.patch_ids:
            self.y_ids[current_id] = torch.from_numpy(
                self.io_manager.load_y_id(identifier=current_id).astype(np.int64)
            )
        print(f"Loaded {len(self.y_ids)} y_id arrays")

    def __len__(self):
        return self.length
    
    
    def __getitem__(self, idx):
        current_id = self.patch_ids[idx]
        if not self.load_X_cloud:
            X, dates, y = self._load_raw_cached(current_id)
        else:
            X, X_cloud, dates, y = self._load_raw_cached(current_id)

        if self.transfer is not None and not self.transfer.get("transfer"):
            y = self.transfer["function"](metadata_frame=self.metadata_frame, idx=idx, X=X, args=self.transfer["params"])
            
        if self.constant_length and X.shape[0] > self.constant_length:
            if self.even_sample:
                indices = np.linspace(0, X.shape[0] - 1, self.constant_length, dtype=np.int16)
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
        if self.provide_rpg_masks:
            masks_in["ag"] = self.agriculture_masks[current_id]
        if self.downsample_loss:
            masks_in["ds"] = self.downsample_masks[current_id]
        if self.load_y_id:
            masks_in["y_id"] = self.y_ids[current_id]
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



  