"""
Webdataset format:
{ID}.X.tif
{ID}.X_cloud.tif
{ID}.y.tif
{ID}.y_id.tif
{ID}.rpg_{mask_type}.tif
{ID}.aef.npy

"""

import io
from typing import Optional

import numpy as np
import webdataset as wds

try:
    from rasterio.io import MemoryFile
except ImportError:
    MemoryFile = None

from hedgementation_utils.io.io_manager import BaseHedgementationIOManager


class WDSHedgementationIOManager(BaseHedgementationIOManager):
    def __init__(self,
                 dataset_root: str,
                 num_X_bands: int = 10,
                 num_X_cloud_bands: int = 2,
                 img_width: int = 128,
                 img_height: int = 128,
                 default_cloud_threshold: float = 0.2,
                 default_cloud_filtering_band: int = 0,
                 wds_cache_dir: Optional[str] = None,
                 X_suffix: str = "x.tif",
                 X_cloud_suffix: str = "x_cloud.tif",
                 y_suffix: str = "y.tif",
                 y_id_suffix: str = "y_id.tif",
                 rpg_suffix_pattern: str = "rpg_{}.tif",
                 aef_suffix: str = "aef.npy"
                 ):
        self.dataset_root = dataset_root
        self.num_X_bands = num_X_bands
        self.num_X_cloud_bands = num_X_cloud_bands
        self.img_width = img_width
        self.img_height = img_height
        self.default_cloud_threshold = default_cloud_threshold
        self.default_cloud_filtering_band = default_cloud_filtering_band

        self.X_suffix = X_suffix
        self.X_cloud_suffix = X_cloud_suffix
        self.y_suffix = y_suffix
        self.y_id_suffix = y_id_suffix
        self.rpg_suffix_pattern = rpg_suffix_pattern
        self.aef_suffix = aef_suffix

        dataset = wds.WebDataset(dataset_root, cache_dir=wds_cache_dir)
        self._index: dict[str, dict] = {
            sample["__key__"]: sample for sample in dataset
        }

    def _lookup(self, identifier) -> dict:
        key = str(identifier)
        if key not in self._index:
            raise KeyError(f"Identifier '{key}' not found in webdataset.")
        return self._index[key]

    def _load_tif_bytes(self, tif_bytes: bytes, num_bands: int, reshape: bool, return_dates: bool):
        with MemoryFile(tif_bytes) as memfile:
            with memfile.open() as src:
                data = src.read()
                if reshape:
                    data = data.reshape((-1, num_bands, self.img_height, self.img_width))
                if return_dates:
                    dates = self._bands_to_dates(src.descriptions, n_bands_per_timestep=num_bands)
                    return data, dates
        return data

    def load_X(self,
               identifier=None,
               load_path: Optional[str] = None,
               file_type: Optional[str] = None,
               return_as_dataset: bool = False,
               return_dates: bool = False,
               reshape: bool = True):
        if return_as_dataset:
            raise NotImplementedError("return_as_dataset is not supported for WebDataset loading.")
        sample = self._lookup(identifier)
        return self._load_tif_bytes(sample[self.X_suffix], self.num_X_bands, reshape, return_dates)

    def load_X_cloud(self,
                     identifier=None,
                     load_path: Optional[str] = None,
                     file_type: Optional[str] = None,
                     return_as_dataset: bool = False,
                     return_dates: bool = False,
                     reshape: bool = True):
        if return_as_dataset:
            raise NotImplementedError("return_as_dataset is not supported for WebDataset loading.")
        sample = self._lookup(identifier)
        return self._load_tif_bytes(sample[self.X_cloud_suffix], self.num_X_cloud_bands, reshape, return_dates)

    def load_y(self,
               identifier=None,
               load_path: Optional[str] = None,
               file_type: Optional[str] = None,
               return_as_dataset: bool = False):
        if return_as_dataset:
            raise NotImplementedError("return_as_dataset is not supported for WebDataset loading.")
        sample = self._lookup(identifier)
        with MemoryFile(sample[self.y_suffix]) as memfile:
            with memfile.open() as src:
                return np.squeeze(src.read())

    def load_y_id(self,
                  identifier=None,
                  load_path: Optional[str] = None,
                  file_type: Optional[str] = None,
                  return_as_dataset: bool = False):
        if return_as_dataset:
            raise NotImplementedError("return_as_dataset is not supported for WebDataset loading.")
        sample = self._lookup(identifier)
        with MemoryFile(sample[self.y_id_suffix]) as memfile:
            with memfile.open() as src:
                return np.squeeze(src.read())

    def load_rpg_mask(self,
                      identifier=None,
                      type: str = "unbuffered",
                      load_path: Optional[str] = None,
                      file_type: Optional[str] = None):
        sample = self._lookup(identifier)
        rpg_key = self.rpg_suffix_pattern.format(type)
        if rpg_key not in sample:
            raise KeyError(f"No {rpg_key} key found in sample '{identifier}'.")
        with MemoryFile(sample[rpg_key]) as memfile:
            with memfile.open() as src:
                return np.squeeze(src.read())

    def load_aef(self,
                 identifier=None,
                 load_path: Optional[str] = None,
                 file_type: Optional[str] = None):
        # AEF embeddings are stored as raw .npy bytes of shape (C, H, W).
        sample = self._lookup(identifier)
        return np.load(io.BytesIO(sample[self.aef_suffix]))
