
from collections import Counter
from typing import Optional
import pandas as pd
import pathlib
try:
    import rasterio
except ImportError:
    rasterio = None
import numpy as np


"""
Webdataset format:
{ID}.X.tif
{ID}.X_cloud.tif
{ID}.y.tif
{ID}.y_id.tif
{ID}.rpg_{mask_type}.tif
{ID}.aef.npy
....
"""


class BaseHedgementationIOManager:
    def _bands_to_dates(self,
                        bands,
                        n_bands_per_timestep):
        individual_dates = bands[::n_bands_per_timestep]
        dates = pd.Series([d.split("_")[0] for d in individual_dates]).sort_values().reset_index(drop=True)
        return dates

    def _get_mean_cloud_scores(self, X_cloud, band_ind=0):
        return np.mean(X_cloud[:, band_ind, :, :], axis=(1, 2))

    def _limit_above_cloud_threshold(
            self,
            X: np.ndarray,
            X_cloud: np.ndarray,
            dates: Optional[pd.Series] = None,
            threshold: float = 0.2,
            band_ind=0
    ):
        mean_score_per_timestep = self._get_mean_cloud_scores(X_cloud, band_ind)
        inds = mean_score_per_timestep > threshold
        X_unobstructed = X[inds, :, :, :]
        cloud_unobstructed = X_cloud[inds, :, :, :]
        if dates is not None:
            dates_unobstructed = dates[inds]
            return X_unobstructed, cloud_unobstructed, dates_unobstructed
        else:
            return X_unobstructed, cloud_unobstructed

    def _filter_to_consensus(self,
                             dates: pd.Series,
                             consensus_counter: Counter):
        occurrence_rank = dates.groupby(dates).cumcount() + 1
        allowed = dates.map(consensus_counter).fillna(0)
        return list(dates[occurrence_rank <= allowed].index)

    def _find_consensus_dates(self,
                              dates_1: pd.Series,
                              dates_2: pd.Series,
                              return_inds=True):
        consensus_counter = Counter(dates_1) & Counter(dates_2)
        consensus = pd.Series(sorted(consensus_counter.elements()))
        if return_inds:
            return consensus, self._filter_to_consensus(dates_1, consensus_counter), self._filter_to_consensus(dates_2, consensus_counter)
        return consensus

    def load_X_data_date_aligned(self,
                                 identifier,
                                 X_kwargs: Optional[dict] = None,
                                 X_cloud_kwargs: Optional[dict] = None):
        X_kwargs = {k: v for k, v in (X_kwargs or {}).items() if k != "return_dates"}
        X_cloud_kwargs = {k: v for k, v in (X_cloud_kwargs or {}).items() if k != "return_dates"}
        X, X_dates = self.load_X(identifier, **X_kwargs, return_dates=True)
        X_cloud, X_cloud_dates = self.load_X_cloud(identifier, **X_cloud_kwargs, return_dates=True)

        consensus_dates, X_inds, X_cloud_inds = self._find_consensus_dates(X_dates, X_cloud_dates)

        X = X[X_inds, :, :, :]
        X_cloud = X_cloud[X_cloud_inds, :, :, :]
        return X, X_cloud, consensus_dates

    def load_X_data_with_cloud_threshold(self,
                                         identifier: int,
                                         cloud_threshold: Optional[float] = None,
                                         band: Optional[int] = None,
                                         X_kwargs: Optional[dict] = None,
                                         X_cloud_kwargs: Optional[dict] = None
                                         ):
        if cloud_threshold is None:
            cloud_threshold = self.default_cloud_threshold
        if band is None:
            band = self.default_cloud_filtering_band

        X, X_cloud, dates = self.load_X_data_date_aligned(identifier=identifier,
                                                           X_kwargs=X_kwargs,
                                                           X_cloud_kwargs=X_cloud_kwargs)

        X, X_cloud, X_dates = self._limit_above_cloud_threshold(
            X=X,
            X_cloud=X_cloud,
            dates=dates,
            threshold=cloud_threshold,
            band_ind=band
        )
        return X, X_cloud, X_dates


class HedgementationIOManager(BaseHedgementationIOManager):
    def __init__(self,
                 dataset_root: str,
                 X_dir: str = "X",
                 y_dir: str = "y",
                 X_cloud_dir: str = "X_cloud",
                 y_id_dir: str = "y_id",
                 rpg_mask_dir: str = "rpg",
                 aef_dir: str = "AEF",
                 X_pattern: str = "X_{}",
                 y_pattern: str = "y_{}",
                 X_cloud_pattern: str = "X_cloud_{}",
                 y_id_pattern: str = "y_id_{}",
                 rpg_pattern: str = "rpg_{}",
                 aef_pattern: str = "AEF_{}",
                 X_file_type="tif",
                 y_file_type="tif",
                 X_cloud_file_type="tif",
                 y_id_file_type="tif",
                 rpg_file_type="npy",
                 aef_file_type="npy",
                 num_X_bands: int = 10,
                 num_X_cloud_bands: int = 2,
                 num_aef_bands: int = 64,
                 img_width: int = 128,
                 img_height: int = 128,
                 default_cloud_threshold: float = 0.2,
                 default_cloud_filtering_band: int = 0,
                 ):

        self.dataset_root = dataset_root
        self.X_dir = pathlib.Path(self.dataset_root, X_dir)
        self.y_dir = pathlib.Path(self.dataset_root, y_dir)
        self.X_cloud_dir = pathlib.Path(self.dataset_root, X_cloud_dir)
        self.y_id_dir = pathlib.Path(self.dataset_root, y_id_dir)
        self.rpg_mask_dir = pathlib.Path(self.dataset_root, rpg_mask_dir)
        self.aef_dir = pathlib.Path(self.dataset_root, aef_dir)

        self.X_pattern = X_pattern
        self.y_pattern = y_pattern
        self.X_cloud_pattern = X_cloud_pattern
        self.y_id_pattern = y_id_pattern
        self.rpg_pattern = rpg_pattern
        self.aef_pattern = aef_pattern

        self.X_file_type = X_file_type
        self.y_file_type = y_file_type
        self.X_cloud_file_type = X_cloud_file_type
        self.y_id_file_type = y_id_file_type
        self.rpg_file_type = rpg_file_type
        self.aef_file_type = aef_file_type

        self.num_X_bands = num_X_bands
        self.num_X_cloud_bands = num_X_cloud_bands
        self.num_aef_bands = num_aef_bands

        self.img_width = img_width
        self.img_height = img_height
        self.default_cloud_threshold = default_cloud_threshold
        self.default_cloud_filtering_band = default_cloud_filtering_band

    def get_load_path(self,
                      identifier: int,
                      load_path: str,
                      file_type: str,
                      directory: str,
                      pattern: str):
        if identifier is not None and load_path is not None:
            raise ValueError("Provide either identifier or load_path, not both.")
        if load_path is None:
            if identifier is None:
                raise ValueError("Either the load path or identifier must be provided.")
            load_path = pathlib.Path(directory, f"{pattern.format(identifier)}.{file_type}")
        else:
            load_path = pathlib.Path(load_path)
        return load_path

    def _load_y(self,
                identifier: Optional[int],
                load_path: Optional[str],
                file_type: str,
                directory: str,
                pattern: str,
                return_as_dataset: bool):
        load_path = self.get_load_path(identifier, load_path, file_type, directory, pattern)
        if load_path.suffix == ".npy":
            with open(load_path, "rb") as infile:
                data = np.load(infile)
                return np.squeeze(data)
        elif load_path.suffix == ".tif":
            if return_as_dataset:
                return rasterio.open(load_path)
            else:
                with rasterio.open(load_path) as src:
                    data = src.read()
                    return np.squeeze(data)
        else:
            with open(load_path) as infile:
                return infile.read()

    def _load_X(self,
                identifier: Optional[int],
                load_path: Optional[str],
                file_type: str,
                return_as_dataset: bool,
                return_dates: bool,
                directory: str,
                pattern: str,
                num_bands: int,
                reshape: bool):
        load_path = self.get_load_path(identifier, load_path, file_type, directory, pattern)

        if load_path.suffix == ".npy":
            if return_dates:
                raise ValueError("return_dates=True is only supported for .tif inputs; .npy files do not carry per-band metadata.")
            with open(load_path, "rb") as infile:
                data = np.load(infile)
                if reshape:
                    data = data.reshape((-1, num_bands, self.img_height, self.img_width))
                return data
        elif load_path.suffix == ".tif":
            if return_as_dataset:
                return rasterio.open(load_path)
            else:
                with rasterio.open(load_path) as src:
                    data = src.read()
                    if reshape:
                        data = data.reshape((-1, num_bands, self.img_height, self.img_width))
                    if return_dates:
                        descs = src.descriptions
                        dates = self._bands_to_dates(descs, n_bands_per_timestep=num_bands)
                        return data, dates
                    return data
        else:
            with open(load_path) as infile:
                return infile.read()

    def load_X(self,
               identifier: Optional[int] = None,
               load_path: Optional[str] = None,
               file_type: Optional[str] = None,
               return_as_dataset: bool = False,
               return_dates: bool = False,
               reshape: bool = True):
        return self._load_X(
            identifier=identifier,
            load_path=load_path,
            file_type=file_type or self.X_file_type,
            return_as_dataset=return_as_dataset,
            return_dates=return_dates,
            directory=self.X_dir,
            pattern=self.X_pattern,
            num_bands=self.num_X_bands,
            reshape=reshape
        )

    def load_X_cloud(self,
                     identifier: Optional[int] = None,
                     load_path: Optional[str] = None,
                     file_type: Optional[str] = None,
                     return_as_dataset: bool = False,
                     return_dates: bool = False,
                     reshape: bool = True):
        return self._load_X(
            identifier=identifier,
            load_path=load_path,
            file_type=file_type or self.X_cloud_file_type,
            return_as_dataset=return_as_dataset,
            return_dates=return_dates,
            directory=self.X_cloud_dir,
            pattern=self.X_cloud_pattern,
            num_bands=self.num_X_cloud_bands,
            reshape=reshape
        )

    def load_y(self,
               identifier: Optional[int] = None,
               load_path: Optional[str] = None,
               file_type: Optional[str] = None,
               return_as_dataset: bool = False):
        return self._load_y(
            identifier=identifier,
            load_path=load_path,
            directory=self.y_dir,
            pattern=self.y_pattern,
            file_type=file_type or self.y_file_type,
            return_as_dataset=return_as_dataset
        )

    def load_y_id(self,
                  identifier: Optional[int] = None,
                  load_path: Optional[str] = None,
                  file_type: Optional[str] = None,
                  return_as_dataset: bool = False):
        return self._load_y(
            identifier=identifier,
            load_path=load_path,
            directory=self.y_id_dir,
            pattern=self.y_id_pattern,
            file_type=file_type or self.y_id_file_type,
            return_as_dataset=return_as_dataset
        )

    def load_rpg_mask(self,
                      identifier: Optional[int] = None,
                      load_path: Optional[str] = None,
                      file_type: Optional[str] = None):
        return self._load_y(
            identifier=identifier,
            load_path=load_path,
            file_type=file_type or self.rpg_file_type,
            directory=self.rpg_mask_dir,
            pattern=self.rpg_pattern,
            return_as_dataset=False
        )

    def load_aef(self,
                 identifier: Optional[int] = None,
                 load_path: Optional[str] = None,
                 file_type: Optional[str] = None):
        # AEF embeddings are already (C, H, W); no reshape and no per-band dates.
        return self._load_X(
            identifier=identifier,
            load_path=load_path,
            file_type=file_type or self.aef_file_type,
            return_as_dataset=False,
            return_dates=False,
            directory=self.aef_dir,
            pattern=self.aef_pattern,
            num_bands=self.num_aef_bands,
            reshape=False,
        )
