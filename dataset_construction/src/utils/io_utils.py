import pathlib
from typing import Optional

import numpy as np
import os

from dotenv import load_dotenv
import os

import rasterio

import pandas as pd



load_dotenv()

DATASET_ROOT = os.environ["DATASET_ROOT"]
RPG_MASK_SUBDIR = os.environ["RPG_MASK_SUBDIR"]
MASK_SAVE_DIR = os.path.join(DATASET_ROOT, RPG_MASK_SUBDIR)
HEDGE_SAVE_DIR = os.path.join(DATASET_ROOT, "y")
SATELLITE_SAVE_DIR = os.path.join(DATASET_ROOT, "X")


MASK_PATTERN = "mask_{identifier}.npy"
Y_PATTERN = "y_{identifier}.npy"
X_PATTERN = "X_{identifier}.npy"

def save_rpg_mask(mask,
                  identifier=None,
                  save_path=None,
                  mask_dir=MASK_SAVE_DIR):
    if identifier is not None:
        with open(os.path.join(mask_dir, MASK_PATTERN.format(identifier=identifier)), "wb+") as outfile:
            np.save(outfile, mask)
    elif save_path is not None:
        with open(save_path, "wb+") as outfile:
            np.save(outfile, mask)
    else:
        raise ValueError("Either the save path or identifier must be provided.")


def load_rpg_mask(identifier=None,
                  load_path=None,
                  mask_dir=MASK_SAVE_DIR):  
    if identifier is not None:
        with open(os.path.join(mask_dir, MASK_PATTERN.format(identifier=identifier)), "rb+") as infile:
            data = np.load(infile)
            return data
    elif load_path is not None:
        with open(load_path, "rb+") as infile:
            data = np.load(infile)
            return data
    else:
        raise ValueError("Either the load path or identifier must be provided.")
    
def load_hedgerow_raster(identifier,
                  hedge_dir=HEDGE_SAVE_DIR):
    with open(os.path.join(hedge_dir, Y_PATTERN.format(identifier=identifier)), "rb+") as infile:
        raster = np.load(infile) > 0
    raster = np.squeeze(raster)[:128,:128]
    return raster

def load_satellite_img(patch_id,
                       satellite_dir=SATELLITE_SAVE_DIR):
    with open(os.path.join(satellite_dir, X_PATTERN.format(identifier=patch_id)), "rb+") as infile:
        patch = np.load(infile)
    
    patch = patch.reshape((-1,10,129,129))
    patch_img = patch[0,:,:128,:128]
    return patch_img

class HedgementationIOManager:
    def __init__(self,
        dataset_root:str,
        X_dir:str="X",
        y_dir:str="y",
        X_cloud_dir:str="X_cloud",
        y_id_dir:str="y_id",
        rpg_mask_dir:str = "rpg",
        X_pattern:str="X_{}",
        y_pattern:str="y_{}",
        X_cloud_pattern:str="X_cloud_{}",
        y_id_pattern:str="y_id_{}",
        rpg_pattern:str="rpg_{}",
        X_file_type="npy",
        y_file_type="npy",
        X_cloud_file_type="npy",
        y_id_file_type="npy",
        rpg_file_type="npy",
        num_X_bands:int=10,
        num_X_cloud_bands:int=2,
        img_width:int=128,
        img_height:int=128
        ):

        self.dataset_root = dataset_root
        self.X_dir = pathlib.Path(self.dataset_root, X_dir)
        self.y_dir = pathlib.Path(self.dataset_root, y_dir)
        self.X_cloud_dir = pathlib.Path(self.dataset_root, X_cloud_dir)
        self.y_id_dir = pathlib.Path(self.dataset_root, y_id_dir)
        self.rpg_mask_dir = pathlib.Path(self.dataset_root, rpg_mask_dir)

        self.X_pattern = X_pattern
        self.y_pattern = y_pattern
        self.X_cloud_pattern = X_cloud_pattern
        self.y_id_pattern = y_id_pattern
        self.rpg_pattern = rpg_pattern

        self.X_file_type = X_file_type
        self.y_file_type = y_file_type
        self.X_cloud_file_type = X_cloud_file_type
        self.y_id_file_type = y_id_file_type
        self.rpg_file_type = rpg_file_type

        self.num_X_bands = num_X_bands
        self.num_X_cloud_bands = num_X_cloud_bands

        self.img_width = img_width
        self.img_height = img_height

    def get_load_path(self,
                      identifier:int,
                      load_path:str,
                      file_type:str,
                      directory:str,
                      pattern:str):
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
                identifier:Optional[int],
               load_path:Optional[str],
               file_type:str,
               directory:str,
               pattern:str):
        load_path = self.get_load_path(identifier, load_path, file_type, directory, pattern)
        if load_path.suffix == ".npy":
            with open(load_path,"rb") as infile:
                data = np.load(infile)
                return np.squeeze(data)
        elif load_path.suffix == ".tif":
            with rasterio.open(load_path) as src:
                data = src.read()
                return np.squeeze(data)
        else:
            with open(load_path) as infile:
                return infile.read()
    
    def _bands_to_dates(self,
                        bands):
        return pd.Series(list((set([d.split("_")[0] for d in bands])))).sort_values()
    
    def _load_X(self,
                identifier:Optional[int],
               load_path:Optional[str],
               file_type:str,
               return_as_dataset:bool,
               return_dates:bool,
               directory:str,
               pattern:str,
               num_bands:int):
        load_path = self.get_load_path(identifier, load_path, file_type, directory, pattern)

        if load_path.suffix == ".npy":
            with open(load_path,"rb") as infile:
                data = np.load(infile)
                data = data.reshape((-1, num_bands, self.img_height, self.img_width))
                return data
        elif load_path.suffix == ".tif":
            if return_as_dataset:
                return rasterio.open(load_path)
            else:
                with rasterio.open(load_path) as src:
                    data = src.read()
                    data = data.reshape((-1, num_bands, self.img_height, self.img_width))
                    if return_dates:
                        descs = src.descriptions
                        dates = self._bands_to_dates(descs)
                        return data, dates
                    return data
        else:
            with open(load_path) as infile:
                return infile.read()
    def load_X(self,
               identifier:Optional[int]=None,
               load_path:Optional[str]=None,
               file_type:Optional[str]=None,
               return_as_dataset:bool = False,
               return_dates:bool = False):
        return self._load_X(
            identifier=identifier,
            load_path=load_path,
            file_type=file_type or self.X_file_type,
            return_as_dataset=return_as_dataset,
            return_dates=return_dates,
            directory=self.X_dir,
            pattern=self.X_pattern,
            num_bands=self.num_X_bands
        )

    def load_X_cloud(self,
               identifier:Optional[int]=None,
               load_path:Optional[str]=None,
               file_type:Optional[str]=None,
               return_as_dataset:bool = False,
               return_dates:bool = False):
        return self._load_X(
            identifier=identifier,
            load_path=load_path,
            file_type=file_type or self.X_cloud_file_type,
            return_as_dataset=return_as_dataset,
            return_dates=return_dates,
            directory=self.X_cloud_dir,
            pattern=self.X_cloud_pattern,
            num_bands=self.num_X_cloud_bands
        )
        
    def load_y(self,
               identifier:Optional[int]=None,
               load_path:Optional[str]=None,
               file_type:Optional[str]=None):
        return self._load_y(
            identifier=identifier,
            load_path=load_path,
            directory=self.y_dir,
            pattern=self.y_pattern,
            file_type=file_type or self.y_file_type
        )
    
    def load_y_id(self,
               identifier:Optional[int]=None,
               load_path:Optional[str]=None,
               file_type:Optional[str]=None):
        return self._load_y(
            identifier=identifier,
            load_path=load_path,
            directory=self.y_id_dir,
            pattern=self.y_id_pattern,
            file_type=file_type or self.y_id_file_type
        )
    
    def load_rpg_mask(self,
                identifier:Optional[int]=None,
               load_path:Optional[str]=None,
               file_type:Optional[str]=None):  
        return self._load_y(
            identifier=identifier,
            load_path=load_path,
            file_type=file_type or self.rpg_file_type,
            directory=self.rpg_mask_dir,
            pattern=self.rpg_pattern
            )