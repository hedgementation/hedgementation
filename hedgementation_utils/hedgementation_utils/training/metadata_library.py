import json
from typing import Optional

import geopandas as gpd
import os
from dotenv import load_dotenv

import torch
import random
import numpy as np
from enum import Enum


class THZClass(Enum):
    TEMPERATE = "temperate"
    SUBTROPIC = "subtropic"


class SizeGroup(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    @classmethod
    def coerce(cls, value: "str | SizeGroup") -> "SizeGroup":
        if isinstance(value, cls):
            return value
        return cls[value.upper()]


SUBTROPICS_QUERY = 'thz_class in ["TRC5: Subtropics, cool", "TRC4: Subtropics, moderately cool"]'
TEMPERATE_QUERY = 'thz_class in ["TRC7: Temperate, cool", "TRC6: Temperate, moderately cool"]'
RETURN_ALL_QUERY = 'ilevel_0 in ilevel_0'

THZ_CLASS_QUERY = "thz_class.str.lower().str.contains({cls})"
GEOGRAPHIC_DISTANCE_QUERY = "tile_group != {ind}"

load_dotenv()


DATASET_ROOT = os.getenv("DATASET_ROOT","")

TRAIN_FOLDS = json.loads(os.getenv("TRAIN_FOLDS", "[0,1,2]"))
VALID_FOLDS = json.loads(os.getenv("VALID_FOLDS", "[3]"))
TEST_FOLDS = json.loads(os.getenv("TEST_FOLDS", "[4]"))

NUM_TILEGROUPS = int(os.environ.get("NUM_TILEGROUPS",5))

RANDOM_SEED = int(os.environ.get("HEDGEMENTATION_RANDOM_SEED",15))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

"""
A central repository class for tracking metadata for different experiments.

Defines a number of internal methods for acquiring certain frames, and 
also has a list of predefined metadata frames that can be retrieved
for certain experiments.
"""


class MetadataLibrary():
    def __init__(self,
                 metadata: Optional[gpd.GeoDataFrame]=None,
                 train_folds: Optional[list[int]]=None,
                 valid_folds: Optional[list[int]]=None,
                 test_folds: Optional[list[int]]=None):
        if metadata is None:
            metadata = gpd.read_file(
                os.path.join(DATASET_ROOT, "metadata.geojson")
            )
        self.metadata = metadata

        self.train_folds = train_folds if train_folds else TRAIN_FOLDS
        self.valid_folds = valid_folds if valid_folds else VALID_FOLDS
        self.test_folds = test_folds if test_folds else TEST_FOLDS

        self.fold_dict = {
            "train": self.train_folds,
            "valid": self.valid_folds,
            "test": self.test_folds
        }

    @staticmethod
    def _apply_size_group_filter(metadata: gpd.GeoDataFrame,
                                 size_group: "SizeGroup") -> gpd.GeoDataFrame:
        """
        Restricts metadata to rows whose `size_group` is <= the requested size group.
        If the column is absent, returns the frame unchanged for backwards compatibility.
        """
        if "size_group" not in metadata.columns:
            return metadata
        allowed = {sg.name.lower() for sg in SizeGroup if sg.value <= size_group.value}
        return metadata[metadata["size_group"].str.lower().isin(allowed)]

    def _resolve_metadata(self,
                          metadata: Optional[gpd.GeoDataFrame],
                          size_group: "Optional[str | SizeGroup]") -> gpd.GeoDataFrame:
        """
        Returns the metadata to use: explicit `metadata` if provided, otherwise
        `self.metadata`, optionally restricted by `size_group`.
        """
        if metadata is not None:
            return metadata
        if size_group is None:
            return self.metadata
        return self._apply_size_group_filter(self.metadata, SizeGroup.coerce(size_group))

    def get_single_split(self,
                    metadata: gpd.GeoDataFrame = None,
                    split: str = "train",
                    size_group: "Optional[str | SizeGroup]" = None) -> gpd.GeoDataFrame:
        """
        Retrieves the patches from a GeoDataFrame that fall into a given split. If no
        metadata is provided, falls back to the MetadataLibrary's default metadata
        dataframe, optionally restricted to the given `size_group`.
        """
        metadata = self._resolve_metadata(metadata, size_group)
        if split not in self.fold_dict:
            raise ValueError("Invalid split.")
        return metadata[metadata["fold"].isin(
            self.fold_dict[split]
            )
        ]

    def get_split_metadata(self,
                           metadata=None,
                           query: Optional[str | dict[str]] = None,
                           size_group: "Optional[str | SizeGroup]" = None) -> dict[str,gpd.GeoDataFrame]:
        """
        Splits a set of metadata into train, valid and test splits, returning a dictionary that
        maps each split to a GeoDataFrame of the tiles that fall into that split. If no metadata
        frame is provided, will fall back to the MetadataLibrary's default metadata GeoDataFrame,
        optionally restricted to the given `size_group`.
        """
        metadata = self._resolve_metadata(metadata, size_group)
        retval = {}

        for split in ["train", "valid", "test"]:
            retval[split] = self.get_single_split(split=split, metadata=metadata)

        if query:
            if isinstance(query,str):
                for split in retval:
                    retval[split] = retval[split].query(query)
            elif isinstance(query, dict[str]):
                for split in retval:
                    retval[split] = retval[split].query(
                        query[split] if split in query else RETURN_ALL_QUERY
                    )

            else:
                raise ValueError("Query must either be a string or a string-to-string dict.")

        return retval

    def get_metadata_queried(self,
                           query: str | dict[str],
                           size_group: "Optional[str | SizeGroup]" = None) -> gpd.GeoDataFrame:

        metadata = self._resolve_metadata(None, size_group)
        return metadata.query(query)


    def get_frames_for_THZ(self,
                            target: str | THZClass,
                            split: bool = True,
                            size_group: "Optional[str | SizeGroup]" = None) -> gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame]:
        """
        Gets the patches whose THZ class fall into a certain category(currently only "temperate" or "subtropic").
        Can either return the raw metatadta, or split it into train, test, and valid splits.
        """
        if isinstance(target, THZClass):
            target = target.value
        query = THZ_CLASS_QUERY.format(
            cls=f"'{target.lower()}'"
        )

        if split:
            retval = self.get_split_metadata(
                query=query,
                size_group=size_group,
            )
            return retval
        else:
            return self.get_metadata_queried(
                query=query,
                size_group=size_group,
            )

    def get_geographic_distance_frames(self,
                                       far_group_ind: int,
                                       split=True,
                                       size_group: "Optional[str | SizeGroup]" = None) -> dict[str,gpd.GeoDataFrame]:
        """
        Given an index representing a far tilegroup, returns a mapping of split to a metadata GeoDataFrame
        containing all of the patches EXCEPT those that fall into that tilegroup. Also returns a test_far
        GeoDataFrame that contains ONLY the test patches that fall into that tilegroup.
        """
        query = GEOGRAPHIC_DISTANCE_QUERY.format(
            ind=far_group_ind
        )

        if split:
            result = self.get_split_metadata(
                query=query,
                size_group=size_group,
            )
            result["valid_far"] = self.get_single_split(
                split="valid",
                size_group=size_group,
            ).query("not " + query)

            result["train_far"] = self.get_single_split(
                split="train",
                size_group=size_group,
            ).query("not " + query)
            
            result["test_far"] = self.get_single_split(
                split="test",
                size_group=size_group,
            ).query("not " + query)

            return result

        else:
            return self.get_metadata_queried(
                query=query,
                size_group=size_group,
            )
        
    def downsample_all_splits(self,
                              splits_left: dict[str,gpd.GeoDataFrame],
                              splits_right: dict[str,gpd.GeoDataFrame]) -> dict[str,gpd.GeoDataFrame]:
        """
        Given two mappings of split to dataframe, matches the left and right dataframes
        by their shared split, and then downsamples left to the size of right.

        Returns a mapping of split to downsampled left dataframe.a
        """
        if splits_left.keys() != splits_right.keys():
            raise ValueError("Both left and right must have the same splits.")
        
        result = {}

        for k in splits_left:
            result[k] = self.downsample_to_other(
                left=splits_left[k],
                right=splits_right[k]
            )
        
        return result



    def downsample_to_other(self,
                            left: gpd.GeoDataFrame,
                            right: gpd.GeoDataFrame):
        """
        Randomly downsamples the left dataframe to the size of the right dataframe. If the left dataframe is
        smaller than the right dataframe or both are the same size, the left dataframe is returned unaltered.
        """
        len_left = len(left)
        len_right = len(right)

        if len_left <= len_right:
            return left
        
        left = left.sample(len_right, random_state=RANDOM_SEED)

        return left

    def get_all_THZ_experiment_frames(self,
                                  thz_groups=None,
                                  size_group: "Optional[str | SizeGroup]" = None) -> dict[str,dict[str,gpd.GeoDataFrame]]:
        """
        Provides the metadata frames needed to run the THZ distance experiments. The code creates 3 dataframes:
        temperate-only, subtropic-only, and both temperate and subtropic. It then finds which of these 3 is the smallest,
        and automatically downsamples the remaining 2 frames to the appropriate size.

        The return value is a nested dictionary where the outer dictionary contains string keys representing individual
        experiments, and the inner dictionary contains string keys mapping splits to dataframes. For example,
        experiment_frames["temperate"]["train"] would map to a GeoDataFrame containing metadata for all the
        train patches in the temperate-only experiment.
        """
        if not thz_groups:
            thz_groups = ["temperate", "subtropic"]

        experiment_frames = {}

        for group in thz_groups:
            experiment_frames[group] = self.get_frames_for_THZ(
                target=group,
                split=True,
                size_group=size_group,
            )

        experiment_frames["all_thz_classes"] = self.get_split_metadata(size_group=size_group)

        for split in ["train", "valid"]:
            min_key = min(experiment_frames, key=lambda k: len(experiment_frames[k][split]))
            for k in experiment_frames:
                experiment_frames[k][split] = self.downsample_to_other(
                    left=experiment_frames[k][split],
                    right=experiment_frames[min_key][split]
                )

        return experiment_frames



    def get_geographic_distance_experiment_frames(
            self,
            n_tilegroups: int=NUM_TILEGROUPS,
            size_group: "Optional[str | SizeGroup]" = None)-> dict[str, dict[str,gpd.GeoDataFrame]]:
        experiment_frames = {}

        for i in range(n_tilegroups):
            experiment_frames[f"far_group_{i}"] = self.get_geographic_distance_frames(
                far_group_ind=i,
                size_group=size_group,
            )
        return experiment_frames
    

