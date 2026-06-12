import itertools
import json
import os
import pathlib
from typing import Dict, List, Optional, Tuple, Union
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import torch
from src.performance_analysis.evaluate_model import get_model_predictions
from hedgementation_utils.metrics.seg_meter import SegMeter
from src.training.dataloader_utils import setup_dataloader
from dotenv import load_dotenv
import os

from hedgementation_utils.training.metadata_library import MetadataLibrary, SizeGroup
from hedgementation_utils.io.io_manager import HedgementationIOManager

from backbones.utae import UTAE

from src.training.train_utils import Backbones, ModelFactory
from src.training.transforms import Normalization, YTransform
from enum import Enum
from collections import namedtuple
from dataclasses import dataclass

load_dotenv()

DATASET_ROOT = os.environ["DATASET_ROOT"]




# Climate classification queries
SUBTROPICS_QUERY = 'thz_class in ["TRC5: Subtropics, cool", "TRC4: Subtropics, moderately cool"]'
TEMPERATE_QUERY = 'thz_class in ["TRC7: Temperate, cool", "TRC6: Temperate, moderately cool"]'

class MaskType(Enum):
    NONE = "none"
    AGRICULTURAL = "agricultural"
    NON_AGRICULTURAL = "non_agricultural"

@dataclass(frozen=True)
class ResultsIdentifier:
    model: str
    dataset: str
    split: str
    mask_type: MaskType = MaskType.NONE

class ModelPerformanceAnalyzer:
    def __init__(self,
                 experiment_names=list[str],
                 metadata_frames=dict,
                 dataset_root:str = DATASET_ROOT,

                 num_classes: int = 2,
                 model_directory: str = "models/",
                 metrics: List[str] = ["precision", "recall", "f1", "iou"],
                 train_folds: Optional[List[int]] = None,
                 valid_folds: Optional[List[int]] = None,
                 test_folds: Optional[List[int]] = None,
                 default_split: str = "test",
                 rpg_mask_dir="rpg_unbuffered",
                 metadata_size_group: SizeGroup = SizeGroup.SMALL,
                 apply_size_group: bool = False):
        """
        Analyzer for climate-based transfer learning experiments with fold support.
        
        Args:
            experiment_names: list of top-level directories containing models for given experiments
            metadata_frames: mapping of experiment metadata to individual splits(i.e. {"temperate_only": {"train": GeoDataFrame(), "valid":GeoDataFrame(), ...}})
            metrics: List of metrics to analyze
            train_folds: List of fold indices for training set (default: [0,1,2])
            valid_folds: List of fold indices for validation set (default: [3])
            test_folds: List of fold indices for test set (default: [4])
            default_split: Which split to use by default ('train', 'valid', 'test', or 'all')
        """

        metadata_frame = gpd.read_file(pathlib.Path(dataset_root, "metadata.geojson"))
        self.dataset_root = dataset_root
        self.lib = MetadataLibrary(metadata=metadata_frame)

        self.manager = HedgementationIOManager(
            dataset_root=dataset_root
        )
        if apply_size_group:
            self.master_metadata = MetadataLibrary._apply_size_group_filter(
                metadata=metadata_frame, 
                size_group=metadata_size_group)
        else:
            self.master_metadata = metadata_frame
        
        self.id_to_index = {
            int(patch_id): idx 
            for idx, patch_id in enumerate(self.master_metadata['ID_PATCH'])
        }

        self.model_directory = model_directory
        self.metrics = metrics
        self.num_classes = num_classes
        
        # Store fold assignments
        self.train_folds = train_folds if train_folds is not None else [0, 1, 2]
        self.valid_folds = valid_folds if valid_folds is not None else [3]
        self.test_folds = test_folds if test_folds is not None else [4]
        self.default_split = default_split
        
        self.experiment_names = experiment_names
        self.metadata_frames = metadata_frames

        

        model_prediction_mapping, ground_truth = self.calc_prediction_mapping(self.experiment_names)
        self.model_prediction_mapping = model_prediction_mapping
        self.ground_truth = ground_truth.reshape((-1,128,128))

        self.model_performance_on_data_subset = {}
        self.rpg_masks = {}
        self.rpg_mask_dir = rpg_mask_dir



    def _get_labels_from_dataloader(self, model_path: str):
        """
        Helper to load ground truth labels by iterating the dataloader 
        without initializing the model or running inference.
        """
        params = json.load(open(os.path.join(model_path, "params.json"), "rb+"))
        
        # Setup dataloader (identical to get_full_dataset_predictions)
        y_transform = YTransform[params["y_transform"].upper()]
        dataloader = setup_dataloader(
            metadata_frame=self.master_metadata,
            data_path=DATASET_ROOT,
            num_buckets=params["num_buckets"],
            image_count=params["image_count"],
            normalization=Normalization[params["normalization"].upper()],
            y_transform=y_transform,
            inclusion_intervals=params["inclusion_intervals"],
            # Use larger batch size for label extraction since we have no model overhead
            batch_size=params["batch_size"] * 4, 
            shuffle=False
        )

        print(f"⚡ Extracting labels from dataloader (skipping inference)...")
        all_labels = []
        
        # Iterate simply to collect labels
        for batch in dataloader:
            # Handle dictionary or tuple batch output
            if isinstance(batch, dict):
                y = batch.get('label', batch.get('y'))
            else:
                _, y = batch # specific to your dataloader return structure
            
            all_labels.append(y)

        return torch.cat(all_labels, dim=0)


    def calculate_performance_on_dfs(self,
                                     model_path: str,
                                     df_dict: dict[str, gpd.GeoDataFrame]):
        gdf_metrics_dict = {}

        for df_name, df in df_dict.items():
            inds = df["ID_PATCH"]
            metrics = self.calc_subset_performance(model_path=model_path, inds=inds)
            gdf_metrics_dict[df_name] = metrics
        
        return gdf_metrics_dict


    def _get_rpg_mask(self,
                       ind: int):
            
        if ind not in self.rpg_masks:
            self.rpg_masks[ind] = self.manager.load_rpg_mask(ind)
        return self.rpg_masks[ind]

    def load_full_rpg_mask(self,
                           inds: list[int],
                           mask_type: MaskType):
        rpg_masks_list = [self._get_rpg_mask(i) for i in inds]
        full_rpg_mask = torch.tensor(np.stack(rpg_masks_list).astype(np.int64))

        if mask_type == MaskType.NON_AGRICULTURAL:
            full_rpg_mask = 1 - full_rpg_mask
        return full_rpg_mask

    def calc_subset_performance(self,
                                model_path: str,
                                inds: list[int],
                                mask_type=None
                                ):
        tensor_indices = torch.tensor([self.id_to_index[int(patch_id)] for patch_id in inds])

        seg_meter = SegMeter(num_classes=self.num_classes)
        preds = self.model_prediction_mapping[model_path]

        

        subset_preds = torch.index_select(preds, 
                                          dim=0,
                                          index=tensor_indices)
    
        subset_ground_truth = torch.index_select(self.ground_truth, 
                                          dim=0,
                                          index=tensor_indices)
        
        if mask_type and mask_type != MaskType.NONE:
            masks = self.load_full_rpg_mask(inds, mask_type)
            subset_preds = subset_preds * masks
            subset_ground_truth = subset_ground_truth * masks
    
        seg_meter.update_batch(subset_preds, subset_ground_truth)
        metrics = seg_meter.ask_metrics()
        return metrics

    def calc_prediction_mapping(self, paths: list[str]):
        prediction_mapping = {}
        ground_truth = None
        
        gt_path = os.path.join(self.model_directory, "ground_truth_cached.pt")
        
        if os.path.exists(gt_path):
            print(f"✓ Loading cached ground truth from {gt_path}")
            ground_truth = torch.load(gt_path)
        else:
            if paths:
                first_model_full_path = os.path.join(self.model_directory, paths[0])
                ground_truth = self._get_labels_from_dataloader(first_model_full_path)
                
                torch.save(ground_truth, gt_path)
                print(f"✓ Saved ground truth cache to {gt_path}")

        for model_path in paths:
            full_path = os.path.join(self.model_directory, model_path)
            pred_cache_path = os.path.join(full_path, "full_dataset_predictions.pt")
            
            if os.path.exists(pred_cache_path):
                print(f"✓ Loading cached predictions for {model_path}")
                try:
                    preds = torch.load(pred_cache_path)
                except:
                    print("Error loading cached data, falling back to inference...")
                    preds, labels = self.get_full_dataset_predictions(full_path)
                
                    if ground_truth is None:
                        ground_truth = labels
                        torch.save(ground_truth, gt_path)
                    
                    torch.save(preds, pred_cache_path)
            else:
                print(f"Running inference for {model_path}...")
                preds, labels = self.get_full_dataset_predictions(full_path)
                
                if ground_truth is None:
                    ground_truth = labels
                    torch.save(ground_truth, gt_path)
                
                torch.save(preds, pred_cache_path)

            prediction_mapping[model_path] = preds
            
        return prediction_mapping, ground_truth
    

    def get_full_dataset_predictions(self,
                                     model_path: str):
        """
        Calculate model predictions for the entire dataset, to be subset
        for later calculations of accuracy on specific things
        """
        params = json.load(open(os.path.join(model_path, "params.json"), "rb+"))
        model_params = torch.load(os.path.join(model_path, "checkpoints", "best_model_params.pt"))
        model = ModelFactory.construct_model(
            backbone=Backbones[params["backbone"].upper()],
            num_buckets=int(params["num_buckets"])
        )
        model.load_state_dict(model_params)
        frame = self.master_metadata
        y_transform = YTransform[params["y_transform"].upper()]
        dataloader = setup_dataloader(
            metadata_frame=frame,
            data_path=DATASET_ROOT,
            num_buckets=params["num_buckets"],
            image_count=params["image_count"],
            normalization=Normalization[params["normalization"].upper()],
            y_transform=y_transform,
            inclusion_intervals=params["inclusion_intervals"],
            batch_size=params["batch_size"],
            shuffle=False
        )
        preds,labels = get_model_predictions(model, 
                                             dataloader=dataloader,
                                             y_transform=y_transform)

        return preds, labels

    def verify_mapping(self):
        """Verify that ID_PATCH mapping is correct"""
        print("Verifying ID_PATCH to index mapping...")
        
        # Check that all patch IDs in metadata are in the mapping
        metadata_ids = set(self.master_metadata['ID_PATCH'])
        mapping_ids = set(self.id_to_index.keys())
        
        if metadata_ids != mapping_ids:
            missing = metadata_ids - mapping_ids
            extra = mapping_ids - metadata_ids
            raise ValueError(f"Mapping mismatch! Missing: {missing}, Extra: {extra}")
        
        # Check that tensor sizes match
        expected_size = len(self.master_metadata)
        actual_size = len(self.ground_truth)
        
        if expected_size != actual_size:
            raise ValueError(f"Size mismatch! Metadata: {expected_size}, Tensor: {actual_size}")
        
        print(f"✓ Mapping verified: {len(self.id_to_index)} patches")
        
        # Spot check: verify a few random patches
        import random
        sample_ids = random.sample(list(self.id_to_index.keys()), min(5, len(self.id_to_index)))
        for patch_id in sample_ids:
            idx = self.id_to_index[patch_id]
            print(f"  Patch {patch_id} -> Index {idx}")
    
    def _load_experiment(self, experiment_name: str) -> Dict:
        """Calculate/load predictions of the model on the full dataset and store them into the top-level prediction mapping for later subsetting."""
        exp_path = os.path.join(self.model_directory, experiment_name)
        full_dataset_predictions_path = os.path.join(exp_path, "full_dataset_predictions.pt")
        if not os.path.exists(full_dataset_predictions_path):
            full_dataset_predictions = self.get_full_dataset_predictions(
                exp_path
            )
            with open(full_dataset_predictions_path, "wb+") as outfile:
                torch.save(full_dataset_predictions, outfile)
        else:
            with open(full_dataset_predictions_path, "rb+") as infile:
                full_dataset_predictions = torch.load(infile)
        self.model_prediction_mapping[experiment_name] = full_dataset_predictions


    def _set_precalculated_metrics(self,
                                   metrics_dict: dict,
                                experiment_name: str,
                                gdf_name: str,
                                split_name: str,
                                mask_type: MaskType = MaskType.NONE):
        results_identifier = ResultsIdentifier(
            model=experiment_name,
            dataset=gdf_name,
            split=split_name,
            mask_type=mask_type
        )
        self.model_performance_on_data_subset[results_identifier] = metrics_dict
    
    def _get_precalculated_metrics(self, 
                                experiment_name: str,
                                gdf_name: str,
                                split_name: Optional[str],
                                mask_type: MaskType = MaskType.NONE):
        results_identifier = ResultsIdentifier(
            model=experiment_name,
            dataset=gdf_name,
            split=split_name,
            mask_type=mask_type
        )
        if results_identifier in self.model_performance_on_data_subset:
            return self.model_performance_on_data_subset[results_identifier]
        return None
        

    def _lazy_load_metrics_for_experiment_on_gdf(self,
                                           experiment_name: str,
                                           gdf_name: str,
                                           split_name: Optional[str],
                                           mask_type: MaskType = MaskType.NONE 
                ):
        """Getter for experiment-gdf metrics with lazy evaluation"""
        if not split_name:
            split_name = self.default_split

        if not self._get_precalculated_metrics(
            experiment_name=experiment_name,
            gdf_name=gdf_name,
            split_name=split_name,
            mask_type=mask_type):
            gdf: gpd.GeoDataFrame = self.metadata_frames[gdf_name][split_name]
            inds = gdf["ID_PATCH"]

            metrics = self.calc_subset_performance(experiment_name, inds, mask_type)

            self._set_precalculated_metrics(
                metrics_dict=metrics,
                experiment_name=experiment_name,
                gdf_name=gdf_name,
                split_name=split_name,
                mask_type=mask_type)

        return self._get_precalculated_metrics(
            experiment_name=experiment_name,
            gdf_name=gdf_name,
            split_name=split_name,
            mask_type=mask_type)
    
    def add_model(self,
                  model_path: str):
        self.experiment_names.append(model_path)
        self._load_experiment(model_path)

    def add_frame(self,
                  frame_name: str, 
                  frame: gpd.GeoDataFrame,):
        self.metadata_frames[frame_name] = frame
    def _get_summary_line(self,
                         experiment,
                         gdf_name, 
                         split,
                         metrics,
                         mask_type: MaskType = MaskType.NONE):
        metrics_dict = self._lazy_load_metrics_for_experiment_on_gdf(experiment, gdf_name, split, mask_type)
        line = [
            experiment, gdf_name, split, *[round(metrics_dict[m],3) for m in metrics], mask_type.value
        ]
        return line
    
    def print_summary(self,
                    experiment_names: list[str],
                    gdf_names: list[str],
                    order_by_experiment=True,
                    splits: list[str] = None,
                    metrics: list[str] = None,
                    mask_type: MaskType = MaskType.NONE
                    ):    

        experiment_gdf_pairs = itertools.product(experiment_names, gdf_names) if order_by_experiment else itertools.product(gdf_names, experiment_names)

        self.print_summary_for_pairs(experiment_gdf_pairs, splits, metrics, mask_type)
      
        

    def print_summary_for_pairs(self,
                                experiment_gdf_pairs,
                                splits: list[str] = None,
                                metrics: list[str] = None,
                                mask_type: MaskType = MaskType.NONE):
        if not splits:
            splits = [self.default_split]
        if not metrics:
            metrics = self.metrics
            
        # 1. Build the data matrix
        header = ["Experiment", "Data", "Split", *metrics, "Mask Type"]
        lines = [header]
        
        # Track the last experiment name to add visual separators between groups
        last_experiment = None
        
        for experiment, gdf in experiment_gdf_pairs:
            # Add a spacer row if the experiment changes (but not before the first one)
            if last_experiment is not None and experiment != last_experiment:
                lines.append([""] * len(header))
            
            for split in splits:
                line = self._get_summary_line(experiment, gdf, split, metrics, mask_type)
                lines.append(line)
            
            last_experiment = experiment

        str_lines = [[str(cell) for cell in row] for row in lines]
        col_widths = [max(len(row[i]) for row in str_lines) for i in range(len(str_lines[0]))]
        
        total_width = sum(col_widths) + (len(col_widths) * 2)
        separator = "-" * total_width

        print(separator)
        
        for i, row in enumerate(str_lines):
            if row[0] == "":
                print("")
                continue

            formatted_row = "  ".join(cell.ljust(width) for cell, width in zip(row, col_widths))
            print(formatted_row)
            
            if i == 0:
                print(separator)

    def _calc_group_statistics(self, 
                               model_paths: list[str], 
                               gdf_name: str, 
                               split_name: str) -> dict:
        """
        Internal helper: Collects metrics for a list of models and returns 
        the mean and standard deviation for each metric.
        """
        aggregated_values = {m: [] for m in self.metrics}
        
        for model in model_paths:
            metrics = self._lazy_load_metrics_for_experiment_on_gdf(model, gdf_name, split_name)
            for m in self.metrics:
                aggregated_values[m].append(metrics[m])

        stats_dict = {}
        for m in self.metrics:
            values = np.array(aggregated_values[m])
            stats_dict[m] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
        return stats_dict        
    
    def _calc_group_statistics(self, 
                               model_paths: list[str], 
                               gdf_name: str, 
                               split_name: str) -> dict:
        """
        Internal helper: Collects metrics for a list of models and returns 
        the mean and standard deviation for each metric.
        """
        # 1. Collect all raw values
        aggregated_values = {m: [] for m in self.metrics}
        
        for model in model_paths:
            # Re-use existing lazy-loading infrastructure
            metrics = self._lazy_load_metrics_for_experiment_on_gdf(model, gdf_name, split_name)
            for m in self.metrics:
                aggregated_values[m].append(metrics[m])

        # 2. Calculate statistics
        stats_dict = {}
        for m in self.metrics:
            values = np.array(aggregated_values[m])
            stats_dict[m] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            
        return stats_dict

    def print_model_comparison(self, 
                               target_model: str, 
                               group_models: list[str], 
                               gdf_names: list[str],
                               split: str = "test",
                               metrics = None):
        """
        Prints a comparison table between a specific target model and the 
        average performance of a group of models (e.g., 5 random seeds).
        """
        if not metrics:
            metrics = self.metrics
        if not split:
            split = self.default_split

        # 1. Build the Header
        lines = [
            ["Model", "GDF", "Split", *metrics]
        ]

        # 2. Iterate through requested DataFrames
        for gdf in gdf_names:
            # --- Row A: The Target Model ---
            target_metrics = self._lazy_load_metrics_for_experiment_on_gdf(target_model, gdf, split)
            target_row = [
                f"Target ({target_model})", 
                gdf, 
                split, 
                *[f"{target_metrics[m]:.4f}" for m in metrics]
            ]
            lines.append(target_row)

            # --- Row B: The Group Mean ---
            group_stats = self._calc_group_statistics(group_models, gdf, split)
            
            # Format: "0.8520 ±0.002"
            formatted_stats = []
            for m in metrics:
                mean = group_stats[m]['mean']
                std = group_stats[m]['std']
                formatted_stats.append(f"{mean:.4f} ±{std:.4f}")

            group_row = [
                f"Group Mean (N={len(group_models)})", 
                gdf, 
                split, 
                *formatted_stats
            ]
            lines.append(group_row)
            
            # Add an empty line for readability between GDF blocks
            lines.append([""] * len(lines[0]))

        # 3. Print using existing alignment logic
        # Remove the very last empty line if it exists
        if lines[-1][0] == "": 
            lines.pop()

        str_lines = [[str(cell) for cell in row] for row in lines]
        col_widths = [max(len(row[i]) for row in str_lines) for i in range(len(str_lines[0]))]
        
        print("-" * (sum(col_widths) + len(col_widths)*2))
        for i, row in enumerate(str_lines):
            print("  ".join(cell.ljust(width) for cell, width in zip(row, col_widths)))
            # Print separator after header
            if i == 0:
                print("-" * (sum(col_widths) + len(col_widths)*2))