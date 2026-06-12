"""
Central registry for all experiments.
To add a new experiment, add an entry to REGISTRY at the bottom of this file.
"""

import os
import re
import pandas as pd
import geopandas as gpd
from dataclasses import dataclass, field
from typing import Callable, Optional
from sklearn.model_selection import KFold
from dotenv import load_dotenv
from hedgementation_utils.training.metadata_library import MetadataLibrary, SizeGroup
import numpy as np
import shapely.wkt
from shapely.geometry.base import BaseGeometry
from src.training.train_utils import LRSchedule

load_dotenv()

DATASET_ROOT = os.environ.get("DATASET_ROOT", "")
VERSION = os.environ.get("VERSION", "1.2")
NUM_TILEGROUPS = int(os.environ.get("NUM_TILEGROUPS", 5))

def nearVSfar_labels(metadata_frame, idx, X, args):
    tile_group = metadata_frame["tile_group"].iloc[idx]
    label = 1.0 if int(tile_group) == int(args["group_index"]) else 0.0
    y = __import__("numpy").full((X.shape[-2], X.shape[-1]), label, dtype=__import__("numpy").int64)
    return y


def temperateVSsubtropic_labels(metadata_frame, idx, X, args):
    thz_class = metadata_frame["thz_class"].iloc[idx]
    label = 0.0 if args["group_index"] in thz_class.lower() else 1.0
    y = __import__("numpy").full((X.shape[-2], X.shape[-1]), label, dtype=__import__("numpy").int64)
    return y

def lat_long_labels(metadata_frame, idx, X, args):
    row = metadata_frame.iloc[idx]
    geom = row.geometry
    if isinstance(geom, str):
        geom = shapely.wkt.loads(geom)
    elif not isinstance(geom, BaseGeometry):
        raise TypeError(
            f"Type de géométrie inattendu à l'index {idx}: {type(geom)}. "
            "Attendu: str (WKT) ou objet géométrique Shapely."
        )
    minx, miny, maxx, maxy = geom.bounds
    
    orig_h, orig_w = X.shape[-2], X.shape[-1]
    
    lats_1d = np.linspace(maxy, miny, orig_h)
    lons_1d = np.linspace(minx, maxx, orig_w)
    
    lons_grid, lats_grid = np.meshgrid(lons_1d, lats_1d)
    
    lat_min, lat_max = args["lat_bounds"]
    lon_min, lon_max = args["lon_bounds"]
    
    lats_grid_scaled = (lats_grid - lat_min) / (lat_max - lat_min)
    lons_grid_scaled = (lons_grid - lon_min) / (lon_max - lon_min)
    
    coords_scaled = np.stack([lats_grid_scaled, lons_grid_scaled], axis=0)
    
    return coords_scaled

def tile_num_labels(metadata_frame, idx, X, args):
    tile_number = metadata_frame["tile"].iloc[idx]
    label = tile_number
    y = __import__("numpy").full((X.shape[-2], X.shape[-1]), label, dtype=__import__("numpy").int64)
    return y

def random_labels(metadata_frame, idx, X, args):
    patch_id = int(metadata_frame["ID_PATCH"].iloc[idx])
    rng = np.random.default_rng(seed=patch_id) 
    label = rng.integers(0, 2)
    return np.full((X.shape[-2], X.shape[-1]), label, dtype=np.int64)

def lat_long_buckets_labels(metadata_frame, idx, X, args):
    row = metadata_frame.iloc[idx]
    geom = row.geometry
    if isinstance(geom, str):
        geom = shapely.wkt.loads(geom)
    elif not isinstance(geom, BaseGeometry):
        raise TypeError("Error reading geometry")
        
    patch_minx, patch_miny, patch_maxx, patch_maxy = geom.bounds
    tile_id_orig = str(row["tile"])
    tile_idx = args["tile_remap"][tile_id_orig]
    
    N = args["subdivisions"]
    H, W = X.shape[-2], X.shape[-1]

    if N == 1:
        return np.full((H, W), tile_idx, dtype=np.int64)
        
    tile_minx, tile_miny, tile_maxx, tile_maxy = args["tile_bbox_index"][str(tile_idx)]
    
    lats = np.linspace(patch_maxy, patch_miny, H)
    lons = np.linspace(patch_minx, patch_maxx, W)
    lons_grid, lats_grid = np.meshgrid(lons, lats)

    tile_width = tile_maxx - tile_minx
    norm_lon = (lons_grid - tile_minx) / tile_width
    
    tile_height = tile_maxy - tile_miny
    norm_lat = (tile_maxy - lats_grid) / tile_height

    col_idx = (norm_lon * N).astype(np.int64)
    row_idx = (norm_lat * N).astype(np.int64)
    
    col_idx = np.clip(col_idx, 0, N - 1)
    row_idx = np.clip(row_idx, 0, N - 1)
    
    sub_tile_idx = row_idx * N + col_idx  
    
    y = tile_idx * (N * N) + sub_tile_idx

    return y.astype(np.int64)

LABEL_FN_REGISTRY = {
    "nearVSfar_labels": nearVSfar_labels,
    "temperateVSsubtropic_labels": temperateVSsubtropic_labels,
    "lat_long_labels": lat_long_labels,
    "tile_num_labels": tile_num_labels,
    "random_labels": random_labels,
    "lat_long_buckets_labels": lat_long_buckets_labels,
}

def deserialize_label_fn(transfer: dict) -> dict:
    if transfer is None:
        return transfer
    if not transfer.get("transfer", True):
        func = transfer.get("function")
        if isinstance(func, str):
            if func not in LABEL_FN_REGISTRY:
                raise ValueError(
                    f"Unknown label function '{func}'. "
                    f"Available: {list(LABEL_FN_REGISTRY.keys())}"
                )
            transfer = {**transfer, "function": LABEL_FN_REGISTRY[func]}
    return transfer


@dataclass
class ExperimentDefinition:
    """
    Defines everything needed to run an experiment.

    Attributes:
        config_factory:             fn(override) -> dict  (fed to TrainerConfig)
        multi_experiment_factory:   fn(nb_exp) -> dict[int, override_dict]
                                    Returns {0: override} for single experiments.
        is_multi:                   Whether this keyword spawns N sub-experiments.
        label_fn:                   Optional label function (for transfer experiments).
    """
    config_factory: Callable
    multi_experiment_factory: Callable

def _base_config():
    return {
        "backbone": "UTAE",
        "normalization": "MINMAX",
        "y_transform": "UNIFORM",
        "loss_function": "CrossEntropyLoss",
        "num_buckets": 1,
        "class_weighted": True,
        "class_weights": [1.0, 3.0],
        "num_epochs": 100,
        "image_count": 10,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "lr_scheduler_patience": 15,
        "early_stopping_patience": 100,
        "early_stopping_min_delta": 0.001,
        "batch_size": 16,
        "augmentation": ["hflip", "vflip", "rotation"],
        "save_results": True,
        "overwrite": False,
        "validation_metric": "iou",
    }


def _get_metadata_library():
    return MetadataLibrary()

def _config_default(override):
    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_base",
        "metadata_frames": _get_metadata_library().get_split_metadata(size_group=SizeGroup.SMALL),
    })
    if override:
        cfg.update(override)
    return cfg


def _config_train_temperate(override):
    cfg = _base_config()
    experiment = _get_metadata_library().get_all_THZ_experiment_frames(size_group=SizeGroup.SMALL)
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_train_temperate",
        "metadata_frames": experiment["temperate"],
    })
    if override:
        cfg.update(override)
    return cfg


def _config_train_subtropic(override):
    cfg = _base_config()
    experiment = _get_metadata_library().get_all_THZ_experiment_frames(size_group=SizeGroup.SMALL)
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_train_subtropic",
        "metadata_frames": experiment["subtropic"],
    })
    if override:
        cfg.update(override)
    return cfg


def _config_downsample_to_subtropic(override):
    cfg = _base_config()
    experiment = _get_metadata_library().get_all_THZ_experiment_frames(size_group=SizeGroup.SMALL)
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_downsample_to_subtropic",
        "metadata_frames": experiment["all_thz_classes"],
    })
    if override:
        cfg.update(override)
    return cfg


def _config_far_group(ind, override):
    cfg = _base_config()
    dataset = _get_metadata_library().get_geographic_distance_frames(ind, size_group=SizeGroup.SMALL)
    del dataset["train_far"]
    del dataset["valid_far"]
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_far_group_{ind}",
        "metadata_frames": dataset,
    })
    if override:
        cfg.update(override)
    return cfg


def _config_agriculture_mask(mask_path, keyword_suffix, override):
    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_{keyword_suffix}",
        "metadata_frames": _get_metadata_library().get_split_metadata(size_group=SizeGroup.SMALL),
        "provide_agriculture_mask": True,
        "agriculture_mask_path": mask_path,
    })
    if override:
        cfg.update(override)
    return cfg


def _config_nuisance_estimation(override):
    if override is None or "transfer" not in override:
        raise ValueError("nuisance_estimation needs 'transfer' in override")

    override["transfer"].update({"transfer": False})
    far_idx = override["transfer"]["params"]["group_index"]
    fold_idx = override["transfer"]["params"]["f0_fold_index"]

    all_groups = _get_metadata_library().get_geographic_distance_frames(far_idx, size_group=SizeGroup.SMALL)

    near_pool = all_groups["train"].copy()
    near_pool = near_pool.sort_values("ID_PATCH").reset_index(drop=True)
    near_valid = all_groups["valid"].copy()
    near_test  = all_groups["test"].copy()
    far_train_before_sampling  = all_groups["train_far"].copy()
    far_train_valid = far_train_before_sampling.sample(frac=1/5, random_state=15)
    far_train = far_train_before_sampling.drop(far_train_valid.index)

    kf = KFold(n_splits=NUM_TILEGROUPS, shuffle=True, random_state=15)
    all_splits = list(kf.split(near_pool))

    test_idx  = all_splits[fold_idx][1]
    train_idx = list(set(range(len(near_pool))) - set(test_idx))

    train_fold = pd.concat([near_pool.iloc[train_idx], near_test, far_train])

    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_nuisance_estimation_{fold_idx}",
        "metadata_frames": {"train": train_fold, "valid": pd.concat([near_valid, far_train_valid])},
        "class_weighted": True,
        "class_weights": False,
        "validation_metric": "loss",
        "y_transform": "none",
        "lr_scheduler": LRSchedule.COSINE_ANNEALING,
        "lr_scheduler_args": {"T_max": 100, "eta_min": 1e-6},
        "early_stopping_min_delta": 0.001,
        "early_stopping_patience": 50,
        "weight_decay": 1.e-3,
        "additional_model_params": {"out_conv_last_relu": False},
    })
    cfg.update(override)
    return cfg


def _config_nuisance_thz(word, override):
    if override is None or "transfer" not in override:
        raise ValueError("nuisance_estimation_thz needs 'transfer' in override")

    other = "subtropic" if word == "temperate" else "temperate"
    override["transfer"].update({"transfer": False})
    fold_idx = override["transfer"]["params"]["f0_fold_index"]

    lib = _get_metadata_library()
    all_groups = {
        g: lib.get_frames_for_THZ(target=g, split=True, size_group=SizeGroup.SMALL)
        for g in ["temperate", "subtropic"]
    }

    near_pool = all_groups[word]["train"].copy().sort_values("ID_PATCH").reset_index(drop=True)
    near_valid = all_groups[word]["valid"].copy()
    near_test  = all_groups[word]["test"].copy()
    far_train_before_sampling  = all_groups[other]["train"].copy()
    far_train_valid = far_train_before_sampling.sample(frac=1/5, random_state=15)
    far_train = far_train_before_sampling.drop(far_train_valid.index)

    kf = KFold(n_splits=NUM_TILEGROUPS, shuffle=True, random_state=15)
    all_splits = list(kf.split(near_pool))

    test_idx  = all_splits[fold_idx][1]
    train_idx = list(set(range(len(near_pool))) - set(test_idx))

    train_fold = pd.concat([near_pool.iloc[train_idx], near_test, far_train])

    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_nuisance_estimation_{word}",
        "metadata_frames": {"train": train_fold, "valid": pd.concat([near_valid, far_train_valid])},
        "class_weighted": True,
        "class_weights": False,
        "validation_metric": "iou",
        "y_transform": "none",
        "lr_scheduler": LRSchedule.COSINE_ANNEALING,
        "lr_scheduler_args": {"T_max": 100, "eta_min": 1e-6},
        "early_stopping_min_delta": 0.005,
        "early_stopping_patience": 15,
        "weight_decay": 1.e-3,
        "additional_model_params": {"out_conv_last_relu": False},
    })
    cfg.update(override)
    return cfg


def _config_model_transfer(override):
    if override is None:
        override = {}
    override.setdefault("transfer", {})
    override["transfer"].update({"transfer": True})
    far_idx = override["transfer"]["params"]["group_index"]
    all_groups = _get_metadata_library().get_geographic_distance_frames(far_idx, size_group=SizeGroup.SMALL)

    train_pool = all_groups["train"].copy()
    valid_pool = all_groups["valid_far"].copy()
    test_pool = all_groups["test_far"].copy()

    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_base",
        "metadata_frames": {"train": train_pool, "valid": valid_pool, "test": test_pool},
    })
    cfg.update(override)
    return cfg

def _config_model_transfer_hyperparameter_tuning(override):
    if override is None:
        override = {}
    override.setdefault("transfer", {})
    override["transfer"].update({"transfer": True})
    far_idx = override["transfer"]["params"]["group_index"]
    hyperparameter_name = override["transfer"]["hyperparameter_name"]
    hyperparameter_list = override["transfer"]["hyperparameter_list"]
    exp_nbr = override["transfer"]["exp_nbr"]
    override[hyperparameter_name] = hyperparameter_list[exp_nbr]

    all_groups = _get_metadata_library().get_geographic_distance_frames(far_idx, size_group=SizeGroup.SMALL)

    train_pool = all_groups["train"].copy()
    valid_pool = all_groups["valid_far"].copy()
    test_pool = all_groups["test_far"].copy()

    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_base",
        "metadata_frames": {"train": train_pool, "valid": valid_pool, "test": test_pool},
        "num_epochs": 30, 
    })
    cfg.update(override)
    return cfg

def _config_lat_long(override):
    if override is None or "transfer" not in override:
        raise ValueError("lat_long needs 'transfer' in override")

    override["transfer"].update({"transfer": False})

    metadta_frames = MetadataLibrary().get_split_metadata(size_group=SizeGroup.SMALL)

    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_base",
        "metadata_frames": metadta_frames,
        "class_weighted": False,
        "class_weights": False,
        "y_transform": "none",
        "loss_function": "MSELoss",
        "regression": True,
        "validation_metric": "mae",
        "augmentation": None,
        "additional_model_params": {"out_conv_last_relu": False}
    })
    cfg.update(override)
    return cfg

def _config_tile_nbr(override):
    if override is None or "transfer" not in override:
        raise ValueError("tile_number needs 'transfer' in override")

    override["transfer"].update({"transfer": False})

    metadta_frames = MetadataLibrary().get_split_metadata(size_group=SizeGroup.SMALL)

    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_base",
        "metadata_frames": metadta_frames,
        "class_weighted": False,
        "class_weights": False,
        "validation_metric": "loss",
        "num_buckets": 35,
        "y_transform": "none",
        "additional_model_params": {"out_conv_last_relu": False, "out_conv": [32, 36]}
    })
    cfg.update(override)
    return cfg

def _config_lat_long_classification(override):
    if override is None or "transfer" not in override:
        raise ValueError("tile_number needs 'transfer' in override")

    override["transfer"].update({"transfer": False})

    metadata_frames = MetadataLibrary().get_split_metadata(size_group=SizeGroup.SMALL)

    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_lat_long_classification_{override['transfer']['params']['num_classes']}_classes",
        "metadata_frames": metadata_frames,
        "class_weighted": False,
        "class_weights": False,
        "validation_metric": "loss",
        "num_buckets": override["transfer"]["params"]["num_classes"],
        "y_transform": "none",
        "additional_model_params": {"out_conv_last_relu": False, "out_conv": [32, override["transfer"]["params"]["num_classes"]+1]}
    })
    cfg.update(override)
    return cfg

def _config_random_labels(override):
    if override is None or "transfer" not in override:
        raise ValueError("random_labels needs 'transfer' in override")

    override["transfer"].update({"transfer": False})
    fold_idx = override["transfer"]["params"]["f0_fold_index"]

    all_groups = _get_metadata_library().get_split_metadata(size_group=SizeGroup.SMALL)

    train = all_groups["train"].sort_values("ID_PATCH").reset_index(drop=True)
    valid = all_groups["valid"].copy()
    test  = all_groups["test"].copy()

    def split_near_far(df):
        labels = df["ID_PATCH"].apply(
            lambda x: np.random.default_rng(seed=x).integers(0, 2)
        )

        near = df[labels == 0].reset_index(drop=True)
        far = df[labels == 1].reset_index(drop=True)

        return near, far
    
    near_pool, far_train = split_near_far(train)
    near_valid, _ = split_near_far(valid)
    near_test, _ = split_near_far(test)

    kf = KFold(n_splits=NUM_TILEGROUPS, shuffle=True, random_state=15)
    all_splits = list(kf.split(near_pool))

    test_idx  = all_splits[fold_idx][1]
    train_idx = list(set(range(len(near_pool))) - set(test_idx))

    train_fold = pd.concat([near_pool.iloc[train_idx], near_test, far_train])

    cfg = _base_config()
    cfg.update({
        "keyword": f"UTAE_hedge_{VERSION}_nuisance_estimation",
        "metadata_frames": {"train": train_fold, "valid": near_valid},
        "class_weighted": True,
        "class_weights": False,
        "validation_metric": "loss",
        "y_transform": "none",
        "additional_model_params": {"out_conv_last_relu": False}
    })
    cfg.update(override)
    return cfg

def _single(override_template):
    return lambda nb_exp: {0: override_template}


def _nuisance_multi_factory(label_fn, group_index):
    def factory(nb_exp):
        return {
            fold_idx: {
                "transfer": {
                    "transfer": False,
                    "function": label_fn,
                    "params": {
                        "group_index": group_index,
                        "f0_fold_index": fold_idx,
                    },
                }
            }
            for fold_idx in range(nb_exp)
        }
    return factory

def _model_transfer_hyperparameter_tuning_multi_factory(hyperpameter_name, hyperparameter_list):
    def factory(nb_exp):
        return {
            exp_nbr: {"transfer": {
                "transfer": True,
                "params": {
                    "ids_path": os.path.join(DATASET_ROOT, "nuisance_weights", "near_train_0_patch_ids.pt"),
                    "f0_path":  os.path.join(DATASET_ROOT, "nuisance_weights", "near_train_0.pt"),
                    "group_index": 0, 
                    "clipping": 19,
                    "hyperparameter_name": hyperpameter_name,
                    "hyperparameter_list": hyperparameter_list,
                    "exp_nbr": exp_nbr
                }
            }}
            for exp_nbr in range(len(hyperparameter_list)) 
        }

def _lat_long_factory():
    lib = MetadataLibrary()
    train_df = lib.get_single_split(split="train", size_group=SizeGroup.SMALL)
    train_lambert = train_df.to_crs(epsg=2154)
    train_wgs84 = train_lambert.to_crs(epsg=4326)
    total_bounds = train_wgs84.total_bounds
    lon_min, lat_min, lon_max, lat_max = total_bounds

    return _single({
        "transfer": {
            "transfer": False,
            "function": lat_long_labels,
            "params": {
                "lat_bounds": (lat_min, lat_max),
                "lon_bounds": (lon_min, lon_max)
            }
        }}
    )


def _tile_num_factory():
    return _single({
        "transfer": {
            "transfer": False,
            "function": tile_num_labels,
            "params": {}
        }}
    )

def _lat_long_classification_factory():
    SUBDIVISIONS = 1

    metadata = gpd.read_file(os.path.join(DATASET_ROOT, "metadata.geojson"))

    distinct_tiles = sorted(metadata["tile"].unique().tolist())
    tile_remap = {tid: i for i, tid in enumerate(distinct_tiles)}

    tile_bbox_index = {}
    for tile_id_orig, group in metadata.groupby("tile"):
        tile_idx = tile_remap[tile_id_orig]
        tile_bbox_index[tile_idx] = group.geometry.unary_union.bounds

    return _single({
        "transfer": {
            "transfer": False,
            "function": lat_long_buckets_labels,
            "params": {
                "tile_remap": tile_remap,
                "tile_bbox_index": tile_bbox_index,
                "num_classes": len(distinct_tiles) * (SUBDIVISIONS ** 2),
                "subdivisions": SUBDIVISIONS,
            }
        }
    })

# ============================================================
# REGISTRY
# ============================================================
# To add a new experiment:
#   1. Write a config_factory function above
#   2. Write a multi_experiment_factory (or use _single() for single experiments)
#   3. Add an ExperimentDefinition entry belo

REGISTRY: dict[str, ExperimentDefinition] = {
    "default": ExperimentDefinition(
        config_factory=_config_default,
        multi_experiment_factory=_single({}),
    ),
    "train_temperate": ExperimentDefinition(
        config_factory=_config_train_temperate,
        multi_experiment_factory=_single({}),
    ),
    "train_subtropic": ExperimentDefinition(
        config_factory=_config_train_subtropic,
        multi_experiment_factory=_single({}),
    ),
    "downsample_to_subtropic": ExperimentDefinition(
        config_factory=_config_downsample_to_subtropic,
        multi_experiment_factory=_single({}),
    ),
    **{
        f"far_group_{i}": ExperimentDefinition(
            config_factory=lambda override, i=i: _config_far_group(i, override),
            multi_experiment_factory=_single({}),
        )
        for i in range(NUM_TILEGROUPS)
    },
    "agriculture_mask_unbuffered": ExperimentDefinition(
        config_factory=lambda override: _config_agriculture_mask(
            "rpg_masks_unbuffered", "agriculture_mask_unbuffered", override
        ),
        multi_experiment_factory=_single({}),
    ),
    "agriculture_mask_buffered_10m": ExperimentDefinition(
        config_factory=lambda override: _config_agriculture_mask(
            "rpg_masks_buffered_10m", "agriculture_mask_buffered_10m", override
        ),
        multi_experiment_factory=_single({}),
    ),
    "agriculture_mask_unbuffered_inverted": ExperimentDefinition(
        config_factory=lambda override: _config_agriculture_mask(
            "rpg_masks_unbuffered_inverted", "agriculture_mask_unbuffered_inverted", override
        ),
        multi_experiment_factory=_single({}),
    ),
    "agriculture_mask_buffered_10m_inverted": ExperimentDefinition(
        config_factory=lambda override: _config_agriculture_mask(
            "rpg_masks_buffered_10m_inverted", "agriculture_mask_buffered_10m_inverted", override
        ),
        multi_experiment_factory=_single({}),
    ),
    "nuisance_estimation": ExperimentDefinition(
        config_factory=_config_nuisance_estimation,
        multi_experiment_factory=_nuisance_multi_factory(nearVSfar_labels, group_index=0),
    ),
    "nuisance_estimation_temperate": ExperimentDefinition(
        config_factory=lambda override: _config_nuisance_thz("temperate", override),
        multi_experiment_factory=_nuisance_multi_factory(temperateVSsubtropic_labels, group_index="temperate"),
    ),
    "nuisance_estimation_subtropic": ExperimentDefinition(
        config_factory=lambda override: _config_nuisance_thz("subtropic", override),
        multi_experiment_factory=_nuisance_multi_factory(temperateVSsubtropic_labels, group_index="subtropic"),
    ),
    "model_transfer": ExperimentDefinition(
        config_factory=_config_model_transfer,
        multi_experiment_factory=_single(
            {"transfer": {
                "transfer": True,
                "params": {
                    "ids_path": os.path.join(DATASET_ROOT, "nuisance_weights", "near_train_0_patch_ids.pt"),
                    "f0_path":  os.path.join(DATASET_ROOT, "nuisance_weights", "near_train_0.pt"),
                    "group_index": 0, 
                    "clipping": 19,
                }
            }}
        ),
    ),
    "model_transfer_tuning": ExperimentDefinition(
        config_factory=_config_model_transfer_hyperparameter_tuning,
        multi_experiment_factory=_model_transfer_hyperparameter_tuning_multi_factory(hyperpameter_name="lr", hyperparameter_list=[1.e-3, 5.e-3, 1.e-2, 5.e-2]),
    ),
    "lat_long": ExperimentDefinition(
        config_factory=_config_lat_long,
        multi_experiment_factory=_lat_long_factory(),
    ),
    "lat_long_classification": ExperimentDefinition(
        config_factory=_config_lat_long_classification,
        multi_experiment_factory=_lat_long_classification_factory(),
    ),
    "tile_number": ExperimentDefinition(
        config_factory=_config_tile_nbr,
        multi_experiment_factory=_tile_num_factory(),
    ),
    "random_labels": ExperimentDefinition(
        config_factory=_config_random_labels,
        multi_experiment_factory=_nuisance_multi_factory(random_labels, group_index=0),
    ),
}