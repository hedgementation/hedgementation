import os
import numpy as np
import torch
import geopandas as gpd
import pandas as pd
from sklearn.model_selection import KFold

from src.performance_analysis.density_ratio_analyzer import DensityMapGenerator
from src.training.dataloader_utils import setup_dataloader
from hedgementation_utils.training.metadata_library import MetadataLibrary, SizeGroup
from src.training.experiments_registry import deserialize_label_fn


def run_nuisance_estimation(transfer, model_base_dir, dataset_root, extra_keys=None, num_tilegroups=5):
    if extra_keys is None:
        extra_keys = []

    lib = MetadataLibrary()
    group_idx = transfer["params"]["group_index"]
    # all_groups = lib.get_geographic_distance_frames(group_idx, size_group=SizeGroup.SMALL)
    all_groups = {
        g: lib.get_frames_for_THZ(target=g, split=True, size_group=SizeGroup.SMALL)
        for g in ["temperate", "subtropic"]
    }

    # near_pool = all_groups["train"].copy().sort_values("ID_PATCH")
    near_pool = all_groups[group_idx]["train"].copy().sort_values("ID_PATCH")
    # extra_dfs = {key: all_groups[key].copy() for key in extra_keys}
    extra_dfs = {key: all_groups["subtropic"][key].copy() for key in extra_keys}

    all_splits = list(KFold(n_splits=num_tilegroups, shuffle=True, random_state=15).split(near_pool))

    train_by_patch_id = {}
    final_results = {}
    final_loaders = {}
    params_ref = None

    for fold_idx in range(num_tilegroups):
        print(f"\n--- FOLD {fold_idx}/{num_tilegroups - 1} ---")
        fold_dir = os.path.join(model_base_dir, f"experiment_{fold_idx}")

        _, test_idx  = all_splits[fold_idx]
        near_test_df = near_pool.iloc[test_idx]

        generator = DensityMapGenerator(fold_dir, dataset_root)
        if params_ref is None:
            params_ref = generator.params

        results = generator.generate_all_data({"train": near_test_df, **extra_dfs}, save=False)

        for i, patch_id in enumerate(near_test_df["ID_PATCH"]):
            train_by_patch_id[int(patch_id)] = results["train"][i]

        fold_ids = list(near_test_df["ID_PATCH"].astype(int))
        final_results[f"train_{fold_idx}"] = torch.stack([train_by_patch_id[pid] for pid in fold_ids])
        final_loaders[f"train_{fold_idx}"] = _make_loader(near_test_df, dataset_root, params_ref)

        for key in extra_keys:
            final_results[f"{key}_{fold_idx}"] = results[key]
            if f"{key}_0" not in final_loaders:
                final_loaders[f"{key}_{fold_idx}"] = _make_loader(extra_dfs[key], dataset_root, params_ref)
            else:
                final_loaders[f"{key}_{fold_idx}"] = final_loaders[f"{key}_0"]

    ordered_patch_ids = list(near_pool["ID_PATCH"].astype(int))
    full_train = torch.stack([train_by_patch_id[pid] for pid in ordered_patch_ids])

    torch.save(full_train, os.path.join(model_base_dir, f"near_train_{group_idx}.pt"))
    torch.save(ordered_patch_ids, os.path.join(model_base_dir, f"near_train_{group_idx}_patch_ids.pt"))
    print(f"\nSaved: {len(ordered_patch_ids)} patches.")

    final_results["train"] = full_train
    final_loaders["train"] = _make_loader(near_pool, dataset_root, params_ref)

    final_gen = DensityMapGenerator.__new__(DensityMapGenerator)
    final_gen.model_dir = model_base_dir
    final_gen.dataset_root = dataset_root
    final_gen.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_gen.params = params_ref
    final_gen.model = None
    final_gen.results = final_results
    final_gen.loaders = final_loaders

    return final_gen

def run_geospatial_inference(model_dir, dataset_root, split_dict):
    generator = DensityMapGenerator(model_dir, dataset_root)
    
    generator.generate_all_data(split_dict, save=False, is_regression=True)
    
    return generator

def run_tile_inference(model_dir, dataset_root, metadata_df, split_name="infer", save=False):
    generator = DensityMapGenerator(model_dir, dataset_root)
    generator.generate_tile_data({split_name: metadata_df}, save=False)
    return generator



def _make_loader(df, dataset_root, params):
    transfer = deserialize_label_fn(params["transfer"])
    return setup_dataloader(
        metadata_frame=df,
        data_path=dataset_root,
        num_buckets=params["num_buckets"],
        image_count=params["image_count"],
        normalization=params["normalization"].upper(),
        y_transform=params["y_transform"].upper(),
        inclusion_intervals=params["inclusion_intervals"],
        batch_size=params["batch_size"],
        shuffle=False,
        transfer=transfer, 
        load_X_cloud=True,
        cloud_threshold=params["cloud_threshold"],
        cloud_band=params["cloud_band"], 
    )

def tensor_to_df(tensor, loader):
    N, _, H, W = tensor.shape
    patch_ids = loader.dataset.metadata_frame["ID_PATCH"].values

    data = tensor.permute(0, 2, 3, 1).reshape(-1, 2).numpy()

    lines, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    return pd.DataFrame({
        "index": np.repeat(np.arange(N), H * W),
        "PATCH_ID": np.repeat(patch_ids, H * W),
        "p0": data[:, 0],
        "f0": data[:, 1],
        "line": np.tile(lines.flatten(), N),
        "column": np.tile(cols.flatten(), N),
    })