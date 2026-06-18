import numpy as np
from sklearn.linear_model import LogisticRegression

from hedgementation_utils.training.metadata_library import MetadataLibrary

from src.embedding_baseline_models.common import (
    eval_on_df,
    flatten_tile,
    load_tile_xy,
    load_xy_metadata_frame,
    try_load_tile_xy,
)


LOGREG_PARAMS = {
    "feature_mode": "patch3x3",
    "center_weight": 3.0,
    "max_pixels_per_tile": 1000,
    "C": 1.0,
    "pred_threshold": 0.8,
    "class_weight": None,
    "balance_pos_neg": True,
    "y_threshold": 0.0,
    "seed": 15,
    "max_iter": 1000,
}


def build_train_pool(
    df_train,
    y_threshold=0.0,
    max_pixels_per_tile=1000,
    seed=15,
    balance_pos_neg=True,
    feature_mode="patch3x3",
    center_weight=1.0,
    skip_bad_tiles=True,
):
    rng = np.random.default_rng(seed)
    x_parts = []
    y_parts = []

    for _, row in df_train.iterrows():
        if skip_bad_tiles:
            loaded, _ = try_load_tile_xy(row, y_threshold=y_threshold)
            if loaded is None:
                continue
            x_tile, y_tile = loaded
        else:
            x_tile, y_tile = load_tile_xy(row, y_threshold=y_threshold)
        x_flat, y_flat, _, _ = flatten_tile(
            x_tile,
            y_tile,
            feature_mode=feature_mode,
            center_weight=center_weight,
        )
        if max_pixels_per_tile is not None:
            take = min(int(max_pixels_per_tile), len(y_flat))
            keep = rng.choice(len(y_flat), size=take, replace=False)
            x_flat = x_flat[keep]
            y_flat = y_flat[keep]
        x_parts.append(x_flat)
        y_parts.append(y_flat)

    if not x_parts:
        raise ValueError("No sampled training pixels")

    x_train = np.vstack(x_parts)
    y_train = np.concatenate(y_parts)

    if balance_pos_neg:
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            raise ValueError("Cannot balance classes because one class is empty")
        k = min(len(pos_idx), len(neg_idx))
        keep_pos = rng.choice(pos_idx, size=k, replace=False)
        keep_neg = rng.choice(neg_idx, size=k, replace=False)
        keep = np.concatenate([keep_pos, keep_neg])
        rng.shuffle(keep)
        x_train = x_train[keep]
        y_train = y_train[keep]

    return x_train, y_train


def train_one_far_group(df, far_group, params=None):
    params = {**LOGREG_PARAMS, **(params or {})}
    frames = MetadataLibrary(metadata=df).get_geographic_distance_frames(far_group_ind=far_group, split=True)
    df_train = frames["train"].copy()
    df_near = frames["test"].copy()
    df_far = frames["test_far"].copy()

    x_train, y_train = build_train_pool(
        df_train,
        y_threshold=params["y_threshold"],
        max_pixels_per_tile=params["max_pixels_per_tile"],
        seed=params["seed"],
        balance_pos_neg=params["balance_pos_neg"],
        feature_mode=params["feature_mode"],
        center_weight=params["center_weight"],
    )
    estimator = LogisticRegression(
        solver="saga",
        penalty="l2",
        max_iter=params["max_iter"],
        random_state=params["seed"],
        class_weight=params["class_weight"],
        C=params["C"],
        n_jobs=-1,
    )
    estimator.fit(x_train, y_train)

    eval_kwargs = {
        "y_threshold": params["y_threshold"],
        "pred_threshold": params["pred_threshold"],
        "feature_mode": params["feature_mode"],
        "center_weight": params["center_weight"],
    }
    near_metrics = eval_on_df(df_near, estimator, **eval_kwargs)
    far_metrics = eval_on_df(df_far, estimator, **eval_kwargs)
    return estimator, df_train, near_metrics, far_metrics


def load_training_frame(data_paths):
    return load_xy_metadata_frame(data_paths)

