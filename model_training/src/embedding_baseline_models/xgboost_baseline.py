import numpy as np
from xgboost import XGBClassifier

from hedgementation_utils.training.metadata_library import MetadataLibrary

from src.embedding_baseline_models.common import (
    collect_global_pool,
    eval_on_df,
    load_xy_metadata_frame,
)


def _detect_device():
    """Return 'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    try:
        import subprocess
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        return "cuda"
    except Exception:
        pass
    return "cpu"


# XGBoost trains on the
# *entire* global per-pixel pool (no balanced sub-sample), and class imbalance
# is handled by scale_pos_weight = n_neg / n_pos computed over that full pool
# (scale_pos_weight=None => auto). Tiles are scored by predict_proba >=
# pred_threshold (PROBA_THRESHOLD). feature_mode/center_weight are kept only for
# the runner's feature_config; features are raw per-pixel.
XGB_PARAMS = {
    "feature_mode": "pixel",
    "center_weight": 1.0,
    # Model parameters
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "gamma": 0.0,
    "tree_method": "hist",
    "device": _detect_device(),
    "scale_pos_weight": None,  # None = auto-compute n_neg / n_pos over the full pool
    # Sampling (full pool, optional per-tile cap)
    "max_pixels_per_tile": None,
    "y_threshold": 0.0,
    # PROBA_THRESHOLD (evaluation only)
    "pred_threshold": 0.5,
    "seed": 15,
}


def train_one_far_group(df, far_group, params=None):
    params = {**XGB_PARAMS, **(params or {})}
    frames = MetadataLibrary(metadata=df).get_geographic_distance_frames(far_group_ind=far_group, split=True)
    df_train = frames["train"].copy()
    df_near = frames["test"].copy()
    df_far = frames["test_far"].copy()

    rng = np.random.default_rng(params["seed"])
    x_train, y_pool = collect_global_pool(
        df_train, y_threshold=params["y_threshold"],
        max_pixels_per_tile=params["max_pixels_per_tile"], rng=rng,
    )
    y_train = y_pool.astype(int)

    scale_pos_weight = params["scale_pos_weight"]
    if scale_pos_weight is None:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    estimator = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],
        tree_method=params["tree_method"],
        device=params["device"],
        scale_pos_weight=scale_pos_weight,
        random_state=params["seed"],
        n_jobs=-1,
        eval_metric="logloss",
        verbosity=0,
    )
    estimator.fit(x_train, y_train)

    eval_kwargs = {
        "y_threshold": params["y_threshold"],
        "pred_threshold": params["pred_threshold"],
        "feature_mode": "pixel",
        "center_weight": params["center_weight"],
    }
    near_metrics = eval_on_df(df_near, estimator, **eval_kwargs)
    far_metrics = eval_on_df(df_far, estimator, **eval_kwargs)
    return estimator, df_train, near_metrics, far_metrics


def load_training_frame(data_paths):
    return load_xy_metadata_frame(data_paths)
