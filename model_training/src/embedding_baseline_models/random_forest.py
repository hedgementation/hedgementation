import numpy as np
from sklearn.ensemble import RandomForestClassifier

from hedgementation_utils.training.metadata_library import MetadataLibrary

from src.embedding_baseline_models.common import (
    balanced_sample,
    collect_global_pool,
    eval_on_df,
    load_xy_metadata_frame,
)


# A global per-pixel pool (center-pixel features) is balanced-sampled to n_pos/n_neg, 
# RF is fit with class_weight="balanced" and a fixed seed, and tiles are scored by
# predict_proba >= pred_threshold (PROBA_THRESHOLD). feature_mode/center_weight
# are kept only for the runner's feature_config; features are raw per-pixel.
RF_PARAMS = {
    "feature_mode": "pixel",
    "center_weight": 1.0,
    # Model parameters
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    # Sampling (N_POS / N_NEG balanced draw from the global pool)
    "n_pos": 100_000,
    "n_neg": 100_000,
    "max_pixels_per_tile": None,
    "y_threshold": 0.0,
    # PROBA_THRESHOLD (evaluation only)
    "pred_threshold": 0.7,
    "seed": 15,
}


def train_one_far_group(df, far_group, params=None):
    params = {**RF_PARAMS, **(params or {})}
    frames = MetadataLibrary(metadata=df).get_geographic_distance_frames(far_group_ind=far_group, split=True)
    df_train = frames["train"].copy()
    df_near = frames["test"].copy()
    df_far = frames["test_far"].copy()

    rng = np.random.default_rng(params["seed"])
    x_pool, y_pool = collect_global_pool(
        df_train, y_threshold=params["y_threshold"],
        max_pixels_per_tile=params["max_pixels_per_tile"], rng=rng,
    )
    x_train, y_train = balanced_sample(
        x_pool, y_pool, n_pos=params["n_pos"], n_neg=params["n_neg"], rng=rng,
    )
    estimator = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        random_state=params["seed"],
        n_jobs=-1,
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
