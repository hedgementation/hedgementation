import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from hedgementation_utils.training.metadata_library import MetadataLibrary

from src.embedding_baseline_models.common import (
    compute_metrics,
    load_xy_metadata_frame,
    try_load_tile_xy,
)


OFFSETS_3X3 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 0), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

KNN_PARAMS = {
    "feature_mode": "concat",
    "center_weight": 6.0,
    "n_neighbors": 3,
    "weights": "distance",
    "n_pos": 100_000,
    "n_neg": 100_000,
    "tile_pos_cap": 100,
    "tile_neg_cap": 100,
    "pos_inner_frac": 1.0,
    "neg_hard_frac": 0.0,
    "pos_neg_ratio": 1.0,
    "dilation": 1,
    "pad_mode": "reflect",
    "pred_chunk_size": 200_000,
    "y_threshold": 0.0,
    "seed": 15,
}


def neigh_sum_3x3(y_bin):
    y = (y_bin > 0).astype(np.uint8)
    height, width = y.shape
    y_pad = np.pad(y, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    out = np.zeros((height, width), dtype=np.uint8)
    for dy, dx in OFFSETS_3X3:
        out += y_pad[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
    return out


def make_offsets_3x3(dilation=1):
    d = int(dilation)
    return [
        (-d, -d), (-d, 0), (-d, d),
        (0, -d), (0, 0), (0, d),
        (d, -d), (d, 0), (d, d),
    ]


def extract_center_features_at_coords(x_tile, ys, xs):
    return x_tile[:, ys.astype(np.int64), xs.astype(np.int64)].T.astype(np.float32)


def extract_patch_features_at_coords(x_tile, ys, xs, pad_mode="reflect", patch_weights=None, dilation=1):
    offsets = make_offsets_3x3(dilation)
    x_hw_c = np.moveaxis(x_tile, 0, -1)
    pad = int(dilation)
    x_pad = np.pad(x_hw_c, ((pad, pad), (pad, pad), (0, 0)), mode=pad_mode)
    ys = ys.astype(np.int64) + pad
    xs = xs.astype(np.int64) + pad
    if patch_weights is None:
        patch_weights = np.ones((9,), dtype=np.float32)
    else:
        patch_weights = np.asarray(patch_weights, dtype=np.float32)
        if patch_weights.shape != (9,):
            raise ValueError("patch_weights must have shape (9,)")
    cols = []
    for idx, (dy, dx) in enumerate(offsets):
        cols.append(x_pad[ys + dy, xs + dx, :] * patch_weights[idx])
    return np.concatenate(cols, axis=1).astype(np.float32)


def extract_patch_mean_features_at_coords(x_tile, ys, xs, pad_mode="reflect", center_weight=1.0, dilation=1):
    channels = x_tile.shape[0]
    patch = extract_patch_features_at_coords(
        x_tile, ys, xs, pad_mode=pad_mode, patch_weights=None, dilation=dilation
    ).reshape(len(ys), 9, channels)
    weights = np.ones((9,), dtype=np.float32)
    weights[4] = float(center_weight)
    weights = weights / weights.sum()
    return (patch * weights[None, :, None]).sum(axis=1).astype(np.float32)


def extract_features_at_coords(
    x_tile, ys, xs, feature_mode="concat", pad_mode="reflect", center_weight=1.0,
    patch_weights=None, dilation=1,
):
    if feature_mode == "center":
        return extract_center_features_at_coords(x_tile, ys, xs)
    if feature_mode == "mean":
        return extract_patch_mean_features_at_coords(
            x_tile, ys, xs, pad_mode=pad_mode, center_weight=center_weight, dilation=dilation
        )
    if feature_mode == "concat":
        if patch_weights is None and center_weight != 1.0:
            patch_weights = np.ones((9,), dtype=np.float32)
            patch_weights[4] = float(center_weight)
        return extract_patch_features_at_coords(
            x_tile, ys, xs, pad_mode=pad_mode, patch_weights=patch_weights, dilation=dilation
        )
    raise ValueError(f"Unknown feature_mode={feature_mode}")


def _take_from(coords, want, rng):
    if want <= 0 or len(coords) == 0:
        return np.empty((0, 2), dtype=np.int64)
    take = min(int(want), len(coords))
    return coords[rng.choice(len(coords), size=take, replace=False)]


def _fill_missing(pick, pool_parts, budget, rng):
    if len(pick) >= budget:
        return pick[:budget]
    if not pool_parts or sum(len(part) for part in pool_parts) == 0:
        return pick
    pool = np.vstack(pool_parts)
    chosen = set(map(tuple, pick)) if len(pick) else set()
    remain_list = [coord for coord in map(tuple, pool) if coord not in chosen]
    if not remain_list:
        return pick
    remaining = np.array(remain_list, dtype=np.int64).reshape(-1, 2)
    extra = _take_from(remaining, budget - len(pick), rng)
    if len(extra):
        pick = np.vstack([pick, extra])
    return pick[:budget]


def sample_train_patch_features(
    df_train, n_pos, n_neg, tile_pos_cap=50, tile_neg_cap=50, y_threshold=0.0,
    seed=15, pad_mode="reflect", feature_mode="concat", center_weight=1.0,
    patch_weights=None, pos_inner_frac=0.9, neg_hard_frac=0.0, pos_neg_ratio=1.0,
    dilation=1,
):
    rng = np.random.default_rng(seed)
    x_chunks = []
    y_chunks = []
    need_pos = int(n_pos)
    need_neg = int(n_neg)
    rows = df_train.sample(frac=1.0, random_state=seed)

    for _, row in rows.iterrows():
        if need_pos <= 0 and need_neg <= 0:
            break
        loaded, _ = try_load_tile_xy(row, y_threshold=y_threshold)
        if loaded is None:
            continue
        x_tile, y_tile = loaded
        neigh = neigh_sum_3x3(y_tile)
        center_pos = y_tile == 1
        center_neg = y_tile == 0
        inner_pos = np.argwhere(center_pos & (neigh == 9))
        boundary_pos = np.argwhere(center_pos & (neigh >= 5) & (neigh <= 8))
        hard_neg = np.argwhere(center_neg & (neigh >= 1))
        easy_neg = np.argwhere(center_neg & (neigh == 0))

        tile_pos_budget = max(0, min(int(tile_pos_cap), need_pos))
        tile_neg_budget = max(0, min(int(tile_neg_cap), need_neg))
        want_pos_inner = int(round(float(pos_inner_frac) * tile_pos_budget))
        want_pos_boundary = tile_pos_budget - want_pos_inner
        want_neg_hard = int(round(float(neg_hard_frac) * tile_neg_budget))
        want_neg_easy = tile_neg_budget - want_neg_hard

        pos_pick = np.vstack([
            _take_from(inner_pos, want_pos_inner, rng),
            _take_from(boundary_pos, want_pos_boundary, rng),
        ])
        pos_pick = _fill_missing(pos_pick, [inner_pos, boundary_pos], tile_pos_budget, rng)
        neg_pick = np.vstack([
            _take_from(hard_neg, want_neg_hard, rng),
            _take_from(easy_neg, want_neg_easy, rng),
        ])
        neg_pick = _fill_missing(neg_pick, [hard_neg, easy_neg], tile_neg_budget, rng)

        if len(pos_pick) > 0:
            ys, xs = pos_pick[:, 0], pos_pick[:, 1]
            x_chunks.append(extract_features_at_coords(
                x_tile, ys, xs, feature_mode=feature_mode, pad_mode=pad_mode,
                center_weight=center_weight, patch_weights=patch_weights, dilation=dilation,
            ))
            y_chunks.append(np.ones((len(pos_pick),), dtype=np.int64))
            need_pos -= len(pos_pick)
        if len(neg_pick) > 0:
            ys, xs = neg_pick[:, 0], neg_pick[:, 1]
            x_chunks.append(extract_features_at_coords(
                x_tile, ys, xs, feature_mode=feature_mode, pad_mode=pad_mode,
                center_weight=center_weight, patch_weights=patch_weights, dilation=dilation,
            ))
            y_chunks.append(np.zeros((len(neg_pick),), dtype=np.int64))
            need_neg -= len(neg_pick)

    if not x_chunks:
        raise ValueError("No samples collected")
    x_train = np.vstack(x_chunks).astype(np.float32)
    y_train = np.concatenate(y_chunks).astype(np.int64)
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("After sampling, one class is empty")
    n_pos_keep = len(pos_idx)
    n_neg_keep = int(round(n_pos_keep * float(pos_neg_ratio)))
    if n_neg_keep > len(neg_idx):
        n_neg_keep = len(neg_idx)
    keep = np.concatenate([
        rng.choice(pos_idx, size=n_pos_keep, replace=False),
        rng.choice(neg_idx, size=n_neg_keep, replace=False),
    ])
    rng.shuffle(keep)
    x_train = x_train[keep]
    y_train = y_train[keep]
    perm = rng.permutation(len(y_train))
    x_train = x_train[perm]
    y_train = y_train[perm]
    print(
        f"[train] feature_mode={feature_mode} dilation={dilation} X={x_train.shape} "
        f"pos={int((y_train == 1).sum())} neg={int((y_train == 0).sum())} "
        f"missing_pos={need_pos} missing_neg={need_neg}"
    )
    return x_train, y_train


def predict_tile_features(
    estimator, x_tile, feature_mode="concat", center_weight=1.0, patch_weights=None,
    chunk_size=200_000, pad_mode="reflect", dilation=1,
):
    _, height, width = x_tile.shape
    n_pixels = height * width
    pred = np.empty((n_pixels,), dtype=np.uint8)
    ys_all = np.repeat(np.arange(height, dtype=np.int64), width)
    xs_all = np.tile(np.arange(width, dtype=np.int64), height)
    for start in range(0, n_pixels, int(chunk_size)):
        end = min(start + int(chunk_size), n_pixels)
        features = extract_features_at_coords(
            x_tile, ys_all[start:end], xs_all[start:end], feature_mode=feature_mode,
            pad_mode=pad_mode, center_weight=center_weight, patch_weights=patch_weights,
            dilation=dilation,
        )
        pred[start:end] = estimator.predict(features).astype(np.uint8)
    return pred.reshape(height, width)


def eval_on_df_newknn(
    df_eval, estimator, y_threshold=0.0, feature_mode="concat", center_weight=1.0,
    patch_weights=None, pred_chunk_size=200_000, pad_mode="reflect", dilation=1,
):
    rows = []
    for _, row in df_eval.iterrows():
        loaded, _ = try_load_tile_xy(row, y_threshold=y_threshold)
        if loaded is None:
            continue
        x_tile, y_tile = loaded
        y_pred = predict_tile_features(
            estimator, x_tile, feature_mode=feature_mode, center_weight=center_weight,
            patch_weights=patch_weights, chunk_size=pred_chunk_size, pad_mode=pad_mode,
            dilation=dilation,
        )
        rows.append({
            "fid": row["fid"],
            "fold": row["fold"],
            "tile_group": row.get("tile_group"),
            **compute_metrics(y_tile, y_pred),
        })
    return pd.DataFrame(rows)


def train_one_far_group(df, far_group, params=None):
    params = {**KNN_PARAMS, **(params or {})}
    frames = MetadataLibrary(metadata=df).get_geographic_distance_frames(far_group_ind=far_group, split=True)
    df_train = frames["train"].copy()
    df_near = frames["test"].copy()
    df_far = frames["test_far"].copy()

    x_train, y_train = sample_train_patch_features(
        df_train=df_train, n_pos=params["n_pos"], n_neg=params["n_neg"],
        tile_pos_cap=params["tile_pos_cap"], tile_neg_cap=params["tile_neg_cap"],
        y_threshold=params["y_threshold"], seed=params["seed"],
        pad_mode=params["pad_mode"], feature_mode=params["feature_mode"],
        center_weight=params["center_weight"], patch_weights=None,
        pos_inner_frac=params["pos_inner_frac"], neg_hard_frac=params["neg_hard_frac"],
        pos_neg_ratio=params["pos_neg_ratio"], dilation=params["dilation"],
    )
    estimator = KNeighborsClassifier(
        n_neighbors=params["n_neighbors"], weights=params["weights"], n_jobs=-1,
    )
    estimator.fit(x_train, y_train)

    eval_kwargs = {
        "y_threshold": params["y_threshold"],
        "feature_mode": params["feature_mode"],
        "center_weight": params["center_weight"],
        "patch_weights": None,
        "pred_chunk_size": params["pred_chunk_size"],
        "pad_mode": params["pad_mode"],
        "dilation": params["dilation"],
    }
    near_metrics = eval_on_df_newknn(df_near, estimator, **eval_kwargs)
    far_metrics = eval_on_df_newknn(df_far, estimator, **eval_kwargs)
    return estimator, df_train, near_metrics, far_metrics


def load_training_frame(data_paths):
    return load_xy_metadata_frame(data_paths)
