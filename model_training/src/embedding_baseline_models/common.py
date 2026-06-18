import glob
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import RasterioIOError
import torch
from hedgementation_utils.metrics.seg_meter import SegMeter


@dataclass(frozen=True)
class DataPaths:
    x_dir: str
    y_dir: str
    metadata_geojson: str

    def to_dict(self):
        return {
            "x_dir": self.x_dir,
            "y_dir": self.y_dir,
            "metadata_geojson": self.metadata_geojson,
        }


def resolve_data_paths(x_dir=None, y_dir=None, metadata_geojson=None):
    x_dir = x_dir or os.environ.get("HEDGEMENTATION_X_DIR")
    y_dir = y_dir or os.environ.get("HEDGEMENTATION_Y_DIR")
    metadata_geojson = metadata_geojson or os.environ.get("HEDGEMENTATION_METADATA_GEOJSON")
    missing = [
        name
        for name, value in {
            "--x-dir or HEDGEMENTATION_X_DIR": x_dir,
            "--y-dir or HEDGEMENTATION_Y_DIR": y_dir,
            "--metadata-geojson or HEDGEMENTATION_METADATA_GEOJSON": metadata_geojson,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError("Missing dataset paths: " + ", ".join(missing))
    return DataPaths(x_dir=x_dir, y_dir=y_dir, metadata_geojson=metadata_geojson)


def parse_x_filename(path):
    base = os.path.basename(path)
    match = re.match(r"^AEF_(\d+)\.tif$", base)
    if not match:
        raise ValueError(f"Bad X filename: {base}. Expected format: AEF_<id>.tif")
    return match.group(1)

def parse_y_index(path):
    base = os.path.basename(path)
    match = re.match(r"^y_(\d+)\.npy$", base)
    if not match:
        raise ValueError(f"Bad Y filename: {base}")
    return int(match.group(1))


def build_xy_df_strict(x_dir, y_dir):
    x_files = sorted(glob.glob(os.path.join(x_dir, "AEF_*.tif")))
    y_files = sorted(glob.glob(os.path.join(y_dir, "y_*.npy")))

    if not x_files:
        raise FileNotFoundError(f"No AEF X files found in {x_dir}")
    if not y_files:
        raise FileNotFoundError(f"No Y files found in {y_dir}")

    x_map = {parse_x_filename(path): path for path in x_files}
    y_map = {str(parse_y_index(path)): path for path in y_files}

    common = sorted(set(x_map) & set(y_map), key=lambda fid: int(fid))
    if not common:
        raise ValueError("No matched AEF X/Y pairs")

    return pd.DataFrame(
        {
            "fid": int(fid),
            "x_path": x_map[fid],
            "y_path": y_map[fid],
            "y_idx": int(fid),
        }
        for fid in common
    )


def load_xy_metadata_frame(data_paths):
    df = build_xy_df_strict(data_paths.x_dir, data_paths.y_dir).copy()
    df["fid"] = df["fid"].astype(int)

    metadata = gpd.read_file(data_paths.metadata_geojson).copy()

    if "ID_PATCH" not in metadata.columns:
        raise ValueError("Metadata must contain an 'ID_PATCH' column")
    metadata["fid"] = metadata["ID_PATCH"].astype(int)

    if "fold" not in metadata.columns:
        raise ValueError("Metadata must contain a 'fold' column for train/valid/test splits")

    merged = df.merge(metadata, on="fid", how="inner", validate="one_to_one")

    if len(merged) != len(df):
        missing = sorted(set(df["fid"]) - set(merged["fid"]))
        raise ValueError(f"Missing metadata for fid(s): {missing[:10]}")

    return merged


def load_tile_xy(row, y_threshold=0.0):
    with rasterio.open(row["x_path"]) as ds:
        x_tile = ds.read()

    if x_tile.shape != (64, 128, 128):
        raise ValueError(f"Unexpected X shape for fid={row['fid']}: {x_tile.shape}")

    y_tile = np.load(row["y_path"])

    if y_tile.shape != (1, 128, 128):
        raise ValueError(f"Unexpected Y shape for fid={row['fid']}: {y_tile.shape}")

    y_tile = y_tile[0]
    y_tile = (y_tile > y_threshold).astype(np.uint8)

    return x_tile, y_tile


def try_load_tile_xy(row, y_threshold=0.0):
    try:
        return load_tile_xy(row, y_threshold=y_threshold), None
    except (RasterioIOError, OSError, ValueError) as exc:
        return None, f"Skipping fid={row['fid']} fold={row['fold']}: {type(exc).__name__}: {exc}"


def extract_3x3_features(x_tile, center_weight=1.0, patch_weights=None):
    channels, height, width = x_tile.shape
    x_pad = np.pad(x_tile, ((0, 0), (1, 1), (1, 1)), mode="reflect")
    if patch_weights is None:
        patch_weights = np.ones(9, dtype=np.float32)
        patch_weights[4] = float(center_weight)
    else:
        patch_weights = np.asarray(patch_weights, dtype=np.float32)
        if patch_weights.shape != (9,):
            raise ValueError("patch_weights must have shape (9,)")

    patches = []
    idx = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            patch = x_pad[:, 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
            patches.append(patch * patch_weights[idx])
            idx += 1

    x_stack = np.stack(patches, axis=0)
    x_out = np.transpose(x_stack, (2, 3, 0, 1)).reshape(height, width, 9 * channels)
    return x_out.reshape(-1, 9 * channels).astype(np.float32)


def flatten_tile(x_tile, y_tile, feature_mode="patch3x3", center_weight=1.0, patch_weights=None):
    channels, height, width = x_tile.shape
    if feature_mode == "pixel":
        x_flat = np.moveaxis(x_tile, 0, -1).reshape(-1, channels).astype(np.float32)
    elif feature_mode == "patch3x3":
        x_flat = extract_3x3_features(x_tile, center_weight=center_weight, patch_weights=patch_weights)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")
    return x_flat, y_tile.reshape(-1).astype(np.uint8), height, width


def collect_global_pool(df_train, y_threshold=0.0, max_pixels_per_tile=None, rng=None, skip_bad_tiles=True):
    """Accumulate per-pixel (center) features from every training tile into one pool.

    Port of random_forest/xgboost_all_data sampling.collect_global_pool: features
    are the raw per-pixel channel vector (np.moveaxis(X, 0, -1)), labels are the
    binary mask Y > y_threshold (1 = positive, 0 = negative). max_pixels_per_tile
    optionally sub-samples each tile before pooling.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    x_list, y_list = [], []
    for _, row in df_train.iterrows():
        if skip_bad_tiles:
            loaded, _ = try_load_tile_xy(row, y_threshold=y_threshold)
            if loaded is None:
                continue
            x_tile, y_tile = loaded
        else:
            x_tile, y_tile = load_tile_xy(row, y_threshold=y_threshold)

        channels = x_tile.shape[0]
        x_flat = np.moveaxis(x_tile, 0, -1).reshape(-1, channels)
        y_flat = y_tile.reshape(-1).astype(np.uint8)

        if max_pixels_per_tile is not None and x_flat.shape[0] > int(max_pixels_per_tile):
            idx = rng.choice(x_flat.shape[0], size=int(max_pixels_per_tile), replace=False)
            x_flat = x_flat[idx]
            y_flat = y_flat[idx]

        x_list.append(x_flat)
        y_list.append(y_flat)

    if not x_list:
        raise ValueError("No samples collected for the global pool")
    x_pool = np.vstack(x_list).astype(np.float32)
    y_pool = np.concatenate(y_list).astype(np.uint8)
    pos_rate = float((y_pool == 1).mean()) if y_pool.size else 0.0
    print(f"Global pool: {x_pool.shape}  pos%={pos_rate:.4f}  #pos={int((y_pool == 1).sum())}")
    return x_pool, y_pool


def balanced_sample(x_pool, y_pool, n_pos, n_neg, rng):
    """Draw n_pos positives and n_neg negatives (without replacement) from the pool.

    Port of the reference balanced_sample: if a class has fewer than requested,
    all of it is used.
    """
    pos_idx = np.where(y_pool == 1)[0]
    neg_idx = np.where(y_pool == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Pool has no positives or no negatives — cannot sample.")
    n_pos = min(int(n_pos), len(pos_idx))
    n_neg = min(int(n_neg), len(neg_idx))
    idx = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ])
    rng.shuffle(idx)
    x_s = x_pool[idx]
    y_s = y_pool[idx].astype(int)
    print(f"Sampled train: {x_s.shape}  pos%={float((y_s == 1).mean()):.4f}")
    return x_s, y_s


def predict_tile(estimator, x_tile, pred_threshold=None, feature_mode="patch3x3", center_weight=1.0, patch_weights=None):
    x_flat, _, height, width = flatten_tile(
        x_tile,
        np.zeros(x_tile.shape[1:], dtype=np.uint8),
        feature_mode=feature_mode,
        center_weight=center_weight,
        patch_weights=patch_weights,
    )
    if pred_threshold is not None and hasattr(estimator, "predict_proba"):
        y_pred = (estimator.predict_proba(x_flat)[:, 1] >= float(pred_threshold)).astype(np.uint8)
    else:
        y_pred = estimator.predict(x_flat).astype(np.uint8)
    return y_pred.reshape(height, width)


def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(np.uint8).reshape(-1)
    y_pred = y_pred.astype(np.uint8).reshape(-1)

    meter = SegMeter(num_classes=2)
    meter.update(
        torch.from_numpy(y_pred.astype(np.int64)),
        torch.from_numpy(y_true.astype(np.int64)),
    )
    metrics = meter.ask_metrics(cl=1)
    return {
        "tp": int(meter.hist[1, 1].item()),
        "fp": int(meter.hist[0, 1].item()),
        "fn": int(meter.hist[1, 0].item()),
        "iou": float(metrics.get("iou", 0.0)),
        "precision": float(metrics.get("pre", 0.0)),
        "recall": float(metrics.get("acc", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
    }


def eval_on_df(
    df_eval,
    estimator,
    y_threshold=0.0,
    pred_threshold=None,
    feature_mode="patch3x3",
    center_weight=1.0,
    patch_weights=None,
    skip_bad_tiles=True,
):
    rows = []
    skipped = []

    for _, row in df_eval.iterrows():
        if skip_bad_tiles:
            loaded, warning = try_load_tile_xy(row, y_threshold=y_threshold)
            if loaded is None:
                skipped.append(warning)
                continue
            x_tile, y_tile = loaded
        else:
            x_tile, y_tile = load_tile_xy(row, y_threshold=y_threshold)

        y_pred = predict_tile(
            estimator,
            x_tile,
            pred_threshold=pred_threshold,
            feature_mode=feature_mode,
            center_weight=center_weight,
            patch_weights=patch_weights,
        )

        rows.append(
            {
                "fid": row["fid"],
                "fold": row["fold"],
                "tile_group": row.get("tile_group"),
                **compute_metrics(y_tile, y_pred),
            }
        )

    if skipped:
        print(f"Skipped {len(skipped)} bad tiles")
        for message in skipped[:10]:
            print(message)

    return pd.DataFrame(rows)


def write_manifest(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")

