#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd

from src.embedding_baseline_models.common import resolve_data_paths, write_manifest
from src.embedding_baseline_models import knn, logreg, random_forest, xgboost_baseline


MODEL_MODULES = {
    "logreg": (logreg.LOGREG_PARAMS, logreg.load_training_frame, logreg.train_one_far_group),
    "knn": (knn.KNN_PARAMS, knn.load_training_frame, knn.train_one_far_group),
    "rf": (random_forest.RF_PARAMS, random_forest.load_training_frame, random_forest.train_one_far_group),
    "xgb": (xgboost_baseline.XGB_PARAMS, xgboost_baseline.load_training_frame, xgboost_baseline.train_one_far_group),
}


def summarize_metrics(near_metrics, far_metrics):
    def summarize(frame):
        if frame.empty:
            return {"iou": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return frame[["iou", "precision", "recall", "f1"]].mean().to_dict()

    near = summarize(near_metrics)
    far = summarize(far_metrics)
    return {
        "near_iou": near["iou"],
        "near_precision": near["precision"],
        "near_recall": near["recall"],
        "near_f1": near["f1"],
        "far_iou": far["iou"],
        "far_precision": far["precision"],
        "far_recall": far["recall"],
        "far_f1": far["f1"],
    }


def train_and_save(model_name, output_root, far_groups, data_paths, dataset_name):
    params, load_frame, train_one_far_group = MODEL_MODULES[model_name]
    df = load_frame(data_paths)
    output_dir = Path(output_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for far_group in far_groups:
        estimator, df_train, near_metrics, far_metrics = train_one_far_group(df, far_group, params)
        metric_summary = summarize_metrics(near_metrics, far_metrics)
        bundle = {
            "model_name": model_name,
            "far_group": far_group,
            "estimator": estimator,
            "params": params,
            "dataset": {
                "name": dataset_name,
                "path_source": "runtime CLI arguments or HEDGEMENTATION_* environment variables",
            },
            "feature_config": {
                "feature_mode": params["feature_mode"],
                "center_weight": params["center_weight"],
            },
            "train_tile_count": int(len(df_train)),
            "summary": metric_summary,
        }
        model_path = output_dir / f"best_model_far_group_{far_group}.joblib"
        joblib.dump(bundle, model_path)
        near_metrics.to_csv(output_dir / f"near_metrics_far_group_{far_group}.csv", index=False)
        far_metrics.to_csv(output_dir / f"far_metrics_far_group_{far_group}.csv", index=False)
        summary_rows.append(
            {
                "model_name": model_name,
                "far_group": far_group,
                "model_path": str(model_path),
                "n_train_tiles": int(len(df_train)),
                **metric_summary,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    write_manifest(
        output_dir / "manifest.json",
        {
            "model_name": model_name,
            "params": params,
            "dataset": {
                "name": dataset_name,
                "path_source": "runtime CLI arguments or HEDGEMENTATION_* environment variables",
            },
            "models": summary_rows,
        },
    )
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Train and save embedding baseline models.")
    parser.add_argument("--dataset-name", default="hedgementation data_1.2")
    parser.add_argument("--model", choices=["logreg", "knn", "rf", "xgb", "all"], default="all")
    parser.add_argument("--output-root", default="models/embedding_baseline_models")
    parser.add_argument("--far-groups", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--x-dir", default=None)
    parser.add_argument("--y-dir", default=None)
    parser.add_argument("--metadata-geojson", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data_paths = resolve_data_paths(
        x_dir=args.x_dir,
        y_dir=args.y_dir,
        metadata_geojson=args.metadata_geojson,
    )
    model_names = ["logreg", "knn", "rf", "xgb"] if args.model == "all" else [args.model]
    for model_name in model_names:
        summary = train_and_save(model_name, args.output_root, args.far_groups, data_paths, args.dataset_name)
        print(f"\n{model_name} saved:")
        print(summary)


if __name__ == "__main__":
    main()

