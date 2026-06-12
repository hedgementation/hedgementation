#!/usr/bin/env python3
"""
Manage and extract training results from SLURM jobs.

This script helps extract, organize, and analyze results from completed experiments.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manage training results from SLURM jobs"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List all result archives")
    list_parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory (default: results)",
    )
    list_parser.add_argument(
        "--sort",
        choices=["name", "size", "time"],
        default="time",
        help="Sort by (default: time)",
    )

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract result archives")
    extract_parser.add_argument(
        "archive",
        nargs="*",
        help="Archive(s) to extract (e.g., 'results_job_123_task_0.tar.gz')",
    )
    extract_parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory (default: results)",
    )
    extract_parser.add_argument(
        "--output_dir",
        type=str,
        default="extracted_results",
        help="Output directory (default: extracted_results)",
    )
    extract_parser.add_argument(
        "--all",
        action="store_true",
        help="Extract all archives",
    )
    extract_parser.add_argument(
        "--job_id",
        type=str,
        help="Extract all archives for a specific job ID",
    )

    # Summary command
    summary_parser = subparsers.add_parser(
        "summary", help="Show summary of extracted results"
    )
    summary_parser.add_argument(
        "--results_dir",
        type=str,
        default="extracted_results",
        help="Directory with extracted results (default: extracted_results)",
    )
    summary_parser.add_argument(
        "--metric",
        choices=["iou", "loss"],
        default="iou",
        help="Metric to compare (default: iou)",
    )

    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean up old files")
    clean_parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Results directory to clean (default: results)",
    )
    clean_parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Logs directory to clean (default: logs)",
    )
    clean_parser.add_argument(
        "--keep_days",
        type=int,
        default=30,
        help="Keep files from last N days (default: 30)",
    )
    clean_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be deleted without deleting",
    )

    return parser.parse_args()


def list_results(results_dir, sort_by="time"):
    """List all result archives."""
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    archives = []
    pattern = re.compile(r"results_job_(\d+)_task_(\d+)\.tar\.gz")

    for filename in os.listdir(results_dir):
        match = pattern.match(filename)
        if match:
            filepath = os.path.join(results_dir, filename)
            stat = os.stat(filepath)

            archives.append({
                "filename": filename,
                "job_id": match.group(1),
                "task_id": match.group(2),
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })

    if not archives:
        print(f"No result archives found in {results_dir}")
        return

    # Sort
    if sort_by == "name":
        archives.sort(key=lambda x: x["filename"])
    elif sort_by == "size":
        archives.sort(key=lambda x: x["size"], reverse=True)
    else:  # time
        archives.sort(key=lambda x: x["mtime"], reverse=True)

    # Display
    print(f"\nFound {len(archives)} result archives in {results_dir}:")
    print(f"\n{'Filename':<45} {'Job ID':<10} {'Task':<6} {'Size':<10}")
    print("-" * 75)

    total_size = 0
    for archive in archives:
        size_mb = archive["size"] / (1024 * 1024)
        total_size += archive["size"]
        print(
            f"{archive['filename']:<45} {archive['job_id']:<10} "
            f"{archive['task_id']:<6} {size_mb:>8.1f}MB"
        )

    total_mb = total_size / (1024 * 1024)
    print(f"\nTotal: {total_mb:.1f}MB")


def extract_archive(archive_path, output_dir):
    """Extract a single archive."""
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(output_dir)
        return True
    except Exception as e:
        print(f"ERROR extracting {archive_path}: {e}")
        return False


def extract_results(archives, results_dir, output_dir, extract_all=False, job_id=None):
    """Extract result archives."""
    os.makedirs(output_dir, exist_ok=True)

    # Determine which archives to extract
    to_extract = []

    if extract_all:
        pattern = re.compile(r"results_job_\d+_task_\d+\.tar\.gz")
        for filename in os.listdir(results_dir):
            if pattern.match(filename):
                to_extract.append(os.path.join(results_dir, filename))

    elif job_id:
        pattern = re.compile(rf"results_job_{job_id}_task_\d+\.tar\.gz")
        for filename in os.listdir(results_dir):
            if pattern.match(filename):
                to_extract.append(os.path.join(results_dir, filename))

    else:
        # Extract specified archives
        for archive in archives:
            if not archive.endswith(".tar.gz"):
                archive += ".tar.gz"

            if os.path.isabs(archive):
                archive_path = archive
            else:
                archive_path = os.path.join(results_dir, archive)

            if os.path.exists(archive_path):
                to_extract.append(archive_path)
            else:
                print(f"WARNING: Archive not found: {archive_path}")

    if not to_extract:
        print("No archives to extract")
        return

    print(f"\nExtracting {len(to_extract)} archive(s) to {output_dir}...")

    success = 0
    for archive_path in to_extract:
        print(f"  {os.path.basename(archive_path)}...", end=" ")
        if extract_archive(archive_path, output_dir):
            print("✓")
            success += 1
        else:
            print("✗")

    print(f"\nSuccessfully extracted {success}/{len(to_extract)} archives")


def read_params_json(params_path):
    """Read params.json file."""
    try:
        with open(params_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def read_metrics_csv(metrics_path):
    """Read metrics CSV file."""
    try:
        import pandas as pd
        return pd.read_csv(metrics_path)
    except Exception:
        return None


def show_summary(results_dir, metric="iou"):
    """Show summary of extracted results."""
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    experiments = []

    for dirname in os.listdir(results_dir):
        exp_dir = os.path.join(results_dir, dirname)
        if not os.path.isdir(exp_dir):
            continue

        params_path = os.path.join(exp_dir, "params.json")
        metrics_path = os.path.join(exp_dir, "metrics_per_epoch.csv")

        if not os.path.exists(params_path):
            continue

        params = read_params_json(params_path)
        metrics_df = read_metrics_csv(metrics_path)

        exp_info = {
            "name": dirname,
            "keyword": params.get("keyword", "unknown") if params else "unknown",
            "backbone": params.get("backbone", "unknown") if params else "unknown",
            "epochs": params.get("num_epochs", 0) if params else 0,
        }

        if metrics_df is not None and not metrics_df.empty:
            if metric == "iou":
                exp_info["best_train"] = metrics_df["train_iou"].max()
                exp_info["best_val"] = metrics_df["val_iou"].max()
                exp_info["final_val"] = metrics_df["val_iou"].iloc[-1]
            else:  # loss
                exp_info["best_train"] = metrics_df["train_loss"].min()
                exp_info["best_val"] = metrics_df["val_loss"].min()
                exp_info["final_val"] = metrics_df["val_loss"].iloc[-1]
        else:
            exp_info["best_train"] = None
            exp_info["best_val"] = None
            exp_info["final_val"] = None

        experiments.append(exp_info)

    if not experiments:
        print(f"No experiments found in {results_dir}")
        return

    # Sort by best validation metric
    experiments.sort(
        key=lambda x: x["best_val"] if x["best_val"] is not None else -float("inf"),
        reverse=(metric == "iou"),
    )

    print(f"\nExperiment Summary ({len(experiments)} experiments):")
    print(f"Metric: {metric.upper()}\n")
    print(
        f"{'Keyword':<40} {'Backbone':<10} {'Best Val':<10} "
        f"{'Final Val':<10} {'Epochs':<8}"
    )
    print("-" * 85)

    for exp in experiments:
        best_val = f"{exp['best_val']:.4f}" if exp["best_val"] is not None else "N/A"
        final_val = f"{exp['final_val']:.4f}" if exp["final_val"] is not None else "N/A"

        print(
            f"{exp['keyword']:<40} {exp['backbone']:<10} {best_val:<10} "
            f"{final_val:<10} {exp['epochs']:<8}"
        )


def clean_old_files(results_dir, logs_dir, keep_days, dry_run):
    """Clean up old files."""
    import time

    cutoff_time = time.time() - (keep_days * 24 * 60 * 60)

    to_delete = []

    # Check results
    if os.path.exists(results_dir):
        for filename in os.listdir(results_dir):
            filepath = os.path.join(results_dir, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                to_delete.append(filepath)

    # Check logs
    if os.path.exists(logs_dir):
        for filename in os.listdir(logs_dir):
            filepath = os.path.join(logs_dir, filename)
            if os.path.isfile(filepath) and os.path.getmtime(filepath) < cutoff_time:
                to_delete.append(filepath)

    if not to_delete:
        print(f"No files older than {keep_days} days found")
        return

    total_size = sum(os.path.getsize(f) for f in to_delete)
    total_mb = total_size / (1024 * 1024)

    print(f"\nFiles older than {keep_days} days:")
    print(f"  Count: {len(to_delete)}")
    print(f"  Total size: {total_mb:.1f}MB")

    if dry_run:
        print("\nWould delete:")
        for filepath in to_delete[:10]:
            print(f"  {filepath}")
        if len(to_delete) > 10:
            print(f"  ... and {len(to_delete) - 10} more")
        print("\nRun without --dry_run to actually delete")
    else:
        response = input("\nDelete these files? (yes/no): ")
        if response.lower() == "yes":
            for filepath in to_delete:
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"ERROR deleting {filepath}: {e}")
            print(f"Deleted {len(to_delete)} files")
        else:
            print("Cancelled")


def main():
    """Main execution function."""
    args = parse_args()

    if not args.command:
        print("ERROR: No command specified")
        print("Use: python manage_results.py {list,extract,summary,clean} --help")
        return 1

    if args.command == "list":
        list_results(args.results_dir, args.sort)

    elif args.command == "extract":
        if not args.all and not args.job_id and not args.archive:
            print("ERROR: Specify --all, --job_id, or archive names")
            return 1

        extract_results(
            args.archive,
            args.results_dir,
            args.output_dir,
            args.all,
            args.job_id,
        )

    elif args.command == "summary":
        show_summary(args.results_dir, args.metric)

    elif args.command == "clean":
        clean_old_files(
            args.results_dir,
            args.logs_dir,
            args.keep_days,
            args.dry_run,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
