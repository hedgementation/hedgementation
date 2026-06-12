#!/usr/bin/env python3
"""
Monitor SLURM array job progress.

This script provides a convenient way to monitor the status of submitted
SLURM array jobs and view their progress.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor SLURM array job progress")
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Specific job ID to monitor",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor (refresh every 30 seconds)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Refresh interval in seconds for --watch mode (default: 30)",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default="logs",
        help="Directory containing log files (default: logs)",
    )
    parser.add_argument(
        "--show_logs",
        action="store_true",
        help="Show recent log entries for each task",
    )
    return parser.parse_args()


def get_queue_status(job_id=None):
    """
    Get job status from SLURM queue.

    Args:
        job_id: Optional specific job ID to query

    Returns:
        Dictionary of job information
    """
    try:
        if job_id:
            cmd = ["squeue", "-j", job_id, "--format=%A,%t,%M,%r,%N"]
        else:
            cmd = ["squeue", "-u", os.environ.get("USER", ""), "--format=%A,%t,%M,%r,%N"]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        jobs = {}
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            if not line:
                continue

            parts = line.split(",")
            if len(parts) >= 5:
                job_info = {
                    "job_id": parts[0],
                    "state": parts[1],
                    "time": parts[2],
                    "reason": parts[3],
                    "node": parts[4],
                }
                jobs[parts[0]] = job_info

        return jobs

    except subprocess.CalledProcessError:
        return {}
    except FileNotFoundError:
        print("ERROR: squeue command not found. Are you on a SLURM cluster?")
        sys.exit(1)


def parse_log_file(log_path):
    """
    Parse a log file to extract progress information.

    Args:
        log_path: Path to log file

    Returns:
        Dictionary with log information
    """
    info = {
        "exists": False,
        "size": 0,
        "last_modified": None,
        "current_epoch": None,
        "total_epochs": None,
        "last_loss": None,
        "last_iou": None,
        "status": "Unknown",
        "error": None,
    }

    if not os.path.exists(log_path):
        return info

    info["exists"] = True
    info["size"] = os.path.getsize(log_path)
    info["last_modified"] = datetime.fromtimestamp(os.path.getmtime(log_path))

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()

        # Look for training progress in last 100 lines
        recent_lines = lines[-100:] if len(lines) > 100 else lines

        for line in recent_lines:
            # Extract epoch number
            epoch_match = re.search(r"Epoch (\d+)/(\d+)", line)
            if epoch_match:
                info["current_epoch"] = int(epoch_match.group(1))
                info["total_epochs"] = int(epoch_match.group(2))

            # Extract loss and IoU
            if "Loss:" in line and "IoU:" in line:
                loss_match = re.search(r"Loss: ([\d.]+)", line)
                iou_match = re.search(r"IoU: ([\d.]+)", line)
                if loss_match:
                    info["last_loss"] = float(loss_match.group(1))
                if iou_match:
                    info["last_iou"] = float(iou_match.group(1))

            # Check for completion
            if "Training complete" in line or "Completed Successfully" in line:
                info["status"] = "Completed"
            elif "ERROR" in line or "failed" in line.lower():
                info["status"] = "Error"
                if "ERROR:" in line:
                    info["error"] = line.split("ERROR:")[1].strip()[:100]

        # Determine status
        if info["status"] == "Unknown":
            if info["current_epoch"] is not None:
                info["status"] = "Running"
            elif info["size"] > 0:
                info["status"] = "Started"

    except Exception as e:
        info["error"] = str(e)

    return info


def get_log_files(logs_dir, job_id=None):
    """
    Find log files for a job.

    Args:
        logs_dir: Directory containing logs
        job_id: Optional specific job ID

    Returns:
        Dictionary mapping task IDs to log file paths
    """
    log_files = {}

    if not os.path.exists(logs_dir):
        return log_files

    # Pattern: parallel_training-JOBID_TASKID.out
    pattern = r"parallel_training-(\d+)_(\d+)\.out"

    for filename in os.listdir(logs_dir):
        match = re.match(pattern, filename)
        if match:
            file_job_id = match.group(1)
            task_id = match.group(2)

            if job_id is None or file_job_id == job_id:
                log_files[task_id] = os.path.join(logs_dir, filename)

    return log_files


def display_status(job_id, logs_dir, show_logs=False):
    """
    Display comprehensive job status.

    Args:
        job_id: Job ID to display (None for all jobs)
        logs_dir: Directory containing logs
        show_logs: Whether to show log excerpts
    """
    print("=" * 80)
    print(f"SLURM Job Status Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Get queue status
    queue_jobs = get_queue_status(job_id)

    if not queue_jobs:
        print("\nNo jobs found in queue.")
        if job_id:
            print(f"Job {job_id} may have completed or does not exist.")
    else:
        print(f"\nActive jobs: {len(queue_jobs)}")
        print("\nQueue Status:")
        print(f"{'Job ID':<15} {'State':<12} {'Runtime':<10} {'Node':<15} {'Reason':<20}")
        print("-" * 80)

        for jid, info in sorted(queue_jobs.items()):
            print(
                f"{info['job_id']:<15} {info['state']:<12} {info['time']:<10} "
                f"{info['node']:<15} {info['reason']:<20}"
            )

    # Get log file status
    log_files = get_log_files(logs_dir, job_id)

    if log_files:
        print(f"\n\nLog Files: {len(log_files)} tasks found")
        print("\nTask Progress:")
        print(
            f"{'Task':<6} {'Status':<12} {'Epoch':<12} {'Loss':<10} {'IoU':<10} {'Modified':<20}"
        )
        print("-" * 80)

        # Group by status
        status_counts = defaultdict(int)

        for task_id in sorted(log_files.keys(), key=int):
            log_path = log_files[task_id]
            info = parse_log_file(log_path)
            status_counts[info["status"]] += 1

            # Format epoch info
            epoch_str = ""
            if info["current_epoch"] is not None:
                epoch_str = f"{info['current_epoch']}/{info['total_epochs']}"

            # Format loss and IoU
            loss_str = f"{info['last_loss']:.4f}" if info["last_loss"] else "-"
            iou_str = f"{info['last_iou']:.4f}" if info["last_iou"] else "-"

            # Format modified time
            mod_str = ""
            if info["last_modified"]:
                mod_str = info["last_modified"].strftime("%Y-%m-%d %H:%M:%S")

            print(
                f"{task_id:<6} {info['status']:<12} {epoch_str:<12} "
                f"{loss_str:<10} {iou_str:<10} {mod_str:<20}"
            )

            # Show error if present
            if info["error"] and show_logs:
                print(f"       Error: {info['error']}")

        # Summary
        print("\n" + "=" * 80)
        print("Summary:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")

    else:
        print(f"\nNo log files found in '{logs_dir}/'")

    print("=" * 80)


def main():
    """Main execution function."""
    args = parse_args()

    if args.watch:
        print(f"Watching jobs (refreshing every {args.interval} seconds)...")
        print("Press Ctrl+C to exit\n")

        try:
            while True:
                # Clear screen (works on Unix/Linux)
                os.system("clear" if os.name != "nt" else "cls")

                display_status(args.job_id, args.logs_dir, args.show_logs)

                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    else:
        display_status(args.job_id, args.logs_dir, args.show_logs)


if __name__ == "__main__":
    main()
