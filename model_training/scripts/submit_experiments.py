#!/usr/bin/env python3
"""
Generate experiment configurations and submit SLURM array jobs.

This script automates the process of:
1. Generating experiment configurations
2. Creating necessary directories
3. Submitting SLURM array jobs (with optional GPU support)
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# Add project root and scripts dir to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.experiment_configs import save_experiment_configs, get_experiment_configs
from slurm_utils import (
    DEFAULT_DATA_ARCHIVE,
    _GPU_SLURM_DIRECTIVES,
    _dataset_extraction_block,
    add_common_slurm_args,
    add_behavior_flags,
    python_env_block,
    submit_job,
)


def parse_args():
    """Parse command line arguments."""
    import argparse

    available_keywords = list(get_experiment_configs().keys())

    parser = argparse.ArgumentParser(
        description="Generate configs and submit SLURM array jobs"
    )
    parser.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Directory to save experiment configurations (default: experiments)",
    )
    add_common_slurm_args(parser)
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        choices=["h100:1", "h100:2", "h100:3", "h100:4", "1g.10gb", "2g.20gb", "3g.40gb"],
        help=(
            "GPU configuration to request. Options:\n"
            "  h100:1       — one full H100-80gb (--gpus=h100:1)\n"
            "  h100:2/3/4   — 2–4 full H100s on one node (--gpus-per-node=h100:N)\n"
            "  1g.10gb      — MIG 1/8 slice, 10 GB (--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1)\n"
            "  2g.20gb      — MIG 2/8 slice, 20 GB (--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1)\n"
            "  3g.40gb      — MIG 3/8 slice, 40 GB (--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1)\n"
            "Omit to request no GPU."
        ),
    )
    add_behavior_flags(parser)
    parser.add_argument(
        "--experiment_indices",
        type=str,
        default=None,
        help="Comma-separated list of experiment indices to run (e.g., '0,2,5'). If not specified, runs all.",
    )
    parser.add_argument(
        "--ssh_key",
        type=str,
        default=None,
        help="Path to SSH private key for installing private packages on compute nodes.",
    )
    parser.add_argument(
        "--experiment_keywords",
        type=str,
        nargs="*",
        default=None,
        help=f"Run only the experiments matching these keywords (takes precedence over --experiment_indices). Available: {available_keywords}",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Absolute path to the project root on the cluster (default: current working directory).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Dataloader workers per task. Defaults to cpus_per_task. Must be <= cpus_per_task.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size override for training (default: 4)",
    )
    parser.add_argument(
        "--data_archive",
        type=str,
        default=DEFAULT_DATA_ARCHIVE,
        help="Dataset archive filename under $DATASET_ROOT (default: compressed_dataset.tar.gz). Supports .tar and .tar.gz. A fallback archive is tried and auto-creation is attempted only when using the default name.",
    )
    parser.add_argument(
        "--config_paths",
        type=str,
        nargs="+",
        default=None,
        help="Paths to JSON config files to run directly. Bypasses keyword-based config generation.",
    )
    return parser.parse_args()


def create_slurm_script(
    num_experiments: int,
    max_parallel: int,
    time_limit: str,
    mem_per_cpu: str,
    cpus_per_task: int,
    account: str,
    output_path: str,
    source_dir: str,
    gpu_config: str = None,
    experiment_indices: str = None,
    ssh_key: str = None,
    batch_size: int = 4,
    num_workers: int = None,
    data_archive: str = DEFAULT_DATA_ARCHIVE,
    metadata_copy_block: str = "",
) -> str:
    """Create a customized SLURM script."""
    if experiment_indices:
        array_spec = f"--array={experiment_indices}%{max_parallel}"
    else:
        array_spec = f"--array=0-{num_experiments-1}%{max_parallel}"

    effective_num_workers = num_workers if num_workers is not None else cpus_per_task

    gpu_directives = ""
    gpu_info_comment = ""
    gpu_setup_code = ""
    ssh_setup_code = ""

    if ssh_key:
        # Compute Canada compute nodes block outbound port 22, so we route
        # github.com through ssh.github.com:443 (GitHub's HTTPS-port SSH
        # endpoint). We also force User=git, since the URL in requirements.txt
        # has no username and SSH would otherwise default to $USER.
        ssh_setup_code = f"""
# Copy SSH key and configure git SSH authentication for private packages
echo "Setting up SSH key for private package installation..."
mkdir -p $SLURM_TMPDIR/.ssh
cp "{ssh_key}" $SLURM_TMPDIR/.ssh/pip_install_key
chmod 600 $SLURM_TMPDIR/.ssh/pip_install_key

cat > $SLURM_TMPDIR/.ssh/config <<EOF
Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile $SLURM_TMPDIR/.ssh/pip_install_key
    IdentitiesOnly yes
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
chmod 600 $SLURM_TMPDIR/.ssh/config

export GIT_SSH_COMMAND="ssh -F $SLURM_TMPDIR/.ssh/config"
"""

    if gpu_config:
        gpu_directives = _GPU_SLURM_DIRECTIVES[gpu_config]
        gpu_info_comment = f"# GPU config: {gpu_config}"
        gpu_setup_code = """
# GPU Information
echo "GPU Information:"
nvidia-smi
echo ""
"""

    script_content = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time_limit}
#SBATCH --job-name=parallel_training
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH {array_spec}
{gpu_directives}

# Auto-generated SLURM submission script
# Total experiments: {num_experiments}
# Max parallel: {max_parallel}
# Time limit: {time_limit}
# Memory per CPU: {mem_per_cpu}
# CPUs per task: {cpus_per_task}
{gpu_info_comment}

set -e
set -a
source .env
set +a

# Expand any leading ~ in DATASET_ROOT (tilde is not expanded inside .env values)
DATASET_ROOT="${{DATASET_ROOT/#\~/$HOME}}"

echo "=========================================="
echo "Starting Job Array Task ${{SLURM_ARRAY_TASK_ID}}"
echo "Job ID: ${{SLURM_JOB_ID}}"
echo "Array Task ID: ${{SLURM_ARRAY_TASK_ID}}"
echo "Node: $(hostname)"
echo "=========================================="
{gpu_setup_code}
# Define paths
SOURCEDIR={source_dir}
EXPERIMENT_DIR=$SOURCEDIR/experiments
CONFIG_FILE=$EXPERIMENT_DIR/experiment_${{SLURM_ARRAY_TASK_ID}}_config.json
EXPERIMENT_SUBDIR=$EXPERIMENT_DIR/experiment_${{SLURM_ARRAY_TASK_ID}}

# Check if experiment directory exists
if [ ! -d "$EXPERIMENT_SUBDIR" ]; then
    echo "ERROR: Experiment directory not found: $EXPERIMENT_SUBDIR"
    exit 1
fi

echo "Experiment directory: $EXPERIMENT_SUBDIR"

{python_env_block(source_dir, ssh_setup_code)}

# Setup data directories
echo "Setting up data directories..."
mkdir -p $SLURM_TMPDIR/data
mkdir -p $SLURM_TMPDIR/models

{_dataset_extraction_block(data_archive)}

{metadata_copy_block}

# Run training
echo "=========================================="
echo "Starting training for experiment ${{SLURM_ARRAY_TASK_ID}}"
echo "=========================================="

cd $SOURCEDIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 -u run.py \\
    --slurm_worker \\
    --experiment_dir "$EXPERIMENT_SUBDIR" \\
    --data_path "$SLURM_TMPDIR/data" \\
    --output_dir "$SLURM_TMPDIR/models" \\
    --cache_dir "$SLURM_TMPDIR/cache" \\
    --cpus_per_task {cpus_per_task} \\
    --num_workers {effective_num_workers} \\
    --batch_size {batch_size}


TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "Training completed successfully"

# Archive results
echo "=========================================="
echo "Archiving results"
echo "=========================================="

JOB_ID=${{SLURM_JOB_ID:-$$}}
TASK_ID=${{SLURM_ARRAY_TASK_ID:-0}}
ARCHIVE_NAME="results_job_${{JOB_ID}}_task_${{TASK_ID}}.tar.gz"

LATEST_DIR=$(ls -td $SLURM_TMPDIR/models/*/ 2>/dev/null | head -1)
LATEST_DIR=${{LATEST_DIR%/}}

if [[ -d "${{LATEST_DIR}}" ]]; then
    echo "Archiving directory: ${{LATEST_DIR}}"
    tar czf "${{ARCHIVE_NAME}}" -C "$(dirname "${{LATEST_DIR}}")" "$(basename "${{LATEST_DIR}}")"

    ARCHIVE_SIZE=$(du -h "${{ARCHIVE_NAME}}" | cut -f1)
    echo "Archive created: ${{ARCHIVE_NAME}} (${{ARCHIVE_SIZE}})"
else
    echo "WARNING: Latest directory not found"
    exit 1
fi

# Copy to shared storage
echo "Transferring results..."
RESULTS_DIR=$SOURCEDIR/results
mkdir -p $RESULTS_DIR
cp "${{ARCHIVE_NAME}}" "$RESULTS_DIR/"

if [ $? -eq 0 ]; then
    echo "Results copied to: $RESULTS_DIR/${{ARCHIVE_NAME}}"
    rm "${{ARCHIVE_NAME}}"
else
    echo "WARNING: Copy failed"
fi

echo "=========================================="
echo "Job Array Task ${{SLURM_ARRAY_TASK_ID}} Completed Successfully"
echo "=========================================="
"""

    with open(output_path, "w") as f:
        f.write(script_content)

    os.chmod(output_path, 0o755)
    return output_path


def main():
    """Main execution function."""
    args = parse_args()

    print("========================================")
    print("SLURM Array Job Submission Tool")
    print("========================================")

    print("\nCreating directories...")
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(args.experiments_dir, exist_ok=True)

    # Clone hedgementation_utils onto the shared filesystem if not already present.
    # This runs on the login node (which has network access) so the compute nodes
    # can install from the local path without any outbound connections.
    utils_dir = Path("hedgementation_utils")
    if not utils_dir.exists():
        print("\nCloning hedgementation_utils (login node)...")
        clone_env = os.environ.copy()
        if args.ssh_key:
            clone_env["GIT_SSH_COMMAND"] = (
                f"ssh -i {args.ssh_key} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no"
            )
        subprocess.run(
            ["git", "clone", "git@github.com:hedgementation/hedgementation_utils.git"],
            check=True,
            env=clone_env,
        )
        print(f"Cloned to {utils_dir.resolve()}")
    else:
        print(f"\nUsing existing hedgementation_utils at {utils_dir.resolve()}")

    if args.config_paths:
        print(f"\nLoading {len(args.config_paths)} custom config(s) into '{args.experiments_dir}'...")
        experiments = {}
        matching = {}
        for i, path in enumerate(args.config_paths):
            dest = Path(args.experiments_dir) / f"experiment_{i}_config.json"
            shutil.copy(path, str(dest))
            with open(path) as f:
                config_data = json.load(f)
            keyword = config_data.get("keyword", Path(path).stem)
            experiments[keyword] = SimpleNamespace(keyword=keyword)
            exp_dir = Path(args.experiments_dir) / f"experiment_{i}"
            (exp_dir / "metadata").mkdir(parents=True, exist_ok=True)
            matching[i] = {"exp_keyword": keyword, "config_keyword": keyword}
        matching_path = Path(args.experiments_dir) / "predefined_experiments_matching.json"
        with open(matching_path, "w") as f:
            json.dump(matching, f, indent=4)
        num_experiments = len(args.config_paths)
    else:
        print(f"\nGenerating experiment configurations in '{args.experiments_dir}'...")
        num_experiments = save_experiment_configs(args.experiments_dir)
        experiments = get_experiment_configs()

    print(f"\nLoaded {num_experiments} experiments:")
    for idx, (config_keyword, exp) in enumerate(experiments.items()):
        print(f"  [{idx}] {config_keyword}  ({exp.keyword})")

    if args.generate_only:
        print("\n--generate_only flag set. Skipping job submission.")
        print(f"\nTo submit manually, run:")
        print(f"  sbatch scripts/slurm_array_job_generated.sh")
        return 0

    # Resolve --experiment_keywords to a comma-separated index string (takes precedence over --experiment_indices)
    selected_indices = None
    if args.experiment_keywords:
        keyword_to_idx = {k: i for i, k in enumerate(experiments.keys())}
        resolved = []
        for kw in args.experiment_keywords:
            if kw not in keyword_to_idx:
                print(f"ERROR: Unknown keyword '{kw}'. Available: {list(keyword_to_idx.keys())}")
                return 1
            resolved.append(keyword_to_idx[kw])
        selected_indices = ",".join(str(i) for i in resolved)
        print(f"\nRunning experiments by keyword: {' '.join(args.experiment_keywords)}")
    elif args.experiment_indices:
        parsed_indices = [int(i) for i in args.experiment_indices.split(",")]
        for idx in parsed_indices:
            if idx >= num_experiments:
                print(f"ERROR: Experiment index {idx} out of range (0-{num_experiments-1})")
                return 1
        selected_indices = ",".join(str(i) for i in parsed_indices)

    if selected_indices:
        print(f"\nRunning specific experiments: {selected_indices}")
    else:
        print(f"\nRunning all {num_experiments} experiments")

    print("\nCreating SLURM submission script...")
    script_path = create_slurm_script(
        num_experiments=num_experiments,
        max_parallel=args.max_parallel,
        time_limit=args.time,
        mem_per_cpu=args.mem_per_cpu,
        cpus_per_task=args.cpus_per_task,
        account=args.account,
        output_path="scripts/slurm_array_job_generated.sh",
        source_dir=args.source_dir or os.getcwd(),
        gpu_config=args.gpu,
        experiment_indices=selected_indices,
        ssh_key=args.ssh_key,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        data_archive=args.data_archive,
    )

    print(f"Created: {script_path}")
    print(f"\nJob configuration:")
    print(f"  Account: {args.account}")
    print(f"  Time limit: {args.time}")
    print(f"  Memory per CPU: {args.mem_per_cpu}")
    print(f"  CPUs per task: {args.cpus_per_task}")
    print(f"  Max parallel: {args.max_parallel}")
    if args.gpu:
        print(f"  GPU config: {args.gpu}  ({_GPU_SLURM_DIRECTIVES[args.gpu].lstrip('#SBATCH ')})")

    if args.dry_run:
        print("\n--dry_run flag set. Would submit:")
        print(f"  sbatch {script_path}")
        return 0

    print(f"\nSubmitting job array to SLURM...")
    job_id = submit_job(script_path)
    if job_id is None:
        return 1

    print(f"\n✓ Successfully submitted job array: {job_id}")
    print(f"\nMonitor jobs with:")
    print(f"  squeue -u $USER")
    print(f"  python scripts/monitor_jobs.py --job_id {job_id}")
    print(f"\nView logs in: logs/")
    print(f"View results in: results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
