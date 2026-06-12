#!/usr/bin/env python3
"""
Create an Optuna hyperparameter search study and submit SLURM array jobs.

Each SLURM task runs run_optuna_trial.py, which picks up one trial from the
shared study database (SQLite by default). All workers coordinate through
the database — no central scheduler needed.

Usage:
    # Launch 50 trials with 8 running in parallel
    python scripts/optuna_search.py \\
        --study_name hedge_hpo_v1 \\
        --base_keyword default \\
        --n_trials 50 \\
        --num_epochs_override 30 \\
        --pruning \\
        --use_gpu \\
        --time 04:00:00 \\
        --max_parallel 8

    # Check results after jobs finish
    python scripts/optuna_search.py \\
        --study_name hedge_hpo_v1 \\
        --base_keyword default \\
        --summary_only \\
        --save_best_config best_hpo_config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import optuna

from slurm_utils import (
    add_common_slurm_args,
    add_behavior_flags,
    python_env_block,
    submit_job,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Launch Optuna HPO on SLURM")

    # Study identity
    parser.add_argument("--study_name", required=True, help="Name for this Optuna study")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Default: sqlite:///<study_name>.db in project root",
    )
    parser.add_argument(
        "--direction",
        default="maximize",
        choices=["maximize", "minimize"],
        help="Optimization direction (default: maximize)",
    )

    # Trial config
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument(
        "--base_keyword",
        required=True,
        help="Predefined config keyword to use as the base for all trials",
    )
    parser.add_argument(
        "--metric",
        default="val_iou",
        help="Metric to optimize (default: val_iou)",
    )
    parser.add_argument(
        "--num_epochs_override",
        type=int,
        default=None,
        help="Override num_epochs for trials (recommend 20-40 for faster HPO)",
    )
    parser.add_argument(
        "--pruning",
        action="store_true",
        help="Enable MedianPruner to cut unpromising trials early",
    )

    # SLURM resources
    add_common_slurm_args(parser)
    parser.add_argument("--use_gpu", action="store_true", help="Request h100 GPU per task")
    parser.add_argument(
        "--gpus_per_task", type=int, default=1, help="Number of h100 GPUs (default: 1)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size override for training (default: 4)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Dataloader workers per task. Defaults to cpus_per_task.",
    )

    # Source directory
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help="Absolute path to the project root on the cluster (default: current working directory).",
    )

    # Behavior flags
    add_behavior_flags(parser)
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Print study summary without submitting new jobs",
    )
    parser.add_argument(
        "--save_best_config",
        type=str,
        default=None,
        help="Path to save best trial params as JSON (use with --summary_only)",
    )

    return parser.parse_args()


def create_slurm_script(
    study_name: str,
    storage: str,
    base_keyword: str,
    metric: str,
    n_trials: int,
    max_parallel: int,
    time_limit: str,
    mem_per_cpu: str,
    cpus_per_task: int,
    account: str,
    output_path: str,
    source_dir: str,
    use_gpu: bool = False,
    gpus_per_task: int = 1,
    pruning: bool = False,
    num_epochs_override: int = None,
    batch_size: int = 4,
    num_workers: int = None,
) -> str:
    gpu_directive = f"#SBATCH --gres=gpu:h100:{gpus_per_task}" if use_gpu else ""
    gpu_info_block = "echo 'GPU Information:'\nnvidia-smi\necho ''" if use_gpu else ""
    epochs_arg = f"--num_epochs_override {num_epochs_override}" if num_epochs_override else ""
    pruning_arg = "--pruning" if pruning else ""
    effective_num_workers = num_workers if num_workers is not None else cpus_per_task

    script = f"""#!/bin/bash
#SBATCH --account={account}
#SBATCH --mem-per-cpu={mem_per_cpu}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time_limit}
#SBATCH --job-name=optuna_{study_name}
#SBATCH --output=logs/optuna_{study_name}-%A_%a.out
#SBATCH --error=logs/optuna_{study_name}-%A_%a.err
#SBATCH --array=0-{n_trials - 1}%{max_parallel}
{gpu_directive}

set -e
set -a
source .env
set +a

# Expand any leading ~ in DATASET_ROOT (tilde is not expanded inside .env values)
DATASET_ROOT="${{DATASET_ROOT/#\~/$HOME}}"

SOURCEDIR={source_dir}

echo "=========================================="
echo "Optuna trial ${{SLURM_ARRAY_TASK_ID}} / {n_trials - 1}"
echo "Study: {study_name}"
echo "Node: $(hostname)"
echo "=========================================="
{gpu_info_block}

{python_env_block(source_dir)}

echo "Setting up data..."
mkdir -p $SLURM_TMPDIR/data
mkdir -p $SLURM_TMPDIR/models

if [ -f "$DATASET_ROOT/compressed_dataset.tar" ]; then
    tar xf $DATASET_ROOT/compressed_dataset.tar -C $SLURM_TMPDIR/data
elif [ -f "$DATASET_ROOT/small-data.tar" ]; then
    tar xf $DATASET_ROOT/small-data.tar -C $SLURM_TMPDIR/data
else
    echo "WARNING: No dataset archive found at $DATASET_ROOT"
fi

cd $SOURCEDIR

echo "Starting trial..."
python3 -u scripts/run_optuna_trial.py \\
    --study_name "{study_name}" \\
    --storage "{storage}" \\
    --base_keyword "{base_keyword}" \\
    --metric "{metric}" \\
    --batch_size {batch_size} \\
    --num_workers {effective_num_workers} \\
    --data_path "$SLURM_TMPDIR/data" \\
    --save_path "$SLURM_TMPDIR/models" \\
    {epochs_arg} \\
    {pruning_arg}

TRIAL_EXIT_CODE=$?

if [ $TRIAL_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Trial failed with exit code $TRIAL_EXIT_CODE"
    exit $TRIAL_EXIT_CODE
fi

echo "=========================================="
echo "Trial ${{SLURM_ARRAY_TASK_ID}} complete"
echo "=========================================="
"""

    with open(output_path, "w") as f:
        f.write(script)
    os.chmod(output_path, 0o755)
    return output_path


def print_summary(study: optuna.Study, save_best_config: str = None, base_keyword: str = None):
    trials = study.trials
    n_complete = sum(1 for t in trials if t.state == optuna.trial.TrialState.COMPLETE)
    n_pruned = sum(1 for t in trials if t.state == optuna.trial.TrialState.PRUNED)
    n_failed = sum(1 for t in trials if t.state == optuna.trial.TrialState.FAIL)

    print("\n" + "=" * 60)
    print(f"Study: {study.study_name}")
    print(f"Direction: {study.direction.name}")
    print(f"  Completed: {n_complete}")
    print(f"  Pruned:    {n_pruned}")
    print(f"  Failed:    {n_failed}")
    print(f"  Total:     {len(trials)}")

    if n_complete == 0:
        print("\nNo completed trials yet.")
        return

    best = study.best_trial
    print(f"\nBest trial: #{best.number}")
    print(f"  Value: {best.value:.6f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    if save_best_config:
        payload = {
            "base_keyword": base_keyword,
            "override": best.params,
            "best_value": best.value,
            "trial_number": best.number,
            "study_name": study.study_name,
        }
        with open(save_best_config, "w") as f:
            json.dump(payload, f, indent=4)
        print(f"\nBest config saved to: {save_best_config}")
        print(
            "Run full training with best params via:\n"
            f"  python scripts/run_single_experiment.py "
            f"--predefined_keyword {base_keyword} "
            f"(and apply overrides from {save_best_config})"
        )


def main():
    args = parse_args()

    storage = args.storage or f"sqlite:///{args.study_name}.db"
    source_dir = args.source_dir or os.getcwd()

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction=args.direction,
        load_if_exists=True,
    )

    n_existing = len(study.trials)
    print(f"Study '{args.study_name}' ready.")
    print(f"  Storage: {storage}")
    print(f"  Existing trials: {n_existing}")

    if args.summary_only:
        print_summary(study, args.save_best_config, args.base_keyword)
        return 0

    os.makedirs("logs", exist_ok=True)

    script_path = f"scripts/optuna_{args.study_name}_slurm.sh"
    create_slurm_script(
        study_name=args.study_name,
        storage=storage,
        base_keyword=args.base_keyword,
        metric=args.metric,
        n_trials=args.n_trials,
        max_parallel=args.max_parallel,
        time_limit=args.time,
        mem_per_cpu=args.mem_per_cpu,
        cpus_per_task=args.cpus_per_task,
        account=args.account,
        output_path=script_path,
        source_dir=source_dir,
        use_gpu=args.use_gpu,
        gpus_per_task=args.gpus_per_task,
        pruning=args.pruning,
        num_epochs_override=args.num_epochs_override,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"\nGenerated SLURM script: {script_path}")

    if args.generate_only:
        print("--generate_only set. Script created but not submitted.")
        return 0

    if args.dry_run:
        print(f"--dry_run set. Would run: sbatch {script_path}")
        return 0

    job_id = submit_job(script_path)
    if job_id is None:
        return 1

    print(f"\nMonitor jobs:  squeue -u $USER")
    print(f"Watch logs:    tail -f logs/optuna_{args.study_name}-{job_id}_*.out")
    print(
        f"\nAfter completion, run:\n"
        f"  python scripts/optuna_search.py "
        f"--study_name {args.study_name} "
        f"--base_keyword {args.base_keyword} "
        f"--summary_only "
        f"--save_best_config best_{args.study_name}.json"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
