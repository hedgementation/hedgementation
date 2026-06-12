# SLURM Parallel Training - Quick Start

5-minute path to running parallel training experiments on Compute Canada via the
`Trainer` / `TrainerConfig` infrastructure. Full reference:
[`scripts/README.md`](scripts/README.md).

## Prerequisites

- Compute Canada access (Cedar / Graham / Béluga / Narval)
- Project at `~/projects/def-mlecuyer/<user>/model_training_from_scratch`
- A dataset archive in `$DATASET_ROOT`: `compressed_dataset.tar.gz` or `small-data.tar.gz`
- A `.env` in the project root (sourced by the SLURM script) with at least:
  ```
  DATASET_ROOT=/home/.../links/projects/def-mlecuyer/.../dataset
  VERSION=1.3
  NUM_TILEGROUPS=5
  MODEL_DIR=models
  ```

## 5-minute path

### 1. Add (or pick) an experiment

Each experiment is a keyword in `TrainerConfig.predefined_config_keywords`
(`src/training/trainer_config.py`). Pick one of the existing keywords (e.g.
`"default"`, `"train_temperate"`, `"agriculture_mask_buffered_10m"`) or add a
new one:

```python
# In TrainerConfig.predefined_config_keywords (list)
"my_experiment",

# In TrainerConfig.get_predefined_config (match statement)
case "my_experiment":
    experiment_config.update({
        "keyword": f"UTAE_hedge_{VERSION}_my_experiment",
        "metadata_frames": metadata_library.get_split_metadata(),
        "lr": 1e-4,
        "batch_size": 16,
        "save_path": os.path.join(MODEL_DIR, "my_experiment"),
    })
```

### 2. Submit

```bash
ssh username@cedar.computecanada.ca
cd ~/projects/def-mlecuyer/<user>/model_training_from_scratch

# All experiments
python scripts/submit_experiments.py

# Or just a subset
python scripts/submit_experiments.py --experiment_indices "0,2,5"

# With H100 GPUs
python scripts/submit_experiments.py --use_gpu
```

### 3. Monitor

```bash
python scripts/monitor_jobs.py --watch
# or one-shot:
python scripts/monitor_jobs.py
```

### 4. Collect results

```bash
python scripts/manage_results.py list
python scripts/manage_results.py extract --all
python scripts/manage_results.py summary --metric iou
```

Done.

## Common adjustments

```bash
# Resources
python scripts/submit_experiments.py \
    --time "24:00:00" \
    --mem_per_cpu "32G" \
    --max_parallel 6

# Dry run / generate-only
python scripts/submit_experiments.py --dry_run
python scripts/submit_experiments.py --generate_only
```

## Local sanity check (no SLURM)

Before submitting, validate the whole pipeline on your local machine with
`--local_test`. It generates the configs and the SLURM script exactly as a real
submission would, syntax-checks the script, then runs the same worker entrypoint
the cluster runs (`run.py --slurm_worker`) with `--smoke_test` (1 epoch on a few
datapoints, evals skipped). Requires `DATASET_ROOT` in `.env` to point at a local
copy of the dataset.

```bash
# Sanity-check locally (same args you plan to submit with, plus --local_test)
python scripts/submit_experiments.py --experiment_indices "0,2,5" --local_test

# If it passes, submit for real
python scripts/submit_experiments.py --experiment_indices "0,2,5"
```

Alternatively, `run_single_experiment.py` works without `submit_experiments.py`:

```bash
python scripts/run_single_experiment.py \
    --predefined_keyword default \
    --data_path /path/to/data \
    --save_path /path/to/models
```

## Where things live

```
scripts/                       SLURM submission, monitoring, results management
src/training/trainer.py        Trainer class
src/training/trainer_config.py TrainerConfig + predefined keywords
src/training/experiment_configs.py
                               get_experiment_configs / save_experiment_configs
                               (used by submit_experiments.py)
experiments/                   Generated JSON configs + per-experiment metadata CSVs (gitignored)
logs/                          SLURM stdout/stderr (gitignored)
results/                       Archived training output (gitignored)
```

## Troubleshooting

**Job pending** → `squeue -u $USER`, check the reason column.
**OOM** → drop `batch_size`, or `--mem_per_cpu 32G`.
**Job fails immediately** → `cat logs/parallel_training-*.err`. Usually missing
dataset archive in `$DATASET_ROOT`, missing `.env`, or import error.
**Need full reference** → see `scripts/README.md`.

## SLURM commands

```bash
squeue -u $USER
squeue -j 12345678
scancel 12345678
sacct -j 12345678 --format=JobID,JobName,State,Elapsed,MaxRSS
sshare -u $USER
```

- Compute Canada docs: https://docs.alliancecan.ca/
- SLURM reference: https://slurm.schedmd.com/quickstart.html
