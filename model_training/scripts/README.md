# SLURM Parallel Training Scripts

Run parallel training experiments on a SLURM cluster (Compute Canada) using the
`Trainer` (`src/training/trainer.py`) and `TrainerConfig`
(`src/training/trainer_config.py`) infrastructure.

## How it fits together

```
TrainerConfig.predefined_config_keywords            ← list of keyword names
TrainerConfig.get_predefined_config(keyword)        ← builds a TrainerConfig

src/training/experiment_configs.py
  get_experiment_configs()                          ← {keyword: TrainerConfig}
  save_experiment_configs(experiments_dir)          ← writes JSON + metadata CSVs

scripts/submit_experiments.py                       ← generates SLURM script and submits
scripts/run_single_experiment.py                    ← invoked by each array task
scripts/monitor_jobs.py                             ← squeue + log parsing
scripts/manage_results.py                           ← list / extract / summarize / clean
```

Each experiment is a keyword in `TrainerConfig.predefined_config_keywords` with a
matching `case` arm in `TrainerConfig.get_predefined_config()`.

## Adding a new experiment

1. Open `src/training/trainer_config.py`.
2. Add the keyword string to `TrainerConfig.predefined_config_keywords`.
3. Add a `case "your_keyword":` arm inside `TrainerConfig.get_predefined_config()`
   that updates `experiment_config` with the parameters for that experiment
   (metadata frames, model overrides, save path, etc.).

Example:

```python
case "my_experiment":
    experiment_config.update({
        "keyword": f"UTAE_hedge_{VERSION}_my_experiment",
        "metadata_frames": metadata_library.get_split_metadata(),
        "lr": 1e-4,
        "batch_size": 16,
        "save_path": os.path.join(MODEL_DIR, "my_experiment"),
    })
```

`BASE_CONFIG` (defined in `trainer_config.py`) supplies the defaults.

## Submitting jobs

```bash
# Generate per-experiment configs and submit a SLURM array job
python scripts/submit_experiments.py

# Run only a subset of experiments
python scripts/submit_experiments.py --experiment_indices "0,2,5"

# Resource overrides
python scripts/submit_experiments.py \
    --max_parallel 5 \
    --time "24:00:00" \
    --mem_per_cpu "32G" \
    --cpus_per_task 16

# H100 GPUs
python scripts/submit_experiments.py --use_gpu --gpus_per_task 1

# Generate configs without submitting
python scripts/submit_experiments.py --generate_only

# Print the sbatch invocation without running it
python scripts/submit_experiments.py --dry_run
```

`submit_experiments.py` writes `scripts/slurm_array_job_generated.sh` (gitignored)
on every run and submits it via `sbatch`.

### `submit_experiments.py` flags

| Flag                   | Default       | Purpose                                         |
| ---------------------- | ------------- | ----------------------------------------------- |
| `--experiments_dir`    | `experiments` | Where per-experiment JSON + metadata are saved  |
| `--max_parallel`       | `4`           | `%N` array concurrency cap                      |
| `--time`               | `12:00:00`    | `--time` per task                               |
| `--mem_per_cpu`        | `20G`         | `--mem-per-cpu`                                 |
| `--cpus_per_task`      | `1`           | `--cpus-per-task`                               |
| `--account`            | `def-mlecuyer`| `--account`                                     |
| `--use_gpu`            | off           | Adds `--gres=gpu:h100:N`                        |
| `--gpus_per_task`      | `1`           | `N` for the GPU spec above                      |
| `--experiment_indices` | all           | Comma-separated subset, e.g. `"0,2,5"`          |
| `--generate_only`      | off           | Write configs + script, skip `sbatch`           |
| `--dry_run`            | off           | Print the `sbatch` command, don't submit        |

## Monitoring

```bash
# Snapshot
python scripts/monitor_jobs.py

# Watch (default refresh 30s)
python scripts/monitor_jobs.py --watch
python scripts/monitor_jobs.py --watch --interval 60

# Single job
python scripts/monitor_jobs.py --job_id 12345678 --watch
```

The monitor parses `logs/parallel_training-<JOB>_<TASK>.{out,err}` to surface
epoch / loss / IoU progress alongside `squeue` state.

## Collecting results

`run_single_experiment.py` writes the trainer's output dir to
`$SLURM_TMPDIR/models/<keyword>/`. The SLURM script then archives that
directory to `results/results_job_<JOB>_task_<TASK>.tar.gz`.

```bash
# List archives
python scripts/manage_results.py list

# Extract everything (or filter by job)
python scripts/manage_results.py extract --all
python scripts/manage_results.py extract --job_id 12345678

# Summary, sorted by best validation metric
python scripts/manage_results.py summary --metric iou
python scripts/manage_results.py summary --metric loss

# Drop archives + logs older than N days
python scripts/manage_results.py clean --keep_days 30 --dry_run
```

## Local / single-experiment run

`run_single_experiment.py` accepts either a JSON config (the SLURM array path)
or a predefined keyword (the manual path):

```bash
# By keyword (no submit_experiments.py step needed)
python scripts/run_single_experiment.py \
    --predefined_keyword default \
    --data_path /path/to/data \
    --save_path /path/to/models

# From a generated JSON config
python scripts/run_single_experiment.py \
    --config_file experiments/experiment_0_config.json \
    --metadata_dir experiments/experiment_0/metadata \
    --data_path $SLURM_TMPDIR/data \
    --save_path $SLURM_TMPDIR/models
```

## Directory layout (after a run)

```
model_training_from_scratch/
├── scripts/
│   ├── submit_experiments.py
│   ├── run_single_experiment.py
│   ├── monitor_jobs.py
│   ├── manage_results.py
│   ├── download_dataset_from_gdrive.sh
│   ├── send_dataset_to_compute_canada.sh
│   └── README.md
├── experiments/                  # gitignored
│   ├── experiment_0_config.json
│   ├── experiment_0/metadata/{train,valid,test}.csv
│   └── ...
├── logs/                         # gitignored
│   └── parallel_training-<JOB>_<TASK>.{out,err}
└── results/                      # gitignored
    └── results_job_<JOB>_task_<TASK>.tar.gz
```

## SLURM cheat sheet

```bash
squeue -u $USER                    # all your jobs
squeue -j 12345678                 # one job
sacct -j 12345678                  # accounting

scancel 12345678                   # cancel
scancel 12345678_5                 # cancel array task 5
scancel -u $USER                   # nuke all of yours

tail -f logs/parallel_training-12345678_0.out
```

## Environment

The auto-generated SLURM script `source`s `.env` before running (via
`set -a; source .env; set +a`), so put cluster-specific variables there:

```
DATASET_ROOT=/home/.../links/projects/def-mlecuyer/.../dataset
VERSION=1.3
NUM_TILEGROUPS=5
MODEL_DIR=models
```

The script extracts whichever of these archives is present in `$DATASET_ROOT`:
- `compressed_dataset.tar.gz`
- `small-data.tar.gz`

Adjust `submit_experiments.py::create_slurm_script` if you need different names.

## Tips

- **Smoke-test first**: `python scripts/submit_experiments.py --experiment_indices "0" --time "1:00:00"`.
- **Memory**: ~20G/CPU works for `batch_size=8`; bump to 32G+ for 16+.
- **Time**: add a 20–30% buffer over expected runtime; cluster ceiling is typically 7d.
- **Fair share**: keep `--max_parallel` modest if you're hitting `QOSMaxJobsPerUserLimit`.

## Troubleshooting

**Job pending forever** → `squeue -u $USER`, look at the reason column. Usually
fair-share limits or resources.

**OOM** → drop `batch_size` in the experiment's config, or raise `--mem_per_cpu`.

**Job dies immediately** → `cat logs/parallel_training-*.err`. Common culprits:
missing dataset archive in `$DATASET_ROOT`, broken `.env`, import errors.

**Slow data loading** → make sure the dataset is being extracted to
`$SLURM_TMPDIR`, set `use_memmap=True` for large datasets, raise `--cpus_per_task`.
