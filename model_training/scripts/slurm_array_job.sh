#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --mem-per-cpu=20G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=parallel_training
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --array=0-9%4  # Run experiments 0-9, max 4 at a time

# Usage: sbatch slurm_array_job.sh
# This script runs multiple training experiments in parallel using SLURM array jobs.
# Adjust --array parameter above to match your number of experiments.
# Format: --array=START-END%MAX_PARALLEL
# Example: --array=0-19%5 means 20 experiments, max 5 running simultaneously

set -e

echo "=========================================="
echo "Starting Job Array Task ${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "=========================================="

# Define paths
SOURCEDIR=home/nathansd/links/projects/def-mlecuyer/nathansd/model_training_from_scratch
EXPERIMENT_DIR=$SOURCEDIR/experiments
CONFIG_FILE=$EXPERIMENT_DIR/experiment_${SLURM_ARRAY_TASK_ID}_config.json

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Configuration file not found: $CONFIG_FILE"
    echo "Did you run: python scripts/experiment_configs.py"
    exit 1
fi

echo "Configuration file: $CONFIG_FILE"

# Setup Python environment
echo "Loading Python module..."
module load python/3.13

echo "Creating virtual environment..."
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Installing requirements..."
pip install --no-index --upgrade pip
pip install --no-index -r $SOURCEDIR/requirements.txt

# Setup data directories
echo "Setting up data directories..."
mkdir -p $SLURM_TMPDIR/data
mkdir -p $SLURM_TMPDIR/models

# Extract dataset
echo "Extracting dataset..."
if [ -f "$SOURCEDIR/data.tar.gz" ]; then
    tar xzf $SOURCEDIR/data.tar.gz -C $SLURM_TMPDIR/data
elif [ -f "$SOURCEDIR/small-data.tar.gz" ]; then
    tar xzf $SOURCEDIR/small-data.tar.gz -C $SLURM_TMPDIR/data
else
    echo "WARNING: No dataset archive found. Using existing data path."
fi

echo "Data directory contents: $(ls -la $SLURM_TMPDIR/data/ 2>/dev/null || echo 'empty')"

# Copy experiment metadata to temp directory
echo "Copying experiment metadata..."
METADATA_DIR=$EXPERIMENT_DIR/experiment_${SLURM_ARRAY_TASK_ID}/metadata
if [ -d "$METADATA_DIR" ]; then
    cp -r $METADATA_DIR $SLURM_TMPDIR/
fi

# Run the training script
echo "=========================================="
echo "Starting training for experiment ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

cd $SOURCEDIR

python3 -u scripts/run_single_experiment.py \
    --config_file "$CONFIG_FILE" \
    --data_path "$SLURM_TMPDIR/data" \
    --save_path "$SLURM_TMPDIR/models" \
    --metadata_dir "$SLURM_TMPDIR/metadata"

TRAIN_EXIT_CODE=$?

echo "Training exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "Training completed successfully"

# Archive results
echo "=========================================="
echo "Archiving results"
echo "=========================================="

JOB_ID=${SLURM_JOB_ID:-$$}
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ARCHIVE_NAME="results_job_${JOB_ID}_task_${TASK_ID}.tar.gz"

# Find the latest directory (the one just created)
LATEST_DIR=$(ls -td $SLURM_TMPDIR/models/*/ 2>/dev/null | head -1)
LATEST_DIR=${LATEST_DIR%/}

if [[ -d "${LATEST_DIR}" ]]; then
    echo "Archiving directory: ${LATEST_DIR}"
    tar czf "${ARCHIVE_NAME}" -C "$(dirname ${LATEST_DIR})" "$(basename ${LATEST_DIR})"

    # Get archive size
    ARCHIVE_SIZE=$(du -h "${ARCHIVE_NAME}" | cut -f1)
    echo "Archive created: ${ARCHIVE_NAME} (${ARCHIVE_SIZE})"
else
    echo "WARNING: Latest directory not found in $SLURM_TMPDIR/models/"
    echo "Available directories:"
    ls -la $SLURM_TMPDIR/models/ || echo "No models directory"
    exit 1
fi

# Transfer results (modify destination as needed)
echo "Transferring results..."

# Option 1: SCP to remote server (uncomment and modify if needed)
# REMOTE_USER=nathan
# REMOTE_HOST=aurora3.cs.ubc.ca
# REMOTE_PATH=/home/nathan/Desktop/model_training/remote_zips
# scp "${ARCHIVE_NAME}" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
# if [ $? -eq 0 ]; then
#     echo "Results transferred successfully"
#     rm "${ARCHIVE_NAME}"
# else
#     echo "WARNING: Transfer failed. Archive saved at: $(pwd)/${ARCHIVE_NAME}"
# fi

# Option 2: Copy to shared storage (recommended for Compute Canada)
RESULTS_DIR=$SOURCEDIR/results
mkdir -p $RESULTS_DIR
cp "${ARCHIVE_NAME}" "$RESULTS_DIR/"
if [ $? -eq 0 ]; then
    echo "Results copied to: $RESULTS_DIR/${ARCHIVE_NAME}"
    rm "${ARCHIVE_NAME}"
else
    echo "WARNING: Copy failed. Archive at: $(pwd)/${ARCHIVE_NAME}"
fi

echo "=========================================="
echo "Job Array Task ${SLURM_ARRAY_TASK_ID} Completed Successfully"
echo "=========================================="
