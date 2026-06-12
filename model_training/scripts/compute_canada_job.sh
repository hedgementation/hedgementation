#!/bin/bash
#SBATCH --account=def-mlecuyer
#SBATCH --mem-per-cpu=20G
#SBATCH --time=1:00:00
#SBATCH --job-name=model_training
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


set -e

echo "Starting Job"
module load python/3.13
SOURCEDIR=/home/nathansd/links/projects/def-mlecuyer/nathansd/model_training_from_scratch
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
echo "Created venv"

#pip install --no-index --upgrade pip
pip install --no-index -r $SOURCEDIR/requirements.txt
echo "Installed requirements"

mkdir $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/models
tar xf $SOURCEDIR/data/compressed_dataset.tar.gz -C $SLURM_TMPDIR
echo "Extracted data"

echo "Current working directory: $(pwd)"
echo "Checking extracted data: $(ls -la $SLURM_TMPDIR/data/hedgementation_dataset_1.2/)"

echo "starting python run"

chmod +x $SOURCEDIR/train.py

srun  python3 $SOURCEDIR/train.py\
    --normalization=minmax \
    --bucketization=uniform \
    --num_buckets=1 \
    --num_epochs=1 \
    --weighted \
    --data_path=$SLURM_TMPDIR/data \
    --save_path=$SLURM_TMPDIR/models \
    --metadata_paths='{"valid":"{}/pastis/one_batch_metadata.csv", "train":"{}/pastis/one_batch_metadata.csv"}'\

echo "completed python run"



echo "Zipping data"

JOB_ID=${SLURM_JOB_ID:-$$} 
ARCHIVE_NAME="results_job_${JOB_ID}.7z"

LATEST_DIR=$(ls -td $SLURM_TMPDIR/models/*/ | head -1)
LATEST_DIR=${LATEST_DIR%/}

echo "Latest directory found: ${LATEST_DIR}"

if [[ -d "${LATEST_DIR}" ]]; then
    7z a "${ARCHIVE_NAME}" "${LATEST_DIR}"
    echo "Data zipped"
else
    echo "Error: Latest directory not found"
    exit 1
fi


echo "sending data"

scp "${ARCHIVE_NAME}" nathan@aurora3.cs.ubc.ca:/home/nathan/Desktop/model_training/remote_zips
rm "${ARCHIVE_NAME}"

echo "data sent"

echo "Completed job successfully"