#!/usr/bin/env python3
"""
Shared utilities for SLURM job submission scripts.
"""

import os
import subprocess

DEFAULT_DATA_ARCHIVE = "compressed_dataset.tar.gz"

_GPU_SLURM_DIRECTIVES = {
    "h100:1":  "#SBATCH --gpus=h100:1",
    "h100:2":  "#SBATCH --gpus-per-node=h100:2",
    "h100:3":  "#SBATCH --gpus-per-node=h100:3",
    "h100:4":  "#SBATCH --gpus-per-node=h100:4",
    "1g.10gb": "#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1",
    "2g.20gb": "#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1",
    "3g.40gb": "#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1",
}


def _tar_flags(archive_name: str) -> str:
    if archive_name.endswith(".tar.gz") or archive_name.endswith(".tgz"):
        return "xzf"
    return "xf"


def _dataset_extraction_block(data_archive: str) -> str:
    if data_archive == DEFAULT_DATA_ARCHIVE:
        return """\
# Extract dataset
echo "Extracting dataset..."

_ARCHIVE="$DATASET_ROOT/compressed_dataset.tar.gz"
_SENTINEL="$DATASET_ROOT/.archive_ok"
_LOCK="$DATASET_ROOT/.archive_lock"

# If the archive is missing or the sentinel (written after a successful creation)
# is absent, the archive is new, incomplete, or corrupted — (re)create it.
# flock ensures only one array task creates it; all others wait up to 2 hours.
if [ ! -f "$_ARCHIVE" ] || [ ! -f "$_SENTINEL" ]; then
    echo "Archive missing or not yet validated. Acquiring lock to (re)create..."
    (
        flock -x -w 7200 200
        if [ ! -f "$_ARCHIVE" ] || [ ! -f "$_SENTINEL" ]; then
            echo "Creating compressed_dataset.tar.gz from $DATASET_ROOT (may take a while)..."
            rm -f "$_ARCHIVE" "$_SENTINEL"
            tar czf "$_ARCHIVE.tmp" \\
                --exclude="$DATASET_ROOT/compressed_dataset.tar.gz" \\
                --exclude="$DATASET_ROOT/compressed_dataset.tar.gz.tmp" \\
                --exclude="$DATASET_ROOT/.archive_ok" \\
                --exclude="$DATASET_ROOT/.archive_lock" \\
                "$DATASET_ROOT/"
            mv "$_ARCHIVE.tmp" "$_ARCHIVE"
            touch "$_SENTINEL"
            echo "Archive created and validated: $_ARCHIVE"
        else
            echo "Archive was built by another task while waiting for lock."
        fi
    ) 200>"$_LOCK"
fi

_extract_archive() {
    local archive="$1"
    local target="$2"
    # Auto-detect strip count from the archive's actual path structure so that
    # externally-created archives (relative paths like X/X_1.tif) and our own
    # (absolute paths like project/.../hedgementation_1.3/X/X_1.tif) both work.
    # Formula: slashes_in_first_file_entry - 1 (keep only the category/file tail).
    local _first
    _first=$(tar tzf "$archive" 2>/dev/null | grep -v '/$' | head -1)
    local _slashes
    _slashes=$(echo "$_first" | tr -cd '/' | wc -c | tr -d '[:space:]')
    local _strip=$(( _slashes - 1 ))
    [ "$_strip" -lt 0 ] && _strip=0
    echo "Auto-detected STRIP_COMPONENTS=$_strip (first file entry: $_first)"
    tar xzf "$archive" --strip-components=$_strip -C "$target"
}

if [ -f "$_ARCHIVE" ]; then
    echo "Extracting dataset archive..."
    _extract_archive "$_ARCHIVE" "$SLURM_TMPDIR/data"
elif [ -f "$DATASET_ROOT/small-data.tar.gz" ]; then
    echo "Falling back to small-data.tar.gz..."
    _extract_archive "$DATASET_ROOT/small-data.tar.gz" "$SLURM_TMPDIR/data"
else
    echo "WARNING: No dataset archive found."
fi"""
    else:
        flags = _tar_flags(data_archive)
        return f"""\
# Extract dataset
echo "Extracting dataset..."
if [ ! -f "$DATASET_ROOT/{data_archive}" ]; then
    echo "ERROR: Dataset archive not found: $DATASET_ROOT/{data_archive}"
    exit 1
fi
echo "Extracting dataset archive..."
tar {flags} "$DATASET_ROOT/{data_archive}" -C $SLURM_TMPDIR/data"""


def python_env_block(source_dir: str, ssh_setup_code: str = "") -> str:
    """Return the bash block for loading modules, creating a venv, and installing deps."""
    return f"""\
# Setup Python environment
echo "Loading Python module..."
module load python/3.13
module load gcc opencv proj gdal

echo "Creating virtual environment..."
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

echo "Installing requirements..."
pip install --no-index --upgrade pip
{ssh_setup_code}pip install --no-index -r {source_dir}/requirements.txt
pip install --no-index {source_dir}/hedgementation_utils

# pyproj 3.7.2+computecanada links against PROJ 9.4.1 (requires database MINOR >= 3).
# The gdal module resets PROJ_LIB to 9.2.0 (MINOR=2), so we must locate the
# 9.4.1 data dir explicitly rather than relying on get_data_dir() / PROJ_LIB.
for _ARCH in x86-64-v4 x86-64-v3 x86-64-v2; do
    _CANDIDATE="/cvmfs/soft.computecanada.ca/easybuild/software/2023/${{_ARCH}}/Compiler/gcccore/proj/9.4.1/share/proj"
    if [ -f "$_CANDIDATE/proj.db" ]; then
        export PROJ_DATA="$_CANDIDATE"
        break
    fi
done
echo "PROJ_DATA set to: $PROJ_DATA\""""


def add_common_slurm_args(parser) -> None:
    """Add shared SLURM resource arguments to an argparse parser."""
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=4,
        help="Maximum number of tasks to run in parallel (default: 4)",
    )
    parser.add_argument(
        "--time",
        type=str,
        default="12:00:00",
        help="Time limit per job (default: 12:00:00)",
    )
    parser.add_argument(
        "--mem_per_cpu",
        type=str,
        default="20G",
        help="Memory per CPU (default: 20G)",
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=1,
        help="CPUs per task (default: 1)",
    )
    parser.add_argument(
        "--account",
        type=str,
        default="def-mlecuyer",
        help="SLURM account (default: def-mlecuyer)",
    )


def add_behavior_flags(parser) -> None:
    """Add shared behavior flags --generate_only and --dry_run to an argparse parser."""
    parser.add_argument(
        "--generate_only",
        action="store_true",
        help="Generate SLURM script but don't submit jobs",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be submitted without actually submitting",
    )


def submit_job(script_path: str) -> str | None:
    """Submit a SLURM script via sbatch. Returns the job ID string, or None on failure."""
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
        if "Submitted batch job" in result.stdout:
            return result.stdout.split()[-1]
        return None
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error submitting job:")
        print(e.stderr)
        return None
    except FileNotFoundError:
        print("\n✗ Error: sbatch command not found.")
        print("Are you on a SLURM cluster?")
        return None
