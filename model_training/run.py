import logging
import os
import re
import sys
import argparse
import debugpy
from dotenv import load_dotenv
import json
import subprocess

from src.training.trainer_config import TrainerConfig
from utils import (
    load_experiments,
    save_experiments,
    train,
    get_multi_experiment_overrides,
)
from src.training.experiments_registry import REGISTRY
from src.logging_utils import setup_logging
from scripts.submit_experiments import create_slurm_script, _GPU_SLURM_DIRECTIVES, DEFAULT_DATA_ARCHIVE

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)
NUM_TILEGROUPS = int(os.environ.get("NUM_TILEGROUPS", "5"))


def run(
    output_dir=None,
    experiments_dir="experiments",
    keywords=None,
    slurm=False,
    slurm_worker=False,
    experiment_dir=None,
    data_path=None,
    max_parallel=4,
    time_limit="12:00:00",
    mem_per_cpu="20G",
    cpus_per_task=1,
    account="def-mlecuyer",
    gpu_config=None,
    generate_only=False,
    dry_run=False,
    experiment_indices=None,
    prepare_only=False,
    run_only=False,
    batch_size=4,
    num_workers=None,
    data_archive=DEFAULT_DATA_ARCHIVE,
    cache_dir=None,
    smoke_test=False,
):
    if prepare_only and run_only:
        logger.error("✗ --prepare_only and --run_only are mutually exclusive")
        return 1

    if slurm and smoke_test:
        logger.error("✗ --smoke_test runs locally; don't combine it with --slurm")
        return 1

    if cpus_per_task is not None:
        if num_workers is None:
            num_workers = cpus_per_task

        elif num_workers > cpus_per_task:
            logger.error(f"✗ --num_workers ({num_workers}) must be <= --cpus_per_task ({cpus_per_task})")
            return 1

    if slurm_worker:
        if experiment_dir is None:
            logger.error("✗ --slurm_worker requires --experiment_dir")
            return 1

        exp_root = os.path.dirname(experiment_dir)
        exp_name = os.path.basename(experiment_dir)
        match = re.match(r"experiment_(\d+)$", exp_name)
        if not match:
            logger.error(f"✗ Cannot parse experiment index from '{exp_name}'")
            return 1
        exp_idx = int(match.group(1))

        config_dict = load_experiments(exp_root)
        if config_dict == 1:
            return 1

        if exp_idx not in config_dict:
            logger.error(f"✗ Experiment {exp_idx} not found after loading '{exp_root}'")
            return 1
        single = {exp_idx: config_dict[exp_idx]}

        return train(
            experiments_dir=exp_root,
            matching_path="predefined_experiments_matching.json",
            config_dict=single,
            output_dir=output_dir,
            data_path=data_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            batch_size=batch_size,
            smoke_test=smoke_test,
        )

    if not run_only:
        logger.info("=" * 10)
        logger.info("Creating directories...")
        logger.info("=" * 10)

        if output_dir is None:
            logger.warning("'output_dir' not provided. Results won't be saved")
        else:
            try:
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"✓ Successfully created {output_dir}")
            except Exception as e:
                logger.error(f"  Error: {e}")
                logger.error(f"✗ Failed creating {output_dir}")
                return 1

        if experiments_dir is None:
            logger.warning("'experiments_dir' not provided")
            logger.warning("Default 'experiments_dir' is set to 'experiments'")
            experiments_dir = "experiments"

        try:
            os.makedirs(experiments_dir, exist_ok=True)
            logger.info(f"✓ Successfully created {experiments_dir}")
        except Exception as e:
            logger.error(f"  Error: {e}")
            logger.error(f"✗ Failed creating {experiments_dir}")
            return 1

        logger.info("=" * 10)
        logger.info("Keywords verification...")
        logger.info("=" * 10)

        if keywords is None:
            logger.warning("no keyword specified")
        elif not isinstance(keywords, list):
            logger.error(f"✗ 'keywords' is not a list: {keywords}")
            return 1
        else:
            existing_keywords = list(REGISTRY.keys())
            for keyword in keywords:
                if keyword not in existing_keywords:
                    logger.error(f"✗ keyword '{keyword}' is not a predefined keyword")
                    logger.error(f"Keywords must be taken from: {existing_keywords}")
                    return 1

        logger.info("✓ Keywords Successfully verified")

        logger.info("=" * 10)
        logger.info("Predefined experiments preparation...")
        logger.info("=" * 10)

        keyword_map = {}
        try:
            for kw in keywords:
                overrides_dict = get_multi_experiment_overrides(kw, nb_exp=NUM_TILEGROUPS)
                if len(overrides_dict) > 1:
                    for i, override in overrides_dict.items():
                        sub_keyword = f"{kw}_{i}"
                        config = TrainerConfig.get_predefined_config(kw, override=override)
                        keyword_map[sub_keyword] = config
                        logger.info(f"✓ Successfully created sub-keyword {sub_keyword}")
                    logger.info(f"✓ Successfully handled multi-experiment {kw}: {len(overrides_dict)} created")
                else:
                    config = TrainerConfig.get_predefined_config(kw, override=overrides_dict[0])
                    keyword_map[kw] = config
                    logger.info(f"✓ Successfully handled single-experiment {kw}")
        except Exception as e:
            logger.error(f"  Error: {e}")
            logger.error("✗ Failed creating keyword map")
            return 1

        logger.info("=" * 10)
        logger.info("Predefined experiments saving...")
        logger.info("=" * 10)

        try:
            save_exp = save_experiments(
                root_dir=experiments_dir,
                configs=keyword_map,
                matching_path="predefined_experiments_matching.json",
            )
            if save_exp == 0:
                logger.info(f"✓ Successfully saved {len(keyword_map)} experiments in {experiments_dir}")
                if prepare_only:
                    return 0
            else:
                logger.error(f"✗ Failed to save predefined experiments in {experiments_dir}")
                return 1
        except Exception as e:
            logger.error(f"  Error: {e}")
            logger.error(f"✗ Failed to save predefined experiments in {experiments_dir}")
            return 1

        num_experiments = len(keyword_map)

    if slurm:
        if run_only:
            pattern = re.compile(r"experiment_(\d+)$")
            num_experiments = sum(
                1 for item in os.listdir(experiments_dir)
                if pattern.match(item)
            )

        logger.info("=" * 10)
        logger.info("SLURM submission...")
        logger.info("=" * 10)

        os.makedirs("logs", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        os.makedirs("scripts", exist_ok=True)

        script_path = create_slurm_script(
            num_experiments=num_experiments,
            max_parallel=max_parallel,
            time_limit=time_limit,
            mem_per_cpu=mem_per_cpu,
            cpus_per_task=cpus_per_task,
            account=account,
            output_path="scripts/slurm_array_job_generated.sh",
            source_dir=os.getcwd(),
            gpu_config=gpu_config,
            experiment_indices=experiment_indices,
            num_workers=num_workers,
            batch_size=batch_size,
            data_archive=data_archive,
            experiments_dir=experiments_dir,
        )

        logger.info(f"✓ Script created: {script_path}")
        logger.info("Job configuration:")
        logger.info(f"  Account      : {account}")
        logger.info(f"  Time limit   : {time_limit}")
        logger.info(f"  Mem per CPU  : {mem_per_cpu}")
        logger.info(f"  CPUs per task: {cpus_per_task}")
        logger.info(f"  Max parallel : {max_parallel}")
        if gpu_config:
            logger.info(f"  GPU config   : {gpu_config}  ({_GPU_SLURM_DIRECTIVES[gpu_config].lstrip('#SBATCH ')})")

        if generate_only:
            logger.info("--generate_only set. Skipping execution.")
            logger.info("To submit manually: sbatch scripts/slurm_array_job_generated.sh")
            return 0

        if dry_run:
            logger.info(f"--dry_run set. Would run: sbatch {script_path}")
            return 0

        logger.info("Submitting to SLURM...")
        try:
            result = subprocess.run(
                ["sbatch", script_path],
                capture_output=True, text=True, check=True,
            )
            logger.info(result.stdout)
            if "Submitted batch job" in result.stdout:
                job_id = result.stdout.split()[-1]
                logger.info(f"✓ Submitted job array: {job_id}")
                logger.info("Monitor: squeue -u $USER")
                logger.info("Logs in: logs/")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ sbatch error: {e.stderr}")
            return 1
        except FileNotFoundError:
            logger.error("✗ sbatch not found. Are you on a SLURM cluster?")
            return 1
        return 0

    logger.info("=" * 10)
    logger.info("Predefined experiments reading...")
    logger.info("=" * 10)

    try:
        loaded = load_experiments(experiments_dir)
        if loaded == 1:
            logger.error("✗ Failed reading predefined experiments")
            return 1
        logger.info(f"✓ Successfully loaded {len(loaded)} configurations.")
    except Exception as e:
        logger.error(f"  Error: {e}")
        logger.error("✗ Failed reading predefined experiments")
        return 1

    logger.info("=" * 10)
    logger.info("Training (predefined experiments)...")
    logger.info("=" * 10)

    if loaded:
        try:
            result = train(
                experiments_dir=experiments_dir,
                matching_path="predefined_experiments_matching.json",
                config_dict=loaded,
                output_dir=output_dir,
                data_path=data_path,
                cache_dir=cache_dir,
                num_workers=num_workers,
                batch_size=batch_size if batch_size != 4 else None,
                smoke_test=smoke_test,
            )
            if result == 1:
                logger.error("✗ Training failed")
                return 1
            logger.info("✓ Training of every predefined experiment completed successfully")
        except Exception as e:
            logger.error(f"  Error: {e}")
            logger.error("✗ Training failed")
            return 1
    else:
        logger.warning("Training cancelled: no predefined experiments found")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch training experiments")

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--experiments_dir", type=str, default="experiments")
    parser.add_argument("--keywords", type=str, default=None,
                        help="Comma-separated list of experiment keywords")

    # SLURM mode flags
    parser.add_argument("--slurm", action="store_true",
                        help="Submit as SLURM array job instead of running locally")
    parser.add_argument("--slurm_worker", action="store_true",
                        help="Internal flag — called by SLURM tasks, do not use manually")
    parser.add_argument("--experiment_dir", type=str, default=None,
                        help="(--slurm_worker) Path to a single experiment_{i} directory")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Override data path (slurm worker or local)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Override cache dir (slurm worker or local)")

    # SLURM job options
    parser.add_argument("--max_parallel", type=int, default=4)
    parser.add_argument("--time", type=str, default="12:00:00")
    parser.add_argument("--mem_per_cpu", type=str, default="20G")
    parser.add_argument("--cpus_per_task", type=int, default=None)
    parser.add_argument("--account", type=str, default="def-mlecuyer")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        choices=list(_GPU_SLURM_DIRECTIVES.keys()),
        help=(
            "GPU configuration to request. Options:\n"
            "  h100:1       — one full H100-80gb\n"
            "  h100:2/3/4   — 2-4 full H100s on one node\n"
            "  1g.10gb      — MIG 1/8 slice, 10 GB\n"
            "  2g.20gb      — MIG 2/8 slice, 20 GB\n"
            "  3g.40gb      — MIG 3/8 slice, 40 GB\n"
            "Omit to request no GPU."
        ),
    )
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--experiment_indices", type=str, default=None,
                        help="Comma-separated indices to run (e.g. '0,2,5')")
    parser.add_argument("--prepare_only", action="store_true",
                        help="Save experiments to disk without training.")
    parser.add_argument("--run_only", action="store_true",
                        help="Read saved experiments from experiments_dir and train them.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of dataloader workers. Defaults to cpus_per_task. Must be <= cpus_per_task.")
    parser.add_argument("--data_archive", type=str, default=DEFAULT_DATA_ARCHIVE,
                        help="Dataset archive filename under $DATASET_ROOT")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Sanity-check run: 1 epoch on a few datapoints, evals skipped. "
                             "Use locally to validate the pipeline before submitting to SLURM.")

    args = parser.parse_args()

    logger.info("=" * 10)
    logger.info("Starting run.py")
    logger.info("=" * 10)

    if args.debug:
        debugpy.listen(("0.0.0.0", 5678))
        logger.info("Waiting for debugger...")
        debugpy.wait_for_client()
        logger.info("Debugger attached!")

    keywords = None
    if args.keywords:
        try:
            keywords = [k.strip() for k in args.keywords.split(",")]
        except Exception as e:
            logger.error(f"✗ Failed parsing --keywords: {e}")

    result = run(
        output_dir=args.output_dir,
        experiments_dir=args.experiments_dir,
        keywords=keywords,
        slurm=args.slurm,
        slurm_worker=args.slurm_worker,
        experiment_dir=args.experiment_dir,
        data_path=args.data_path,
        max_parallel=args.max_parallel,
        time_limit=args.time,
        mem_per_cpu=args.mem_per_cpu,
        cpus_per_task=args.cpus_per_task,
        account=args.account,
        gpu_config=args.gpu,
        generate_only=args.generate_only,
        dry_run=args.dry_run,
        experiment_indices=args.experiment_indices,
        prepare_only=args.prepare_only,
        run_only=args.run_only,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_archive=args.data_archive,
        cache_dir=args.cache_dir,
        smoke_test=args.smoke_test,
    )

    if result == 0:
        logger.info("✓ run.py terminated successfully")
    else:
        logger.error("✗ run.py failed")
    sys.exit(result)