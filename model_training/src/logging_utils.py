import logging
import sys


def setup_logging(level=logging.INFO):
    """Configure the root logger to write to stdout.

    Logging must go to stdout (not the default stderr): SLURM splits the
    streams into separate .out/.err files, so logger output on stderr never
    shows up next to the prints in the .out log. force=True overrides any
    handler a library installed first.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
