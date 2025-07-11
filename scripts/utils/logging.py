# configure logging for scripts
# logs should have millisecond-timestamp (human readable), level, and emitting module & line

import logging
import sys


def setup_logging(log_level=logging.INFO):
    """Setup logging configuration with timestamp, level, and module info"""
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d | %(levelname)s | %(module)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=log_level,
        stream=sys.stdout,
    )
