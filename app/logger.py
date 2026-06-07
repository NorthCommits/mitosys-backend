import logging
import sys

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | MITOSYS | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_FORMATTER)

_root = logging.getLogger("mitosys")
_root.setLevel(logging.INFO)
_root.addHandler(_handler)
_root.propagate = False


def get_logger(name: str = "mitosys") -> logging.Logger:
    return logging.getLogger(name)
