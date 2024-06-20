import warnings
import logging
import loguru
from typing import Any, Generator
from rdkit import RDLogger
from contextlib import contextmanager


@contextmanager
def disable_warnings(disable: bool = True) -> Generator[Any, Any, Any]:
    """Disables warnings for given context."""

    if disable:
        action = "ignore"
        logging.disable(logging.WARNING)
        loguru.logger.disable("graphein")
        RDLogger.DisableLog("rdApp.*")
    else:
        action = "default"
        loguru.logger.enable("graphein")

    with warnings.catch_warnings(action=action):
        yield

    logging.disable(logging.NOTSET)
    loguru.logger.disable("graphein")  # Graphein logging is by default disabled.
    RDLogger.EnableLog("rdApp.*")
