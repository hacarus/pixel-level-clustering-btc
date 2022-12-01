from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler


def setup_logger(name: str, logfile: Path) -> logging.Logger:
    """Setup logger.

    Parameters
    ----------
    name : str
    logfile : Path

    Returns
    -------
    logging.Logger
    """
    logfile.parent.mkdir(mode=0o775, exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even DEBUG messages.
    fh = TimedRotatingFileHandler(logfile, when="midnight")
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s"
        " - %(message)s"
    )
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)
    if logger.hasHandlers():
        return logger

    # create console handler with a INFO log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s - %(message)s"
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)
    return logger
