"""io.py."""
from pathlib import Path
import xml.etree.ElementTree as ET 
import openslide
from .ndpi import NDPI
from .ndpa import NDPA
from .utils import parse_ndpa


def load_ndpi(path: Path) -> NDPI:
    """Load ndpi file.

    Parameters
    ----------
        path: Path,
            path to ndpi file.

    Returns
    -------
        NDPI object.
    """
    slide = openslide.OpenSlide(str(path))
    return NDPI(slide)


def load_ndpa(path: Path) -> NDPA:
    """Load ndpa file.

    Parameters
    ----------
        path: Path,
            path to ndpa file.

    Returns
    -------
        NDPA object.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    return NDPA(parse_ndpa(root))
