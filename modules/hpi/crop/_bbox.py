"""bbox.py."""
from typing import Tuple, Sequence
import numpy as np


def _minmax_address(addresses: Sequence[Tuple[int, int]]):
    """Get bbox."""
    _adds = np.array(addresses)
    ad_xmin, ad_ymin = _adds.min(axis=0)
    ad_xmax, ad_ymax = _adds.max(axis=0) - 1
    return ad_xmin, ad_xmax, ad_ymin, ad_ymax