from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

from skimage import measure
import numpy as np
import pandas as pd


def _vectorize_properties(region, properties: Sequence[str]) -> np.ndarray:
    """Vectorize region properties.

    Parameters
    ----------
        region: skimage.measure._regionprops.RegionProperties
        properties: Sequence[str]
    """
    return np.hstack([np.asarray(getattr(region, nm)).ravel() for nm in properties])


def _propnames_flatten(region, properties: Sequence[str], sep: str = "-") -> List[str]:
    """Vectorize names of region property.

    Parameters
    ----------
        region: skimage.measure._regionprops.RegionProperties
        properties: Sequence[str]
    """
    nm_flatten_list: List[str] = []
    for nm in properties:
        prop = np.asarray(getattr(region, nm)).ravel()
        length = len(prop)
        if length > 1:
            _nm_seq = [nm + sep + str(i) for i in range(length)]
            nm_flatten_list.extend(_nm_seq)
        else:
            nm_flatten_list.append(nm)
    return nm_flatten_list
