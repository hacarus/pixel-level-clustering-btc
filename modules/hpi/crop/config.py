"""config.py."""
from typing import Tuple, Dict, Sequence, Optional
import numpy as np


def generate_config(level: int, tile_size: int, addresses: Optional[Sequence[Tuple[float, float]]] = None):
    return dict(level=level, tile_size=tile_size, addresses=addresses)


def update_config(config: Dict, level: int):
    ldiff = np.power(2, float(level - config["level"]))
    config.update(dict(level=level))
    config.update(dict(addresses=[tuple((np.array(ad) * ldiff).astype(int)) for ad in config["addresses"]]))
    return config