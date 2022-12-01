
from typing import Tuple, Optional
import numpy as np

def _convert_tile_size(tile_size: int, base_level: int, target_level: int) -> int:
    """Convert tile_size."""
    return int(tile_size * np.power(2, target_level - base_level))
    
    
def _address2pixel(address: float, tile_size: int,
                   base_level: int, target_level: Optional[int] = None) -> int:
    """Convert address of DeepZoom object into pixel."""
    return int(address * _convert_tile_size(tile_size, base_level, target_level))


def address2pixel(addresses: Tuple[float, ...], tile_size: int,
                  base_level: int, target_level: Optional[int] = None) -> Tuple[int, int]:
    """Convert addresses of DeepZoom object into pixels.
    
    NOTE: overlap must be 0.
    
    Parameters
    ----------
        address: Tuple[float, ...],
            addresses of DeepZoom object.
        tile_size: int,
            tize of a tile.
        base_level: int,
            base level of DeepZoom.
        target_level: Optinal[int],
            target level of DeepZoom.
    
    Returns
    -------
        pixels: Tuple[int, ...]
    """
    return tuple(_address2pixel(pos, tile_size, base_level, target_level) for pos in addresses)


def convert_level_dz2ndp(level_max_dz: int, level: int):
    """Convert level from DeepZoom into NDP."""
    return level_max_dz - level