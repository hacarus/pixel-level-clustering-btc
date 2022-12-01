"""contours.py"""
import numpy as np
from skimage import morphology


def get_contours(mask: np.ndarray, width: int = 1) -> np.ndarray:
    """Returns edge mask."""
    copied_mask = mask.copy()
    for i in range(width):
        copied_mask = morphology.erosion(copied_mask)
    contours = np.logical_xor(mask, copied_mask)
    return contours 
