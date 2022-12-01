"""largest.py"""
import numpy as np
from skimage import measure


def largest_region(mask: np.ndarray):
    """Get the largest region from the mask.

    Parameters
    ----------
        mask: (N, M) ndarray,
        
    Returns
    -------
        mask: (N, M) ndarray,
            mask of largest area.
    """
    segments = measure.label(mask)
    _, counts = np.unique(segments, return_counts=True)
    # ignore label 0. 0 indicates background.
    mask_largeset = segments == (np.argmax(counts[1:]) + 1)
    return mask_largeset
