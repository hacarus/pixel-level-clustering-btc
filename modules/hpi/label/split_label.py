from typing import Tuple, Optional
import numpy as np


def split(img_label: np.ndarray, intensity_image: Optional[np.ndarray] = None) -> Tuple:
    """Convert from labels to splitted masks.
    
    Parameters
    ----------
        label: (N, M) ndarray,
            labeled image.
        intensity_image: (N, M, C) np.ndarray, Optional,
            intensity image, where C is number of channels.
    
    Returns
    -------
        (masks[, img_splitted]): tuple,
            masks: (K, N, M) ndarray,
                masks for each label.
            img_split: (K, N, M, C) ndarray,
                splitted intensity image.
            
    """ 
    unique_labels = np.unique(img_label)
    masks = np.array([img_label == i for i in unique_labels])
    if intensity_image is None:
        return masks,

    img_split = np.zeros((masks.shsape[0], *intensity_image.shape), dtype=intensity_image.dtype)
    for i, mask in enumerate(masks):
        img_split[i] = intensity_image * mask
    return masks, img_split