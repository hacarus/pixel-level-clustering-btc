"""_transform.py."""
import numpy as np


def drop_margin(img: np.ndarray, margin: int):
    """Drop margin of given image."""
    return img[margin:-margin, margin:-margin]

def get_bbox(img: np.ndarray, bbox):
    """Get a patch from the image in the bounding box."""
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]