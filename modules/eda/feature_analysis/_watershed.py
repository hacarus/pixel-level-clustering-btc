"""_watershed.py."""
from warnings import warn
from itertools import product
from tqdm import tqdm
import numpy as np
import cv2
from skimage import util
from ._util import _check_tilesize


def _denoise(mask, kernel, iterations: int = 3) -> np.ndarray:
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            kernel, iterations=iterations).astype(np.uint8)
    
    
def _sure_bg(mask, kernel, iterations: int = 15) -> np.ndarray:
    return cv2.dilate(mask, kernel, iterations=iterations)
    

def _preprocess(labels: np.ndarray, label_offset: int = 2,
                kernel_size: int = 3) -> np.ndarray:
    """
    Parameters
    ----------
        labels: (N, M) ndarray,
            label image.
                0 indicates unknown, 1 indicates background,
        label_offset: int, default is 2.
            offset of label.
        kernel_size: int,
            kernel size for morphological opening.
    
    Returns
    -------
        new_mask: (N, M) ndarray,
            preprocecced mask.
    """
    _labels = labels.copy()
    mask = (_labels >= label_offset).astype(np.uint8) * 255

    # noise removal
    # To do: replace with np.unique
    kernel = np.ones((kernel_size, kernel_size), np.uint8) 
    _open = _denoise(mask, kernel, iterations=3)

    # compute sure background area 
    sure_bg = _sure_bg(_open, kernel, iterations=15)
    
    # assume sure foreground is equal to original
    sure_fg = mask.astype(np.uint8)
    
    # compute unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # mark unknown region as 0 
    _labels[unknown == 255] = 0
    
    return _labels


def watershed_recursive(label, intensity_image, tile_size: int):
    if (_check_tilesize(tile_size, label.shape[0]) or
        _check_tilesize(tile_size, label.shape[1])):
        warn("shape of label % tile_size have to be 0.")

    # to avoid border artifact by watershed.
    step = tile_size
    margin = tile_size // 2
    wshape_img = (tile_size + 2 * margin, tile_size + 2 * margin, 3)
    segments = np.zeros(shape=label.shape, dtype=label.dtype)
    views = np.squeeze(util.view_as_windows(
        intensity_image, window_shape=wshape_img, step=step))
    views_marker = np.squeeze(util.view_as_windows(
        label, window_shape=wshape_img[:2], step=step))
    views_segments = np.squeeze(util.view_as_windows(
        segments, window_shape=wshape_img[:2], step=step))

    nrows = views.shape[0]
    ncols = views.shape[1]
    bar = tqdm(total=nrows * ncols)
    prod = product(range(nrows), range(ncols))
    for p in prod:
        if views_marker[p[0], p[1]].sum() == 0:
            bar.update(1)
            continue
        labels = _preprocess(views_marker[p[0], p[1]])
        _segments = cv2.watershed(views[p[0], p[1]], labels)
        _segments[_segments == -1] = 0
        _segments[_segments == 1] = 0

        # avoid border artifact and fill padding.
        rmin = margin
        rmax = margin + tile_size
        cmin = margin
        cmax = margin + tile_size
        if p[0] == 0:
            rmin = 0
        if p[1] == 0:
            cmin = 0
        if p[0] == (nrows - 1):
            rmax = _segments.shape[0]
        if p[1] == (ncols - 1):
            cmax = _segments.shape[1]
        views_segments[p[0], p[1]][rmin:rmax, cmin:cmax] = \
            _segments[rmin:rmax, cmin:cmax]
        bar.update(1)

    return segments