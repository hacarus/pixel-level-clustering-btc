
"""_watershed.py."""
from warnings import warn
from itertools import product
from tqdm import tqdm
import numpy as np
from skimage import util
from ._util import _check_tilesize

from hpi.features import elementwise_shannon_entropy


def shannon_entropy_recursive(img, tile_size: int, *, kernel_size: int,
                              multichannel: bool = True) -> np.ndarray:
    """Recursive extraction of shannon entropy.

    Parameters
    ----------
    img: (N, M, [, C]) np.ndarray
    tile_size : int
    margin: int
    kernel_size: int
        kernel size to calculate shannon entropy.
    overlap: float, optional
        by default 0.25
    multichannel: bool, optional
        by default True

    Returns
    -------
    shannon_entropy: np.ndarray
        pixel-wise shannon entropy
    """
    if (_check_tilesize(tile_size, img.shape[0])
       or _check_tilesize(tile_size, img.shape[1])):
        warn("shape of label % tile_size have to be 0.")

    # to avoid border artifact by watershed.
    step = tile_size
    margin = tile_size // 2
    wshape_img = (tile_size + 2 * margin, tile_size + 2 * margin)

    if multichannel:
        wshape_img = (*wshape_img, img.shape[2])

    shannon_buffer = np.zeros(shape=img.shape, dtype=np.float16)
    views = np.squeeze(
        util.view_as_windows(img, window_shape=wshape_img, step=step)
    )
    views_entr = np.squeeze(
        util.view_as_windows(shannon_buffer, window_shape=wshape_img, step=step)
    )

    nrows = views.shape[0]
    ncols = views.shape[1]
    bar = tqdm(total=nrows * ncols)
    prod = product(range(nrows), range(ncols))
    for p in prod:
        if views[p[0], p[1]].sum() == 0:
            bar.update(1)
            continue
        _patch_inv = util.invert(views[p[0], p[1]])
        se = elementwise_shannon_entropy(_patch_inv, kernel_size,
                                         multichannel=multichannel)

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
            rmax = se.shape[0]
        if p[1] == (ncols - 1):
            cmax = se.shape[1]

        views_entr[p[0], p[1]][rmin:rmax, cmin:cmax] = \
            se[rmin:rmax, cmin:cmax]   # avoid border artifact
        bar.update(1)

    return shannon_buffer
