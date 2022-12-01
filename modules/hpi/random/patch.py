from typing import Tuple, Union, Optional
import numpy as np
from skimage.util import view_as_windows


def random_2d(rows: int, cols: int):
    """Generate 2-dimensional random values."""
    return (np.random.randint(rows), np.random.randint(cols))


def sample(img: np.ndarray, shape: Union[Tuple[int, int], int], step: int = 1, mask: Optional[np.ndarray] = None,
           n_samples: int = 1, multichannel: bool = False, filled_ratio: float = 0.5):
    """Random sampling of patces in a image.
    
    Paramters
    ---------
        img: (N, M, C) ndarray,
            image to sample.
        shape: Tuple[int, int] or int,
            shape of patch.
        step: int,
            step size for sampling.
        mask: (N, M) ndarray,
            mask image for sampling.
        n_samples: int,
            number of patches to sample.
        multichannel: bool,
            If img has channel, multichannel must be True.
        filled_ratio: float, [0, 1]
            Filled ratio of mask in a patch to sample.
            Patches with mask area ratio of `filled_ratio` or more were sampled.
    
    Returns
    -------
        patches: (I, J, h_shape, w_shape, C) views of ndarray,
            Where I and J are identifier of patches,
            h_shape and w_shape are shape of a patch,
            C is number of channel.
        patches_mask: (I, J, h_shape, w_shape) views of ndarray,
            patches of given mask.
    """
    if isinstance(shape, int):
        shape = (shape, shape)
    if filled_ratio < 0:
        raise ValueError(f"`filled_ratio` must be range in [0, 1], but {filled_ratio} was given.")
    if mask is None:
        mask = np.zeros(img.shape[:2], dtype=bool)

    view_mask = view_as_windows(mask, window_shape=shape, step=step)
    view = np.squeeze(view_as_windows(img, window_shape=(*shape, img.shape[-1]) if multichannel else shape, step=step))

    patch_area = shape[0] * shape[1]
    patches = np.zeros(
        shape=(n_samples, *shape, img.shape[-1]) if multichannel else (n_samples, *shape),
        dtype=img.dtype
    )
    patches_mask = np.zeros(shape=(n_samples, *shape), dtype=mask.dtype)
    cnt = 0
    while cnt < n_samples:
        r, c = random_2d(view_mask.shape[0], view_mask.shape[1])
        _pmask = view_mask[r, c]
        if (_pmask.sum() / patch_area) > filled_ratio:
            patches[cnt] = view[r, c]
            patches_mask[cnt] = _pmask
            cnt += 1
    return (patches, patches_mask)