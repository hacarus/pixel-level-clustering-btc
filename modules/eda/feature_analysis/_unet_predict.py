"""_unet_predict.py"""
import numpy as np
from warnings import warn
from skimage import util
from tqdm import tqdm
from itertools import product
from ._util import _check_overlap, _check_tilesize


def unet_tilewise(model, img: np.ndarray, size: int, overlap: float = 0.25,
                  batch_size: int = 128) -> np.ndarray:
    """Tilewise prediction by Unet.

    Parameters
    ----------
        model: Model of tf.keras
        img: (N, M, C) ndarray,
            image as input for the model.
        size: int,
            tile size.
        overlap: float, [0, 1]
            Fraction of overlap.
        batch_size: int,
            size of mini-batch

    Returns
    -------
        pred: (N, M) ndarray
            prediction mask.
    """
    if _check_overlap(overlap):
        raise ValueError("overlap must be between [0, 1].")
    if (_check_tilesize(size, img.shape[0]) or
        _check_tilesize(size, img.shape[1])):
        warn("shape of label % size have to be 0.")
    if size % 2 != 0:
        raise ValueError("size must be even number")

    wshape = (size, size, 3)
    step = int(size * (1 - overlap))
    views = np.squeeze(util.view_as_windows(img, window_shape=wshape, step=step))
    pred_wsi = np.zeros(shape=(img.shape[:2]), dtype=np.float16)
    views_pred = np.squeeze(util.view_as_windows(pred_wsi, window_shape=wshape[:2], step=step))

    nrows = views.shape[0]
    ncols = views.shape[1]
    bar = tqdm(total=nrows * ncols)
    prod = product(range(nrows), range(ncols))
    combs = []
    for i, p in enumerate(prod):
        combs.append(p)
        if (i + 1) % batch_size == 0:
            rcs = np.array(combs).T
            pred = model.predict(views[rcs[0], rcs[1]] / 255.)
            pred = np.squeeze(pred)
            views_pred[rcs[0], rcs[1]] = pred
            combs = []
            bar.update(batch_size)

    rcs = np.array(combs).T
    pred = model.predict(views[rcs[0], rcs[1]] / 255.)
    pred = np.squeeze(pred)
    views_pred[rcs[0], rcs[1]] = pred
    return pred_wsi