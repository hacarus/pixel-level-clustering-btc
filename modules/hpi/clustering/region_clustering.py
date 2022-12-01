from typing import Optional, Tuple, Union
import numpy as np
from skimage.util import view_as_blocks
from skimage.feature import multiscale_basic_features
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from hpi.random import sample


def _tile_size(tile_size: Union[Tuple[int, int], int]):
    if isinstance(tile_size, int):
        return (tile_size, tile_size)
    return tile_size


def train_region_clustering(
    img: np.ndarray,
    clustering_pipeline: Pipeline,
    mask: Optional[np.ndarray] = None,
    tile_size: Union[Tuple[int, int], int] = 256,
    n_patches: int = 1000,
    multichannel: bool = True,
    filled_ratio: float = 0.9,
    n_samples: int = 1000000,
    random_state: Optional[int] = None,
    **params_multiscale_basic_features,
):
    """Region clustering.

    Parameters
    ----------
        img: (N, M, C) ndarray,
            image to clustering.
        clustering_pipeline: Pipeline,
            pre-trained Pipeline object.
        mask: (N, M) ndarray, Optional,
            mask to extract features.
        tile_size: Tuple[int, int] or int,
            tile size when processing.
        n_patches: int,
            number of patches to sample. Default = 1000.
        multichannel: bool,
            If img has channel, multichannel must be True.
        filled_ratio: float, [0, 1]
            Filled ratio of mask in a patch to sample. Default = 0.9.
            Patches with mask area ratio of `filled_ratio` or more were sampled.
        n_samples: int,
            Number of features for training. Default = 1000000.
        random_state: int,
        **params_multiscale_basic_features: Dict,
            parameters for multiscale_basic_features.
            
    Returns
    -------
        clustering_pipeline: Pipeline,
            trained pipeline.
    """
    if random_state is not None:
        np.random.seed(random_state)
    img = img.copy(order="C")
    if mask is None:
        mask = np.ones(img.shape[:2], order="C").astype(bool)
    else:
        mask = mask.copy(order="C")
    block_shape = _tile_size(tile_size)
    params_mbf = dict(
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=1,
        sigma_max=10,
        multichannel=multichannel,
    )

    if params_multiscale_basic_features is not None:
        params_mbf.update(params_multiscale_basic_features)

    patches, patches_mask = sample(img, shape=block_shape, mask=mask, n_samples=n_patches,
                                                 multichannel=multichannel, filled_ratio=filled_ratio)
    X = np.vstack([multiscale_basic_features(_img, **params_mbf)[_mask] for _img, _mask in zip(patches, patches_mask)])

    inds = np.random.choice(range(X.shape[0]), size=n_samples)
    clustering_pipeline.fit(X[inds])

    return clustering_pipeline


def predict_region_clustering(
    img: np.ndarray,
    clustering_pipeline: Pipeline,
    mask: Optional[np.ndarray] = None,
    tile_size: Union[Tuple[int, int], int] = 256,
    label_offset: int = 0,
    **params_multiscale_basic_features,
):
    """Region clustering.

    Parameters
    ----------
        img: (N, M, C) ndarray,
            image to clustering.
        clustering_pipeline: Pipeline,
            pre-trained Pipeline object.
        mask: (N, M) ndarray,
        tile_size: Tuple[int, int] or int,
            tile size when processing.
        label_offset: int,
            offset of labeling. Default = 0
        **params_multiscale_basic_features: Dict,
            parameters for multiscale_basic_features.
            
    Returns
    -------
        klabels: (N, M) ndarray,
            labeled image by clustering.
    """
    img = img.copy(order="C")
    klabels = np.zeros(img.shape[:2], dtype=np.int32)
    block_shape = _tile_size(tile_size)
    params_mbf = dict(
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=1,
        sigma_max=10,
        multichannel=True,
    )
    if params_multiscale_basic_features is not None:
        params_mbf.update(params_multiscale_basic_features)

    views_img = np.squeeze(view_as_blocks(img, block_shape=(*block_shape, img.shape[2])))
    views_klabel = view_as_blocks(klabels, block_shape=block_shape)
    if mask is None:
        views_mask = np.squeeze(view_as_blocks(np.ones_like(img).astype(bool),
                                               block_shape=block_shape))
    else:
        mask = mask.copy(order="C")
        views_mask = np.squeeze(view_as_blocks(mask, block_shape=block_shape))

    for i in tqdm(range(views_mask.shape[0])):
        for j in range(views_mask.shape[1]):
            if (mask is not None) and (views_mask[i, j].sum() == 0):
                continue

            _X = multiscale_basic_features(
                views_img[i, j], **params_mbf)[views_mask[i, j]]

            pred = clustering_pipeline.predict(_X) + label_offset
            views_klabel[i, j, views_mask[i, j]] = pred

    return klabels
