"""_densecrf.py."""
from typing import Tuple, Optional, Union
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as dutils 
from skimage import segmentation
from skimage.util import view_as_blocks
from tqdm import tqdm


def _check_type(img: np.ndarray):
    """Check value type of image."""
    if img.dtype == np.uint8:
        return False
    return True


def dcrf2d(img_label: np.ndarray, img: np.ndarray, gt_prob: float = 0.7, n_cluster: Optional[int] = None,
           zero_unsure: bool = True, stdxy_g: float = 1, compat_g: float = 10, stdxy_b: float = 10, stdrgb: float = 5,
           compat_b: float = 10, niter: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Apply DenseCRF to an image given. 
    
    Parameters
    ----------
        img_label : (N, M) ndarray
            Array which is a pixel level label, corresponding to an image. 
        img : (N, M, C) ndarray
            Multichannel image.
        gt_prob : float,
            Ranged in [0, 1], default = 0.7.
            Probability of ground truth labels.  
        n_cluster : Optional[int], default = None 
            Number of clusters. Only specify this if maximum number of classes is known. 
        zero_unsure: bool, default = False.
        stdxy_g: flaot,
            Standard deviation of (x, y) coordinates for gaussian kernel.
        compat_g: int
            compatibility for gaussian kernel.
        stdxy_b: flaot,
            Standard deviation of (x, y) coordinates for gaussian kernel.
        stdrgb: flaot,
            Standard deviation of RGB channels for gaussian kernel.
        compat_b: int
            compatibility for bilateral  kernel.
        niter: int,
            iteration number of inference. Default = 5.
        bg_label: int
            background label to make overlayed image.
            default = -1.
        
    Returns 
    -------
        MAP : (N, M) ndarray,
            Refined label image as a result of DenseCRF.  
        Q : (K, N, M) ndarray,
            pixelwise probability of the class.
    """
    if _check_type(img):
        raise TypeError(f"Type of img must be `uint8`, but {img.dtype} was given.")
    if not img.flags["C_CONTIGUOUS"]:
        img = np.ascontiguousarray(img, dtype=img.dtype)

    h, w = img_label.shape[:2]
    # if overall numer of clusters are not defined, then we assign unique labels within labels 
    RELABEL = False
    if n_cluster is None:
        img_label, _, inverse_map = segmentation.relabel_sequential(img_label)
        n_cluster = np.unique(img_label).shape[0]   # 0 indicates background
        if zero_unsure:
            n_cluster -= 1
        RELABEL = True
    
    d = dcrf.DenseCRF2D(w, h, n_cluster)

    # get unary potentials (neg log probability)
    U = dutils.unary_from_labels(img_label, n_cluster, gt_prob=gt_prob, zero_unsure=zero_unsure)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(stdxy_g, stdxy_g), compat=compat_g, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=(stdxy_b, stdxy_b),
                           srgb=(stdrgb, stdrgb, stdrgb),
                           rgbim=img,
                           compat=compat_b,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
    # Run n inference steps.
    # transpose axes from (class, h, w) to (h, w, class)
    Q = np.asarray(d.inference(niter)).reshape(-1, h, w).transpose(1, 2, 0)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=2)

    # relabel to original label
    if RELABEL:
        for label_in, label_out in zip(inverse_map.in_values[::-1], inverse_map.out_values[::-1]):
            MAP[MAP == label_in] = label_out

    return MAP


def _tile_size(tile_size: Union[Tuple[int, int], int]):
    if isinstance(tile_size, int):
        return (tile_size, tile_size)
    return tile_size


def dcrf2d_tilewise(
    img_label: np.ndarray,
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    tile_size: Union[Tuple[int, int], int] = 256,
    zero_unsure: bool = True,
    **dcrf_kwargs,
):
    """Apply DenseCRF to an image given. 
    
    Parameters
    ----------
        img_label : (N, M) ndarray
            Array which is a pixel level label, corresponding to an image. 
        img : (N, M, C) ndarray
            Array which represents 3 channel RGB image.
        mask: (N, M) ndarray,
            mask image of the target regions.
        tile_size: Tuple[int, int] or int,
            tile size of a patch.
        zero_unsure: bool, default = False.
        **dcrf_kwargs: Dict,
            keyword arguments for DenseCRF.
                gt_prob : float,
                    Ranged in [0, 1], default = 0.7.
                    Probability of ground truth labels.  
                n_cluster : Optional[int], default = None 
                    Number of clusters. Only specify this if maximum number of classes is known. 
                stdxy_g: flaot,
                    Standard deviation of (x, y) coordinates for gaussian kernel. Defalt = 1.
                compat_g: int
                    compatibility for gaussian kernel. Default = 10.
                stdxy_b: flaot,
                    Standard deviation of (x, y) coordinates for gaussian kernel. Default = 10.
                stdrgb: flaot,
                    Standard deviation of RGB channels for gaussian kernel. Default = 10.
                compat_b: int
                    compatibility for bilateral  kernel. Default = 5.
                niter: int,
                    iteration number of inference. Default = 50.
        
    Returns 
    -------
        MAP : (N, M) ndarray,
            Refined label image as a result of DenseCRF.  
        Q : (K, N, M) ndarray,
            pixelwise probability of the class.
    """
    img = img.copy(order="C")
    block_shape = _tile_size(tile_size)
    dlabels = np.zeros(img.shape[:2], dtype=np.int32)

    dcrf_params = dict(
        stdxy_g=1,
        compat_g=10,
        stdxy_b=10,
        stdrgb=10,
        compat_b=5,
        niter=50,
    )
    if dcrf_kwargs is not None:
        dcrf_params.update(dcrf_kwargs)

    views_img = np.squeeze(view_as_blocks(img, block_shape=(*block_shape, img.shape[2])))
    views_klabel = view_as_blocks(img_label, block_shape=block_shape)
    views_dlabel = view_as_blocks(dlabels, block_shape=block_shape)
    # views_dinfer = np.squeeze(view_as_blocks(dinfer, block_shape=(*block_shape, n_cluster)))
    if mask is None:
        views_mask = np.squeeze(view_as_blocks(np.ones_like(img).astype(bool),
                                               block_shape=block_shape))
    else:
        views_mask = np.squeeze(view_as_blocks(mask, block_shape=block_shape))

    for i in tqdm(range(views_mask.shape[0])):
        for j in range(views_mask.shape[1]):
            if (mask is not None) and (views_mask[i, j].sum() == 0):
                continue

            # views_dlabel[i, j], views_dinfer[i, j] = dcrf2d(
            views_dlabel[i, j] = dcrf2d(
                views_klabel[i, j],
                views_img[i, j],
                zero_unsure=zero_unsure,
                **dcrf_kwargs
            )
    return dlabels