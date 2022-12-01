from typing import List, Tuple, Optional
import numpy as np
from scipy import ndimage
from skimage import measure
from ..dataloader import DeepZoom, NDP
from .transform import address2pixel


def _new_tile(shift: int, ad_xmin: int, ad_xmax: int,
              ad_ymin: int, ad_ymax: int, channel: int = 3, dtype=float):
    """Generate new empty stitched tile."""
    h = int(shift * (ad_ymax - ad_ymin + 1))
    w = int(shift * (ad_xmax - ad_xmin + 1))
    if channel == 1:
        return np.zeros(shape=(h, w), dtype=dtype)
    return np.zeros(shape=(h, w, channel), dtype=dtype)


def _crop(dz: DeepZoom, addresses: List[Tuple[float, float]], tile_size: int,
          base_level: int, target_level: Optional[int] = None,
          return_mask: bool = False, groupby_mask: Optional[str] = None
          ) -> np.ndarray:
    """Crop image from WSI.

    Parameters
    ----------
        dz: DeepZoom,
            DeepZoom object.
        addresses: List[Tuple[int, int]],
            top-left and bottom-right addresses of (x, y) coordinates
            on DeepZoom image at 'base_level'.
        tile_size: int,
        base_level: int
            DeepZoom level of given addresses.
        target_level: int
            DeepZoom level of target addresses.
        return_mask: bool, default is False.
            It True, return annotation mask for cropped image.
        groupby: Optional[str],
            Mask grouping method.
            "color", "displayname" and "type" are available.

    Returns
    -------
        cropped: (N, M) ndarray
            cropped image
        mask: Dict[str, (N, M) ndarray] (If return_mask is True)
            masks for the cropped regions
    """
    if target_level is None:
        target_level = base_level
    level_max_dz = dz.dzg.level_count - 1
    target_level_ndp = level_max_dz - target_level
    xmin0, ymin0 = address2pixel(addresses[0], tile_size, base_level,
                                 level_max_dz)
    xmin, ymin = address2pixel(addresses[0], tile_size, base_level,
                               target_level)
    xmax, ymax = address2pixel(addresses[1], tile_size, base_level,
                               target_level)
    w = xmax - xmin
    h = ymax - ymin
    ndp = NDP(dz.ndpi, dz.ndpa)

    cropped = np.asarray(ndp.ndpi.slide.read_region((xmin0, ymin0),
                         level=target_level_ndp, size=(w, h)))
    if return_mask:
        masks = ndp.get_mask(level=target_level_ndp, bbox=(xmin, ymin, w, h),
                             groupby=groupby_mask)
        # fill holes of each regions.
        for _, mask in masks.items():
            labels = measure.label(mask)
            regions = measure.regionprops(labels)
            for region in regions:
                bbox = region.bbox
                bbox_filled = ndimage.binary_fill_holes(region.filled_image)
                bbox_slice_row = slice(bbox[0], bbox[2])
                bbox_slice_col = slice(bbox[1], bbox[3])
                mask[bbox_slice_row, bbox_slice_col] = np.logical_or(
                    mask[bbox_slice_row, bbox_slice_col], bbox_filled
                )
        return cropped, masks
    return cropped,
