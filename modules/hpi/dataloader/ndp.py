# type: ignore
"""ndp.py."""
from typing import Any, Dict, Optional, List, Sequence, Tuple, Union
from warnings import warn
import copy
from dataclasses import dataclass, field
import numpy as np
from openslide.deepzoom import DeepZoomGenerator
from skimage.draw import polygon2mask
from .ndpi import NDPI
from .ndpa import Annotation, NDPA

_tileMask = Union[Dict[int, np.ndarray], Dict[str, np.ndarray]]


def _get_annotation(ndpi: NDPI,
                    ndpa: NDPA,
                    level: int = 4,
                    offset: Tuple[int, ...] = (0, 0)
                    ) -> List[Annotation]:
    """Rescale annotations into specified level.

    Parameters
    ----------
        level: int,
            scale level of whole slide image.
        offset: (x, y) tuple,
            offset of tile in whole slide image.

    Returns
    -------
        annotations: list of Annotation object,
    """
    w0, h0 = ndpi.slide.dimensions
    annotations = copy.deepcopy(ndpa.annotations)
    for annotation in annotations:
        xx = annotation.annotation["pointlist"][:, 0]
        yy = annotation.annotation["pointlist"][:, 1]

        # offset of coordinates
        xx = xx - ndpi.x_off
        yy = yy - ndpi.y_off

        # covert `nano meter` to `pixel`
        xx = xx / (1000 * ndpi.x_mpp)
        yy = yy / (1000 * ndpi.y_mpp)

        # shift coordinates
        xx = (xx + w0 / 2)
        yy = (yy + h0 / 2)

        # scaling
        xx = xx * ndpi.levels[level]
        yy = yy * ndpi.levels[level]

        # update
        annotation.annotation["pointlist"][:, 0] = np.int32(xx) - int(offset[0])
        annotation.annotation["pointlist"][:, 1] = np.int32(yy) - int(offset[1])
    return annotations


def _getattr_1(annotation: Annotation, attr: str):
    return getattr(annotation, attr)


def _getattr_2(annotation: Annotation, attr: str):
    return getattr(annotation.annotation, attr)


def _get_getattr(groupby: str):
    if groupby in Annotation.TAG_GROUP_1:
        return _getattr_1
    elif groupby in Annotation.TAG_GROUP_2:
        return _getattr_2
    raise ValueError(
        f"{groupby} does not define as a member of groups."
        f"Only a member of {Annotation.TAG_GROUP_1}"
        f"or {Annotation.TAG_GROUP_2} is available."
    )


def _get_unique_groups(annotations: Sequence[Annotation], groupby: str):
    _getattr = _get_getattr(groupby)
    unames = []
    for annotation in annotations:
        try:
            unames.append(_getattr(annotation, groupby))
        except TypeError as e:
            warn(
                f"_getattr(annotation, {groupby}) "
                "return {_getattr(annotation, groupby)}"
                f"on {annotation.id}"
            )
            print(e)
    return unames


def _get_masks_bygroup(shape, annotations: Sequence[Annotation], groupby: str
                       ) -> Dict[str, np.ndarray]:
    _getattr = _get_getattr(groupby)
    unames = _get_unique_groups(annotations, groupby)
    masks = {g: np.zeros(shape=shape, dtype=bool) for g in unames}
    for annotation in annotations:
        g = _getattr(annotation, groupby)
        mask = polygon2mask(shape,
                            annotation.annotation["pointlist"][:, ::-1]) > 0
        masks[g] = np.logical_or(masks[g], mask)
    return masks


class NDP:
    """Class for `.ndpi` and `.ndpa` files.

    Attributes
    ----------
        ndpi: NDPI
            NDPI object.
        ndpa: Path,
            ndpa object.
    """

    def __init__(self, ndpi: NDPI, ndpa: Optional[NDPA]):
        """Constructor.

        Parameters
        ----------
            ndpi: NDPI
                NDPI object.
            ndpa: Path,
                ndpa object.
        """
        self.ndpi = ndpi
        self.ndpa = ndpa

    def get_image(self,
                  level: int = 4,
                  bbox: Tuple[int, ...] = ()) -> np.ndarray:
        """Get image from WSI.

        Parameters
        ----------
            level: int,
                scale level of whole slide image.
            bbox: (x, y, width, height) tuple,
                bbox to crop from the whole slide image.

        Returns
        -------
            img: (N, M [,3)]) ndarray.
        """
        img = self.ndpi.level_image(level, bbox)
        return img

    def get_annotation(self,
                       level: int = 4,
                       bbox: Tuple[int, ...] = (0, 0)) -> List[Annotation]:
        """Rescale annotations into specified level.

        Parameters
        ----------
            level: int,
                scale level of whole slide image.
            bbox: (x, y, width, height) tuple,
                bbox to crop from the whole slide image.

        Returns
        -------
            annotations: list of Annotation object,
        """
        if self.ndpa is None:
            raise ValueError("The NDP class must be initialized.")

        return _get_annotation(self.ndpi, self.ndpa, level, bbox)

    def get_mask(self,
                 level: int = 4,
                 bbox: Optional[Tuple[int, ...]] = None,
                 groupby: Optional[str] = None) -> np.ndarray:
        """Get image from WSI.

        Parameters
        ----------
            level: int,
                scale level of whole slide image.
            bbox: (x, y, width, height) tuple, Optional
                bbox to crop from the whole slide image.
            groupby: str,
                Mask grouping method.
                "color", "displayname" and "type" are available.

        Returns
        -------
            mask: (K, N, M) ndarray.
        """
        if self.ndpa is None:
            raise ValueError("NDP.ndpa is None.")

        # generate 2d mask for each annotation
        if bbox is None:
            shape = self.ndpi.slide.level_dimensions[level][::-1]
            bbox = (0, 0, 0, 0)
        elif len(bbox) == 4:
            shape = (bbox[3], bbox[2])
        else:
            raise ValueError("length of bbox must be 4.")

        # load ndpa file and generate annotations in pixel scale.
        annotations = self.get_annotation(level, bbox)
        if groupby is None:
            masks = {}
            for (i, annotation) in enumerate(annotations):
                masks[i] = polygon2mask(
                    shape, annotation.annotation["pointlist"][:, ::-1]
                ) > 0
            return masks
        return _get_masks_bygroup(shape, annotations, groupby)


@dataclass
class DeepZoom:
    """Class to treat DeepZoom format.

    Parameters
    ----------
        ndpi: NDPI,
            NDPI object.
        ndpa: NDPA,
            NDPA object.

    Attributes
    ----------
        ndpi: NDPI,
            NDPI object.
        ndpa: NDPA,
            NDPA object.
        tile_size: int,
            the width and height of a single tile. For best viewer
            performance, tile_size + 2 * overlap should be a power
            of two.
        overlap: int,
            the number of extra pixels to add to each interior edge
            of a tile.
        limit_bounds: bool,
            True to render only the non-empty slide region.
    """

    # main attributes
    ndpi: NDPI

    # options
    ndpa: Optional[NDPA] = field(default=None, init=True)
    tile_size: int = 254
    overlap: int = 1
    limit_bounds: bool = False

    # internal
    dzg: DeepZoomGenerator = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Constructor."""
        self.dzg = DeepZoomGenerator(
            self.ndpi.slide,
            tile_size=self.tile_size,
            overlap=self.overlap,
            limit_bounds=self.limit_bounds
        )

    def info(self) -> None:
        """Show meta information."""
        # openslide.slide object
        print("\n===openslide.slide object===")
        print(f"level dimensions\t: {self.ndpi.slide.level_dimensions}")

        # properties of DeepZoomGenerator
        print("\n===openslide.deepzoom.DeepZoomGenerator object===")
        print(f"level count\t\t: {self.dzg.level_count}")
        print(f"tile count\t\t: {self.dzg.tile_count}")
        print(f"level tiles\t\t: {self.dzg.level_tiles}")
        print(f"level dimensions\t: {self.dzg.level_dimensions}")
        print(f"get dzi\t\t\t: {self.dzg.get_dzi('png')}")

    @property
    def dimension(self) -> Any:  # Tuple[int, int]
        """Dimension of WSI."""
        return self.dzg.level_dimensions[-1]

    def get_tile(
        self,
        level: int,
        address: Tuple[float, float],
        return_mask: bool = False,
        skip_no_mask: bool = False,
        groupby: Optional[str] = None
    ) -> Tuple[np.ndarray, Union[_tileMask, Dict[Any, Any]]]:
        """Get tile from WSI.

        Parameters
        ----------
            level: int,
                deep zoom level (not level in ndpi file).
            address: (float, float) tuple,
                address of a tile.
            return_mask: bool,
                return mask image.
            skip_no_mask: bool,
                skip reading from a tile without mask.
            groupby: str,
                Mask grouping method.
                "color", "displayname" and "type" can be available.

        Returns
        -------
            (tile,) or (tile, masks).
        """
        if not return_mask:
            tile = np.asarray(self.dzg.get_tile(level=level, address=address))
            return (tile, {})

        # shape of WSI
        wsi_shape = self.dimension
        # shape of deep zoom image
        dzimage_shape = self.dzg.level_dimensions[level]
        # calculate scale
        scale = dzimage_shape[0] / wsi_shape[0]

        # get top left coordinates of a tile in WSI
        coordinates = self.dzg.get_tile_coordinates(level, address)
        coords_in_wsi, level_in_ndpi, tile_shape = coordinates
        coords_rescale = tuple(np.array(coords_in_wsi) * scale)

        # get annotation from scaled WSI
        annotations = self.get_annotation(level=level_in_ndpi,
                                          offset=coords_rescale)
        _cap = self.tile_size + 2 * self.overlap
        have_mask = np.array([
            (np.sum(ann.annotation["pointlist"] <= _cap) > 0) and np.sum(
                ann.annotation["pointlist"] > 0) for ann in annotations
        ])
        no_mask = have_mask.sum() == 0

        if skip_no_mask & no_mask:
            return (np.array([], dtype=np.uint8), {})

        tile = np.asarray(self.dzg.get_tile(level=level, address=address))
        masks = self._get_tile_mask(tile_shape[::-1], annotations, groupby)
        return (tile, masks)

    def _get_tile_mask(
        self,
        tile_shape: Tuple[int, int],
        annotations: List[Annotation],
        groupby: Optional[str] = None
    ) -> _tileMask:
        """Get tile mask.

        Parameters
        ----------
            tile_shape: tuple,
                shape of tinle.
            annotations: list,
                list of Annotation object.
            groupby: str,
                Mask grouping method.
                "color", "displayname" and "type" are available.

        Returns
        -------
            mask: (K, N, M) image,
                stacked mask images.
        """
        if groupby is None:
            masks = {}
            for (i, annotation) in enumerate(annotations):
                mask = polygon2mask(
                    tile_shape, annotation.annotation["pointlist"][:, ::-1]
                )
                masks[i] = np.asarray(mask)
            return masks
        return _get_masks_bygroup(tile_shape, annotations, groupby)

    def get_annotation(self,
                       level: int = 4,
                       offset: Tuple[int, ...] = (0, 0)
                       ) -> List[Annotation]:
        """Rescale annotations into specified level.

        Parameters
        ----------
            level: int,
                scale level of whole slide image.
            offset: (x, y) tuple,
                offset of tile in whole slide image.

        Returns
        -------
            annotations: list of Annotation object,
        """
        if self.ndpa is None:
            raise ValueError("The DeepZoom class must be initialized.")
        return _get_annotation(self.ndpi, self.ndpa, level, offset)
