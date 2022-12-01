"""ndpy.py."""
from typing import Dict, Tuple
from dataclasses import dataclass, field
import numpy as np
import openslide


@dataclass
class NDPIMeta:
    """Data format of a `.ndpi` file.

    Attributes
    ----------
        PROP_KEY_XOFFSET: str,
            field for offset of x axis from the center of a whole slide.
        PROP_KEY_YOFFSET: str,
            field for offset of y axis from the center of a whole slide.
        PROP_KEY_XMPP: str,
            field for um/pixel of x axis.
        PROP_KEY_YMPP: str,
            field for um/pixel of x axis.
    """

    # offset from center of a whole slide.
    PROP_KEY_XOFFSET: str = field(default="hamamatsu.XOffsetFromSlideCentre", init=False)
    PROP_KEY_YOFFSET: str = field(default="hamamatsu.YOffsetFromSlideCentre", init=False)
    # micro meter per pixel
    PROP_KEY_XMPP: str = field(default="openslide.mpp-x", init=False)
    PROP_KEY_YMPP: str = field(default="openslide.mpp-y", init=False)


@dataclass
class NDPI(NDPIMeta):
    """Class for `.ndpi` file.

    Attributes
    ----------
        slide: OpenSlide,
            OpenSlide object of the `.ndpi` file.
        levels: {level: scale} dict,
            Levels of the whole slide images in the `.ndpi`.

    Parameters
    ----------
        slide: OpenSlide,
            OpenSlide object of the `.ndpi` file.
    """

    # Main attributes
    slide: openslide.OpenSlide

    # internal
    x_off: int = field(init=False)
    y_off: int = field(init=False)
    x_mpp: float = field(init=False)
    y_mpp: float = field(init=False)

    def __post_init__(self) -> None:
        """Constructor."""
        self.x_off = int(self.slide.properties[self.PROP_KEY_XOFFSET])
        self.y_off = int(self.slide.properties[self.PROP_KEY_YOFFSET])
        self.x_mpp = float(self.slide.properties[self.PROP_KEY_XMPP])
        self.y_mpp = float(self.slide.properties[self.PROP_KEY_YMPP])

    @property
    def levels(self) -> Dict[int, float]:
        """Levels of the whole slide images in the `.ndpi`.

        Returns
        -------
            levels: {level: scale} dict,
        """
        base = self.slide.dimensions[0]
        levels = {}
        for (i, res) in enumerate(self.slide.level_dimensions):
            levels[i] = res[0] / base
        return levels

    def level_image(self,
                    level: int = 3,
                    bbox: Tuple[int, ...] = ()) -> np.ndarray:
        """Get image on a specified level.

        Parameters
        ----------
            levels: int,
                level of the whole slide image.
            bbox: (x, y, width, height) tuple,
                bounding box to crop from the whole slide image.

        Returns
        -------
            img: ndarray,
                image with the level specified.
                if bbox was given, the image was cropped.
        """
        img_pil = self.slide.read_region(
            (0, 0),
            level,
            self.slide.level_dimensions[level]
        )
        img = np.asarray(img_pil.convert("RGB"))

        if len(bbox) == 0:
            return img
        if len(bbox) == 4:
            x, y, w, h = bbox
            return np.asarray(img[y:y + h, x:x + w])

        raise ValueError(f"bbox must have four elements but {bbox} was given.")

#     def generate_deepzoom(self, tile_size=254, overlap=1, limit_bounds=False):
#         """Generate DeepZoomGenerator object.

#         Parameters
#         ----------
#             Create a DeepZoomGenerator wrapping an OpenSlide object.

#             osr:          a slide object.
#             tile_size:    the width and height of a single tile.  For best viewer
#                           performance, tile_size + 2 * overlap should be a power
#                           of two.
#             overlap:      the number of extra pixels to add to each interior edge
#                           of a tile.
#             limit_bounds: True to render only the non-empty slide region.

#         Returns
#         -------
#             DeepZoomGenerator object.
#         """
#         return DeepZoomGenerator(self.slide, tile_size, overlap, limit_bounds)
