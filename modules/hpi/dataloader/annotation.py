"""annotation.py"""
from typing import Any, Dict, List


class Annotation:
    """Annotation class.

    Parameters
    ----------
        annotation: dict
            dictionary of an annotation.

    Attributes
    ----------
        title: str,
            annotation title.
        details: str,
            Unknown.
        coordformat: str,
            unit of distance.
        lens: float,
            Unknown.
        x: int,
            offset of x.
        y: int,
            offset of y.
        z: int,
            offset of y.
        showtitle: int,
            if showtitle=1, show title in NDPView system.
        showhistogram: int,
            Unknown.
        showlineprofile: int,
            Unknown
        annotation: dict,
            Annotation has elements below.
            type: str,
                annotation type.
                e.g.) circle, freehand
            displayname: str,
                name to display.
            color: str,
                Hexadecimal color index.
            x: int, (when annotation type is circle only)
            y: int, (when annotation type is circle only)
            r: int, (when annotation type is radius only)
            measuretype: int,
                Unknown.
            closed: int,  (when annotation type is freehand only)
                Unknown.
            pointlist: ndarray
                array of coordinates (x, y) of an annotation.
    """

    TAG_GROUP_1: List = ["title"]
    TAG_GROUP_2: List = ["type", "displayname", "color"]

    def __init__(self, annotation: Dict[str, Any]) -> None:
        """Constructor."""
        for k, v in annotation.items():
            setattr(self, k, v)

    def __setattr__(self, name:str, value:Any):
        """Magic method to set attributes."""
        super().__setattr__(name, value)

