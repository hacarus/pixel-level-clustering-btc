from typing import Sequence
import numpy as np
import xml.etree.ElementTree as ET 
from skimage.draw import circle_perimeter
from .types import cast


def _freehand_to_numpy(pointlist: Sequence[ET.Element]):
    """Extract pointlist from the elemtents in pointlist tag.

    annotation tag is like below.
    ```xml
        <annotation type="freehand" displayname="AnnotateFreehandLine" color="#00ff00">
            <measuretype>0</measuretype>
            <closed>0</closed>
            <pointlist>
                <point>
                    <x>6745728</x>
                    <y>-834005</y>
                </point>
                <point>
                    <x>6746452</x>
                    <y>-835454</y>
                    ...
    ```

    Paramters
    ---------
        pointlist: Sequence[ET.Element]
            Sequence of roots in pointlist tag.
    
    Returns
    -------
        pointlist: ndarray,
            2d array of xy coordinates of the circle perimeter.
    """
    n = len(pointlist)
    coordinates = np.zeros((n, 2), dtype=np.int32)
    for i, point in enumerate(pointlist):
        for j, elem in enumerate(point):
            coordinates[i, j] = int(elem.text)
    return coordinates


def _parse_annotation_freehand(root: ET.Element):
    """Function to parse freehand annotation."""
    annotation = {}
    for child in root:
        if child.tag == "pointlist":
            annotation[child.tag] = _freehand_to_numpy(child)
        else:
            annotation[child.tag] = child.text
    return annotation


def _circle_to_numpy(r: int, c: int, radius: int):
    """Generate pointlist from center coodinate and radius of the circle.

    annotation tag is like below.
    ```xml
        <annotation type="circle" displayname="AnnotateCircle" color="#ffffff">
            <x>24050376</x>
            <y>-2118018</y>
            <radius>2209</radius>
            <measuretype>0</measuretype>
        </annotation>
    ```

    Paramters
    ---------
        r: int,
            row of center coordinate.
        c: int,
            column of center coordinate.
        radius: int,
            radius of the circle.
    
    Returns
    -------
        pointlist: ndarray,
            2d array of xy coordinates of the circle perimeter.
    """
    return np.vstack(circle_perimeter(r, c, radius,)[::-1]).T

def _parse_annotation_circle(root: ET.Element):
    """Function to parse circle annotation."""
    annotation = {}
    for child in root:
        annotation[child.tag] = cast(child.text)
    annotation["pointlist"] = _circle_to_numpy(
        int(annotation["y"]),
        int(annotation["x"]),
        int(annotation["radius"])
    )
    return annotation


def _parse_annotation(annottype: str):
    """Return function to parse annotation tag in case by annotation type."""
    if annottype == "freehand":
        return _parse_annotation_freehand
    if annottype == "circle":
        return _parse_annotation_circle
    raise ValueError(f"Unkown annotation type: {annottype}.")


def parse_annotation(root: ET.Element):
    """Parse elements in annotation tag."""
    annotation = {k: v for k, v in root.attrib.items()}
    annotation.update(_parse_annotation(annotation["type"])(root))
    return annotation


def _parse_ndpviewstate(root: ET.Element):
    """Parse elements in ndpviewstate tag."""
    ndpstate = {}
    ndpstate["id"] = root.attrib["id"]
    for child in root:
        ndpstate[child.tag] = parse_annotation(child) if child.tag == "annotation" else cast(child.text)
    return ndpstate


def parse_ndpa(root: ET.Element):
    """Parse ndpa file."""
    annot = {}
    for child in root:
        annot[child.attrib["id"]] = _parse_ndpviewstate(child)
    return annot
