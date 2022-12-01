"""ndpa.py."""
from typing import Any, Dict
from .annotation import Annotation


class NDPA:
    """Annotations class.

    Parameters
    ----------
        annotations: dict
            dictionary of annotations in a xml (.ndpa) file.

    Attributes
    ----------
        annotations: list,
            list of `Annotation` objects.
    """

    def __init__(self, xmldict: Dict[Dict, Any]) -> None:
        """Constructor."""
        if not isinstance(xmldict, dict):
            raise TypeError(
                "`annotations` must be list or dict."
                f"but {type(xmldict)} was given."
            )
        self.annotations = [Annotation(annotation) for annotation in xmldict.values()]

    def __getitem__(self, i: int) -> Annotation:
        """Get item."""
        return self.annotations[i]
