"""_resolve.py."""
from typing import Union
from pathlib import Path


def resolve_path(path: Union[str, Path]) -> Path:
    """Resolve relative path.
    
    Parameters
    ----------
        path: str or Path,
            path to resolve relative path.
    
    Returns
    -------
        full_path: Path
    """
    full_path = ""
    for part in Path(path).parts:
        full_path += f"/{part}"
        full_path = str(Path(full_path).resolve())
    return Path(full_path)
