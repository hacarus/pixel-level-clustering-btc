"""Tools for EDA.

A collection of functions to enhance analysis with Jupyter Notebooks.
"""

from ._git import git, git_root, get_active_branch_name
from ._resolve import resolve_path

__all__ = [
    "git",
    "git_root",
    "get_active_branch_name",
    "resolve_path",
]
