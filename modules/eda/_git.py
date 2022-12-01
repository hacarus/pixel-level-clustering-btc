import os
from pathlib import Path
from subprocess import Popen
from typing import Any, Optional, Union


class GitError(BaseException):
    """GitError."""


def git(cmd: str, *args: str) -> str:
    """Run a Git command.

    Parameters
    ----------
    cmd: str
        Git subcommand such as `add`, `push`, and so on.
    *args:
        Additional arguments to the command.
    """
    command = ("git", cmd) + args
    with Popen(command, stdin=-1, stdout=-1, stderr=-1) as proc:
        out, err = proc.communicate()

    if len(err) != 0:
        raise GitError(err.strip())

    return out.strip().decode()


def git_root(*args: Union[str, Any],  # Any should be builtins._PathLike
             absolute: bool = False) -> Path:
    """Find a Git root path.

    Parameters
    ----------
    *args: Tuple[Union[str, Path], ...]
        Pathname components that define path relative to the Git root.
    absolute: bool, default=False
        Whether to return an absolute path.

    Returns
    -------
    pathlib.Path
    """
    absolute_path = Path(git("rev-parse", "--show-toplevel"))
    if absolute:
        return Path(absolute_path, *args)

    current_depth = len(str(Path.cwd().relative_to(absolute_path)).split(os.sep))
    return Path(*([".."] * current_depth), *args)


def get_active_branch_name() -> Optional[str]:
    """Get branch name.

    Returns
    -------
    str
        branch name.
    """
    head_dir = git_root(absolute=True) / ".git/HEAD"
    with head_dir.open("r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]
    return None
