"""Locating the repository root.

Notebooks and scripts frequently need repo-relative paths, and hand-rolling the
lookup means every caller encodes its own directory depth.
Moving a file then silently breaks its paths.
Use :func:`get_root` instead of counting parent directories.
"""

from __future__ import annotations

from pathlib import Path

ROOT_MARKER = "pyproject.toml"


def get_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upwards until the marker is found.

    The root is the first directory at or above ``start`` that contains
    ``pyproject.toml``.

    Parameters
    ----------
    start : str or Path, optional
        Where to begin searching.
        A file is replaced by its containing directory.
        Defaults to the current working directory, which is usually what a
        notebook or an interactive session wants.
        Pass ``__file__`` from inside a module to search from that module's
        location instead, which does not depend on the working directory.

    Returns
    -------
    Path
        Absolute path to the repository root.

    Raises
    ------
    FileNotFoundError
        If no ancestor directory contains ``pyproject.toml``.
        This is deliberately loud: silently returning the wrong root produces
        confusing missing-data errors much further downstream.

    Examples
    --------
    >>> from dusty_colors import get_root
    >>> root = get_root()
    >>> stack = root / "results" / "stacks" / "dp1_default"
    """
    origin = Path.cwd() if start is None else Path(start)
    origin = origin.resolve()
    if origin.is_file():
        origin = origin.parent

    for candidate in (origin, *origin.parents):
        if (candidate / ROOT_MARKER).exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {ROOT_MARKER} in {origin} or any parent directory. "
        "Run from inside the dusty_colors repository, or pass an explicit "
        "'start' path."
    )
