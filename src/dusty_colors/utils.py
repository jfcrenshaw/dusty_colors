"""Small cross-cutting helpers: locating the repository root and the plot style.

Both concerns are things every notebook and script needs before it does anything
else, and neither belongs to a pipeline stage.
"""

from __future__ import annotations

from pathlib import Path

ROOT_MARKER = "pyproject.toml"

# Single-panel figures are square and column-width, which is what the paper
# uses. Shared by every plot module so the figures stay a consistent size.
FIGSIZE = (3.0, 3.0)


def get_root(start: str | Path | None = None) -> Path:
    """Find the repository root by walking upwards until the marker is found.

    The root is the first directory at or above ``start`` that contains
    ``pyproject.toml``.

    Notebooks and scripts frequently need repo-relative paths, and hand-rolling
    the lookup means every caller encodes its own directory depth, so moving a
    file silently breaks its paths. Use this instead of counting parents.

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


def default_style_path() -> Path:
    """Return the package-local Matplotlib style path.

    Returns
    -------
    Path
        Path to the ``matplotlibrc`` shipped inside the package.

    Raises
    ------
    FileNotFoundError
        If the style file is missing, which means the package was installed
        without its package data.
    """

    path = Path(__file__).with_name("matplotlibrc")
    if not path.exists():
        raise FileNotFoundError(f"Could not find dusty_colors matplotlibrc at {path}")
    return path


def use_matplotlib_style(style_path: str | Path | None = None) -> Path:
    """Apply the project Matplotlib settings and return the style path used.

    The style is not picked up implicitly, so every notebook and figure function
    has to call this. ``matplotlibrc`` deliberately carries no ``backend`` key;
    a caller needing a specific backend should call ``mpl.use`` itself.
    """

    import matplotlib as mpl

    path = default_style_path() if style_path is None else Path(style_path)
    mpl.rc_file(path)
    return path


__all__ = [
    "FIGSIZE",
    "ROOT_MARKER",
    "default_style_path",
    "get_root",
    "use_matplotlib_style",
]
