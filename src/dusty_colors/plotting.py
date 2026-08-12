"""Plotting helpers for TreeCorr stack outputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import warnings

import numpy as np

from .results import StackResults, load_stack_results

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

DEFAULT_FIGSIZE = (3.0, 3.0)
DEFAULT_COLOR_STYLES = {
    "g-i": "k",
    "g-r": "C0",
    "r-i": "C2",
    "i-z": "C3",
}
DEFAULT_RADIAL_STYLES = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")


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
    """Apply the project Matplotlib settings and return the style path used."""

    path = default_style_path() if style_path is None else Path(style_path)
    import matplotlib as mpl

    mpl.rc_file(path)
    return path


def plot_first_color_jackknife(
    source: StackResults | str | Path,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    stack_dir: str | Path | None = None,
    ax: "Axes | None" = None,
    style: bool = True,
    style_path: str | Path | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    sample_alpha: float = 0.18,
) -> "Figure":
    """Plot jackknife samples and their jackknife mean for the first color."""

    results = _coerce_stack_results(
        source,
        mode=mode,
        root=root,
        stack_dir=stack_dir,
    )
    fig, ax = _figure_and_axis(ax, style=style, style_path=style_path, figsize=figsize)

    color = results.first_color
    radius = _profile_array(results, color, "bin_centers")
    samples = _profile_array(results, color, "jackknife_samples")
    mean = _profile_array(results, color, "jackknife_avg")
    err = _profile_array(results, color, "jackknife_err")
    _require_2d_samples(samples, radius, color)

    color_style = _color_style(color, 0)
    for sample in samples:
        show = _positive_xy_mask(radius, sample)
        if np.any(show):
            ax.plot(
                radius[show],
                sample[show],
                marker=".",
                markersize=2.5,
                lw=0.7,
                alpha=sample_alpha,
                color=color_style,
                zorder=1,
            )

    show = _profile_mask(radius, mean, err)
    if not np.any(show):
        raise ValueError(f"{color} has no positive finite jackknife mean values")

    ax.errorbar(
        radius[show],
        mean[show],
        yerr=err[show],
        marker="s",
        markerfacecolor="none",
        markersize=4,
        capsize=2,
        ls="",
        color="k",
        label="Jackknife mean",
        zorder=10,
    )
    _format_stack_axis(ax, ylabel=_single_color_ylabel(color))
    ax.legend(frameon=False, handlelength=1.5)
    return fig


def plot_all_color_signals(
    source: StackResults | str | Path,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    stack_dir: str | Path | None = None,
    ax: "Axes | None" = None,
    style: bool = True,
    style_path: str | Path | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    x_offset: float = 0.04,
) -> "Figure":
    """Plot the full stack signal with jackknife errors for every color."""

    results = _coerce_stack_results(
        source,
        mode=mode,
        root=root,
        stack_dir=stack_dir,
    )
    fig, ax = _figure_and_axis(ax, style=style, style_path=style_path, figsize=figsize)

    plotted = 0
    for index, color in enumerate(results.colors):
        radius = _profile_array(results, color, "bin_centers")
        signal = _profile_array(results, color, "avg")
        err = _jackknife_err(results, color)
        show = _profile_mask(radius, signal, err)
        if not np.any(show):
            warnings.warn(
                f"{color} has no positive finite full-signal values for log plotting",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        offset = 1.0 + (index - (len(results.colors) - 1) / 2.0) * x_offset
        ax.errorbar(
            radius[show] * offset,
            signal[show],
            yerr=err[show],
            marker="o",
            markersize=3,
            capsize=2,
            ls="",
            color=_color_style(color, index),
            label=_legend_color_label(color),
            zorder=10 - index,
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("No positive finite full-signal values were available to plot")

    _format_stack_axis(ax, ylabel="Color excess [mag]")
    ax.legend(frameon=False, handletextpad=0.2, columnspacing=0.6)
    return fig


def plot_photoz_radial_distributions(
    source: StackResults | str | Path,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    stack_dir: str | Path | None = None,
    ax: "Axes | None" = None,
    style: bool = True,
    style_path: str | Path | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> "Figure":
    """Plot pair-weighted background photo-z distributions by radial bin."""

    results = _coerce_stack_results(
        source,
        mode=mode,
        root=root,
        stack_dir=stack_dir,
    )
    fig, ax = _figure_and_axis(ax, style=style, style_path=style_path, figsize=figsize)
    counts = np.asarray(
        results.require_diagnostic("diagnostic_photoz_counts"),
        dtype=float,
    )
    edges = np.asarray(
        results.require_diagnostic("diagnostic_photoz_bin_edges"),
        dtype=float,
    )
    radial_edges = np.asarray(
        results.require_diagnostic("diagnostic_radial_bin_edges"),
        dtype=float,
    )
    _plot_radial_histograms(ax, counts, edges, radial_edges)
    _format_distribution_axis(
        ax,
        xlabel="Photometric redshift",
        ylabel="Pair density",
    )
    return fig


def plot_color_radial_distributions(
    source: StackResults | str | Path,
    color: str | None = None,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    stack_dir: str | Path | None = None,
    ax: "Axes | None" = None,
    style: bool = True,
    style_path: str | Path | None = None,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> "Figure":
    """Plot pair-weighted background color distributions by radial bin."""

    results = _coerce_stack_results(
        source,
        mode=mode,
        root=root,
        stack_dir=stack_dir,
    )
    selected_color = results.first_color if color is None else str(color)
    fig, ax = _figure_and_axis(ax, style=style, style_path=style_path, figsize=figsize)
    counts = np.asarray(
        results.require_diagnostic(f"{selected_color}_diagnostic_color_counts"),
        dtype=float,
    )
    edges = np.asarray(
        results.require_diagnostic(f"{selected_color}_diagnostic_color_bin_edges"),
        dtype=float,
    )
    radial_edges = np.asarray(
        results.require_diagnostic("diagnostic_radial_bin_edges"),
        dtype=float,
    )
    _plot_radial_histograms(ax, counts, edges, radial_edges)
    _format_distribution_axis(
        ax,
        xlabel=rf"${selected_color}$ [mag]",
        ylabel="Pair density",
    )
    return fig


def save_stack_figures(
    source: StackResults | str | Path,
    output_dir: str | Path,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    stack_dir: str | Path | None = None,
    extension: str = "pdf",
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Create and save the two standard stack figures."""

    results = _coerce_stack_results(
        source,
        mode=mode,
        root=root,
        stack_dir=stack_dir,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    extension = extension.lstrip(".")
    stem = f"{results.stack_dir.name}_{results.mode}"
    color = _slug(results.first_color)

    jackknife_path = output_path / f"{stem}_{color}_jackknife.{extension}"
    signals_path = output_path / f"{stem}_all_colors.{extension}"

    fig = plot_first_color_jackknife(results)
    fig.savefig(jackknife_path, dpi=dpi, bbox_inches="tight")
    _close_figure(fig)

    fig = plot_all_color_signals(results)
    fig.savefig(signals_path, dpi=dpi, bbox_inches="tight")
    _close_figure(fig)
    return jackknife_path, signals_path


def save_stack_diagnostic_figures(
    source: StackResults | str | Path,
    output_dir: str | Path,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    stack_dir: str | Path | None = None,
    extension: str = "pdf",
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Create and save available radial-bin diagnostic figures."""

    results = _coerce_stack_results(
        source,
        mode=mode,
        root=root,
        stack_dir=stack_dir,
    )
    if not _has_diagnostic_arrays(results):
        return ()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    extension = extension.lstrip(".")
    stem = f"{results.stack_dir.name}_{results.mode}"
    paths: list[Path] = []

    if {
        "diagnostic_photoz_counts",
        "diagnostic_photoz_bin_edges",
    }.issubset(results.diagnostics):
        photoz_path = output_path / f"{stem}_photoz_distribution.{extension}"
        fig = plot_photoz_radial_distributions(results)
        fig.savefig(photoz_path, dpi=dpi, bbox_inches="tight")
        _close_figure(fig)
        paths.append(photoz_path)

    for color in results.colors:
        if not {
            f"{color}_diagnostic_color_counts",
            f"{color}_diagnostic_color_bin_edges",
        }.issubset(results.diagnostics):
            continue
        color_path = output_path / (
            f"{stem}_{_slug(color)}_color_distribution.{extension}"
        )
        fig = plot_color_radial_distributions(results, color)
        fig.savefig(color_path, dpi=dpi, bbox_inches="tight")
        _close_figure(fig)
        paths.append(color_path)

    return tuple(paths)


def _coerce_stack_results(
    source: StackResults | str | Path,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    stack_dir: str | Path | None = None,
) -> StackResults:
    if isinstance(source, StackResults):
        return source

    path = Path(source)
    if path.is_dir():
        return load_stack_results(stack_dir=path, mode=mode, root=root)
    return load_stack_results(path, stack_dir=stack_dir, mode=mode, root=root)


def _figure_and_axis(
    ax: "Axes | None",
    *,
    style: bool,
    style_path: str | Path | None,
    figsize: tuple[float, float],
) -> tuple["Figure", "Axes"]:
    if ax is not None:
        _set_square_axis(ax)
        return ax.figure, ax

    if style:
        use_matplotlib_style(style_path)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _set_square_axis(ax)
    return fig, ax


def _format_stack_axis(ax: "Axes", *, ylabel: str) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$r_\perp$ [kpc]")
    ax.set_ylabel(ylabel)
    _set_square_axis(ax)


def _format_distribution_axis(ax: "Axes", *, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, handlelength=1.4, fontsize=7)
    _set_square_axis(ax)


def _set_square_axis(ax: "Axes") -> None:
    if hasattr(ax, "set_box_aspect"):
        ax.set_box_aspect(1)


def _close_figure(fig: "Figure") -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)


def _profile_array(results: StackResults, color: str, suffix: str) -> np.ndarray:
    return np.asarray(results.require(f"{color}_{suffix}"), dtype=float)


def _jackknife_err(results: StackResults, color: str) -> np.ndarray:
    key = f"{color}_jackknife_err"
    if key in results.arrays:
        return np.asarray(results.arrays[key], dtype=float)
    raise KeyError(
        f"{results.stack_dir / f'stack_{results.mode}.npz'} is missing {key!r}; "
        "full-signal plots require jackknife errors"
    )


def _require_2d_samples(samples: np.ndarray, radius: np.ndarray, color: str) -> None:
    if samples.ndim != 2:
        raise ValueError(f"{color}_jackknife_samples must be two-dimensional")
    if samples.shape[1] != radius.shape[0]:
        raise ValueError(
            f"{color}_jackknife_samples has {samples.shape[1]} radial bins, "
            f"but {color}_bin_centers has {radius.shape[0]}"
        )


def _profile_mask(
    radius: np.ndarray,
    signal: np.ndarray,
    err: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(radius)
        & np.isfinite(signal)
        & np.isfinite(err)
        & (radius > 0)
        & (signal > 0)
        & (err >= 0)
    )


def _positive_xy_mask(radius: np.ndarray, signal: np.ndarray) -> np.ndarray:
    return np.isfinite(radius) & np.isfinite(signal) & (radius > 0) & (signal > 0)


def _plot_radial_histograms(
    ax: "Axes",
    counts: np.ndarray,
    edges: np.ndarray,
    radial_edges: np.ndarray,
) -> None:
    _require_histogram_shape(counts, edges, radial_edges)
    plotted = 0
    for index, row in enumerate(counts):
        total = np.sum(row)
        if total <= 0:
            continue
        widths = np.diff(edges)
        density = np.divide(
            row,
            total * widths,
            out=np.zeros_like(row),
            where=widths > 0,
        )
        ax.stairs(
            density,
            edges,
            color=_radial_style(index),
            label=_radial_bin_label(radial_edges[index], radial_edges[index + 1]),
            lw=1.0,
        )
        plotted += 1
    if plotted == 0:
        raise ValueError("No finite diagnostic pairs were available to plot")


def _require_histogram_shape(
    counts: np.ndarray,
    edges: np.ndarray,
    radial_edges: np.ndarray,
) -> None:
    if counts.ndim != 2:
        raise ValueError("Diagnostic counts must be two-dimensional")
    if edges.ndim != 1 or len(edges) != counts.shape[1] + 1:
        raise ValueError("Diagnostic bin edges do not match diagnostic counts")
    if radial_edges.ndim != 1 or len(radial_edges) != counts.shape[0] + 1:
        raise ValueError("Diagnostic radial bin edges do not match diagnostic counts")


def _has_diagnostic_arrays(results: StackResults) -> bool:
    if "diagnostic_radial_bin_edges" not in results.diagnostics:
        return False
    if "diagnostic_photoz_counts" in results.diagnostics:
        return True
    return any(
        f"{color}_diagnostic_color_counts" in results.diagnostics
        for color in results.colors
    )


def _color_style(color: str, index: int) -> str:
    return DEFAULT_COLOR_STYLES.get(color, f"C{index}")


def _legend_color_label(color: str) -> str:
    return f"${color}$"


def _single_color_ylabel(color: str) -> str:
    return rf"$E({color})$ [mag]"


def _radial_style(index: int) -> str:
    return DEFAULT_RADIAL_STYLES[index % len(DEFAULT_RADIAL_STYLES)]


def _radial_bin_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} kpc"


def _slug(value: str) -> str:
    return value.replace("-", "_").replace("/", "_")


__all__ = [
    "StackResults",
    "default_style_path",
    "load_stack_results",
    "plot_all_color_signals",
    "plot_color_radial_distributions",
    "plot_first_color_jackknife",
    "plot_photoz_radial_distributions",
    "save_stack_diagnostic_figures",
    "save_stack_figures",
    "use_matplotlib_style",
]
