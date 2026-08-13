"""The science figures: measured color excess profiles and their jackknife.

Each figure is one linear function: make the axes, pull the arrays it needs out
of the stack, draw, label, return. They are meant to be readable top to bottom
and easy to copy into a notebook and modify.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..results import StackResults, load_stack_source
from ..utils import FIGSIZE, use_matplotlib_style
from .base import PostRunContext, register

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Fixed per color so a given color-pair looks the same in every figure and in
# the paper. Anything not listed falls back to its position in the cycle.
COLOR_STYLES = {"g-i": "k", "g-r": "C0", "r-i": "C2", "i-z": "C3"}


def plot_jackknife_samples(
    results: StackResults,
    *,
    figsize: tuple[float, float] = FIGSIZE,
    sample_alpha: float = 0.18,
) -> "Figure":
    """Plot every jackknife sample of the first color, over their mean."""

    import matplotlib.pyplot as plt

    use_matplotlib_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    color = results.first_color
    radius = np.asarray(results.require(f"{color}_bin_centers"), dtype=float)
    samples = np.asarray(results.require(f"{color}_jackknife_samples"), dtype=float)
    mean = np.asarray(results.require(f"{color}_jackknife_avg"), dtype=float)
    err = np.asarray(results.require(f"{color}_jackknife_err"), dtype=float)

    if samples.ndim != 2 or samples.shape[1] != radius.shape[0]:
        raise ValueError(
            f"{color}_jackknife_samples has shape {samples.shape}, expected "
            f"(n_patches, {radius.shape[0]}) to match {color}_bin_centers"
        )

    for sample in samples:
        # Log axes drop non-positive points silently, so mask per sample rather
        # than leaving it to matplotlib.
        show = np.isfinite(radius) & np.isfinite(sample) & (radius > 0) & (sample > 0)
        ax.plot(
            radius[show],
            sample[show],
            marker=".",
            markersize=2.5,
            lw=0.7,
            alpha=sample_alpha,
            color=COLOR_STYLES.get(color, "C0"),
            zorder=1,
        )

    show = (
        np.isfinite(radius)
        & np.isfinite(mean)
        & np.isfinite(err)
        & (radius > 0)
        & (mean > 0)
        & (err >= 0)
    )
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

    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$r_\perp$ [kpc]",
        ylabel=rf"$E({color})$ [mag]",
    )
    ax.set_box_aspect(1)
    ax.legend(frameon=False, handlelength=1.5)
    return fig


def plot_color_signals(
    results: StackResults,
    *,
    figsize: tuple[float, float] = FIGSIZE,
    x_offset: float = 0.04,
) -> "Figure":
    """Plot the measured signal with jackknife errors for every color."""

    import matplotlib.pyplot as plt

    use_matplotlib_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    plotted = 0
    for index, color in enumerate(results.colors):
        radius = np.asarray(results.require(f"{color}_bin_centers"), dtype=float)
        signal = np.asarray(results.require(f"{color}_avg"), dtype=float)
        if f"{color}_jackknife_err" not in results.arrays:
            raise KeyError(
                f"{results.stack_dir / f'stack_{results.mode}.npz'} is missing "
                f"'{color}_jackknife_err'; this figure requires jackknife errors"
            )
        err = np.asarray(results.arrays[f"{color}_jackknife_err"], dtype=float)

        show = (
            np.isfinite(radius)
            & np.isfinite(signal)
            & np.isfinite(err)
            & (radius > 0)
            & (signal > 0)
            & (err >= 0)
        )
        if not np.any(show):
            warnings.warn(
                f"{color} has no positive finite signal values for log plotting",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        # Nudge each color sideways so overlapping error bars stay readable.
        offset = 1.0 + (index - (len(results.colors) - 1) / 2.0) * x_offset
        ax.errorbar(
            radius[show] * offset,
            signal[show],
            yerr=err[show],
            marker="o",
            markersize=3,
            capsize=2,
            ls="",
            color=COLOR_STYLES.get(color, f"C{index}"),
            label=f"${color}$",
            zorder=10 - index,
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("No positive finite signal values were available to plot")

    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"$r_\perp$ [kpc]",
        ylabel="Color excess [mag]",
    )
    ax.set_box_aspect(1)
    ax.legend(frameon=False, handletextpad=0.2, columnspacing=0.6)
    return fig


def save_stack_figures(
    source: StackResults | str | Path,
    output_dir: str | Path,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    extension: str = "pdf",
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Write the science figures for one stack mode.

    ``source`` may be a loaded stack, a stack directory, or the analysis YAML
    that produced it.
    """

    import matplotlib.pyplot as plt

    results = (
        source
        if isinstance(source, StackResults)
        else load_stack_source(source, mode=mode, root=root)
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{results.stack_dir.name}_{results.mode}"
    extension = extension.lstrip(".")

    jackknife_path = out_dir / (
        f"{stem}_{results.first_color.replace('-', '_')}_jackknife.{extension}"
    )
    fig = plot_jackknife_samples(results)
    fig.savefig(jackknife_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    signals_path = out_dir / f"{stem}_all_colors.{extension}"
    fig = plot_color_signals(results)
    fig.savefig(signals_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return jackknife_path, signals_path


@register("stack_figures")
def _stage(context: PostRunContext, mode: str) -> tuple[Path, ...]:
    """Write the science figures for one stack mode."""

    return save_stack_figures(
        context.results(mode), context.stack_dir, root=context.root
    )


__all__ = [
    "COLOR_STYLES",
    "plot_color_signals",
    "plot_jackknife_samples",
    "save_stack_figures",
]
