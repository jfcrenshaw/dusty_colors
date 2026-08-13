"""Figures checking whether the background sample varies with separation.

These read the pair-weighted histograms the stacker writes to
``stack_<mode>_diagnostics.npz``. A trend in photo-z or color with separation
would mean the measured reddening has a selection component rather than being
purely dust, so these are read alongside the science figures, not instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..results import StackResults, load_stack_source
from ..utils import FIGSIZE, use_matplotlib_style
from .base import PostRunContext, register

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Radial bins are drawn in order, so these are indexed by bin, not named.
RADIAL_STYLES = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")


def plot_radial_distributions(
    results: StackResults,
    quantity: str,
    *,
    figsize: tuple[float, float] = FIGSIZE,
) -> "Figure":
    """Plot pair-weighted background distributions, one curve per radial bin.

    Parameters
    ----------
    results : StackResults
        A stack that ran with diagnostics enabled.
    quantity : str
        ``"photoz"`` for the photometric-redshift distribution, or a color name
        such as ``"g-r"`` for that color's distribution.
    figsize : tuple of float, optional
        Figure size in inches.

    Returns
    -------
    Figure
        The drawn figure.
    """

    import matplotlib.pyplot as plt

    use_matplotlib_style()
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if quantity == "photoz":
        prefix, xlabel = "diagnostic_photoz", "Photometric redshift"
    else:
        prefix, xlabel = f"{quantity}_diagnostic_color", rf"${quantity}$ [mag]"

    counts = np.asarray(results.require_diagnostic(f"{prefix}_counts"), dtype=float)
    edges = np.asarray(results.require_diagnostic(f"{prefix}_bin_edges"), dtype=float)
    radial_edges = np.asarray(
        results.require_diagnostic("diagnostic_radial_bin_edges"), dtype=float
    )
    if counts.ndim != 2:
        raise ValueError(f"{prefix}_counts must be two-dimensional")
    if edges.shape != (counts.shape[1] + 1,):
        raise ValueError(f"{prefix}_bin_edges does not match {prefix}_counts")
    if radial_edges.shape != (counts.shape[0] + 1,):
        raise ValueError(f"diagnostic_radial_bin_edges does not match {prefix}_counts")

    widths = np.diff(edges)
    plotted = 0
    for index, row in enumerate(counts):
        total = row.sum()
        if total <= 0:
            continue
        # Normalise each radial bin to unit area, so bins with very different
        # pair counts can be compared by shape.
        density = np.divide(
            row, total * widths, out=np.zeros_like(row), where=widths > 0
        )
        ax.stairs(
            density,
            edges,
            color=RADIAL_STYLES[index % len(RADIAL_STYLES)],
            label=f"{radial_edges[index]:g}-{radial_edges[index + 1]:g} kpc",
            lw=1.0,
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("No finite diagnostic pairs were available to plot")

    ax.set(xlabel=xlabel, ylabel="Pair density")
    ax.set_box_aspect(1)
    ax.legend(frameon=False, handlelength=1.4, fontsize=7)
    return fig


def save_stack_diagnostic_figures(
    source: StackResults | str | Path,
    output_dir: str | Path,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    extension: str = "pdf",
    dpi: int = 300,
) -> tuple[Path, ...]:
    """Write whichever radial-bin diagnostic figures the stack has arrays for.

    Returns an empty tuple when the stack ran without ``diagnostic_plots``.
    """

    import matplotlib.pyplot as plt

    results = (
        source
        if isinstance(source, StackResults)
        else load_stack_source(source, mode=mode, root=root)
    )
    if "diagnostic_radial_bin_edges" not in results.diagnostics:
        return ()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{results.stack_dir.name}_{results.mode}"
    extension = extension.lstrip(".")
    paths: list[Path] = []

    if "diagnostic_photoz_counts" in results.diagnostics:
        path = out_dir / f"{stem}_photoz_distribution.{extension}"
        fig = plot_radial_distributions(results, "photoz")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    for color in results.colors:
        if f"{color}_diagnostic_color_counts" not in results.diagnostics:
            continue
        path = out_dir / (
            f"{stem}_{color.replace('-', '_')}_color_distribution.{extension}"
        )
        fig = plot_radial_distributions(results, color)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    return tuple(paths)


@register("diagnostic_figures")
def _stage(context: PostRunContext, mode: str) -> tuple[Path, ...]:
    """Write the diagnostic figures for one stack mode."""

    return save_stack_diagnostic_figures(
        context.results(mode), context.stack_dir, root=context.root
    )


__all__ = [
    "RADIAL_STYLES",
    "plot_radial_distributions",
    "save_stack_diagnostic_figures",
]
