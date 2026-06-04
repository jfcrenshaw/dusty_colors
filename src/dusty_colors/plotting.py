"""Plotting helpers for TreeCorr stack outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import warnings

import numpy as np
import yaml

from .config import load_resolved_config
from .pipeline import build_stage_specs, stack_modes

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


@dataclass(frozen=True)
class StackResults:
    """Loaded stack arrays with the config metadata needed for plotting."""

    stack_dir: Path
    mode: str
    colors: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    config_path: Path | None = None

    @property
    def first_color(self) -> str:
        if not self.colors:
            raise ValueError("Stack results do not define any colors")
        return self.colors[0]

    def require(self, key: str) -> np.ndarray:
        try:
            return self.arrays[key]
        except KeyError as exc:
            raise KeyError(
                f"{self.stack_dir / f'stack_{self.mode}.npz'} is missing {key!r}"
            ) from exc


def default_style_path() -> Path:
    """Return the package-local Matplotlib style path."""

    path = Path(__file__).with_name("matplotlibrc")
    if path.exists():
        return path

    fallback = Path(__file__).resolve().parents[2] / "notebooks" / "matplotlibrc"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Could not find dusty_colors matplotlibrc")


def use_matplotlib_style(style_path: str | Path | None = None) -> Path:
    """Apply the project Matplotlib settings and return the style path used."""

    path = default_style_path() if style_path is None else Path(style_path)
    import matplotlib as mpl

    mpl.rc_file(path)
    return path


def load_stack_results(
    analysis_config: str | Path | None = None,
    *,
    stack_dir: str | Path | None = None,
    mode: str | None = None,
    root: str | Path | None = None,
    colors: Sequence[str] | None = None,
) -> StackResults:
    """Load one ``stack_<mode>.npz`` file plus plotting metadata.

    Passing an analysis YAML is the most reproducible path: the loader reads the
    configured ``stack.colors`` order and resolves the canonical
    ``results/stacks/<analysis-id>`` directory. Passing ``stack_dir`` directly is
    useful for ad hoc outputs and will infer colors from ``config_resolved.yaml``
    or the NPZ keys.
    """

    config_path: Path | None = None
    configured_colors = tuple(str(color) for color in colors or ())
    configured_modes: tuple[str, ...] = ()

    if analysis_config is not None:
        resolved = load_resolved_config(analysis_config, root=root)
        config_path = resolved.analysis.path
        stack_config = resolved.analysis.data.get("stack", {})
        configured_colors = configured_colors or _stack_colors(stack_config)
        configured_modes = stack_modes(resolved.analysis)
        if stack_dir is None:
            stack_dir = build_stage_specs(resolved, root=resolved.root)[
                "stack"
            ].output_dir

    if stack_dir is None:
        raise ValueError("Provide either analysis_config or stack_dir")

    stack_path = Path(stack_dir).resolve()
    if not configured_colors:
        stack_config = _load_stack_config(stack_path)
        configured_colors = _stack_colors(stack_config)
        configured_modes = tuple(
            str(candidate) for candidate in stack_config.get("modes", ())
        )

    selected_mode = _resolve_mode(stack_path, mode, configured_modes)
    arrays = _read_stack_npz(stack_path / f"stack_{selected_mode}.npz")
    if not configured_colors:
        configured_colors = _infer_colors(arrays)
    if not configured_colors:
        raise ValueError(f"Could not infer stack colors from {stack_path}")

    return StackResults(
        stack_dir=stack_path,
        mode=selected_mode,
        colors=configured_colors,
        arrays=arrays,
        config_path=config_path,
    )


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


def _read_stack_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _resolve_mode(
    stack_dir: Path,
    mode: str | None,
    configured_modes: Sequence[str],
) -> str:
    if mode is not None:
        return str(mode)
    if configured_modes:
        return str(configured_modes[0])
    if (stack_dir / "stack_fcolors.npz").exists():
        return "fcolors"
    if (stack_dir / "stack_mcolors.npz").exists():
        return "mcolors"

    matches = sorted(stack_dir.glob("stack_*.npz"))
    if len(matches) == 1:
        return matches[0].stem.removeprefix("stack_")
    raise FileNotFoundError(f"Could not choose a stack mode in {stack_dir}")


def _load_stack_config(stack_dir: Path) -> dict[str, Any]:
    path = stack_dir / "config_resolved.yaml"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        return {}

    analysis = data.get("analysis")
    if isinstance(analysis, Mapping):
        analysis_data = analysis.get("data")
        if isinstance(analysis_data, Mapping):
            stack = analysis_data.get("stack", {})
            return dict(stack) if isinstance(stack, Mapping) else {}

    stack = data.get("stack")
    if isinstance(stack, Mapping):
        return dict(stack)
    return dict(data)


def _stack_colors(stack_config: Mapping[str, Any]) -> tuple[str, ...]:
    colors = stack_config.get("colors", ())
    if not isinstance(colors, Sequence) or isinstance(colors, str):
        return ()
    return tuple(str(color) for color in colors)


def _infer_colors(arrays: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    suffix = "_bin_centers"
    return tuple(key[: -len(suffix)] for key in arrays if key.endswith(suffix))


def _color_style(color: str, index: int) -> str:
    return DEFAULT_COLOR_STYLES.get(color, f"C{index}")


def _legend_color_label(color: str) -> str:
    return f"${color}$"


def _single_color_ylabel(color: str) -> str:
    return rf"$E({color})$ [mag]"


def _slug(value: str) -> str:
    return value.replace("-", "_").replace("/", "_")


__all__ = [
    "StackResults",
    "default_style_path",
    "load_stack_results",
    "plot_all_color_signals",
    "plot_first_color_jackknife",
    "save_stack_figures",
    "use_matplotlib_style",
]
