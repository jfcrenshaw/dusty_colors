"""Band-relative color excesses compared against reference extinction curves.

One panel per radial bin, each showing the measured color excesses relative to
a reference band alongside the extinction laws being tested. This is the figure
that shows whether the reddening is chromatically consistent with Milky Way
dust or with a steeper SMC-like curve.

Ported from ``scripts/plot_dp1_mw_extinction_curve.py``, which hardcoded the
analysis it ran on. Everything that is a science choice now comes from the
``postrun.chromaticity`` block.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..results import StackResults
from ..utils import use_matplotlib_style
from .base import PostRunContext, register
from .dust_extinction_fit import DEFAULT_FILTER_WAVELENGTHS_UM, _dust_law

DEFAULT_BANDS = ("g", "r", "i", "z")
DEFAULT_REFERENCE_BAND = "r"
DEFAULT_POINT_BANDS = ("g", "i", "z")
DEFAULT_RADIAL_PIVOT_KPC = 20.0

# Reproduces the two curves the paper figure compares: Milky Way dust against a
# markedly steeper SMC bar curve.
DEFAULT_LAWS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "F99",
        "rv": 3.1,
        "short": "MW",
        "label": r"F99 MW $R_V=3.1$",
        "color": "#1f5bd8",
        "linestyle": "-",
    },
    {
        "name": "G03_SMCBar",
        "short": "SMC",
        "label": r"G03 SMC Bar",
        "color": "#2e7d32",
        "linestyle": "--",
    },
)

# Aesthetic choices, kept together so the figure can be tuned without touching
# the measurement or the fit.
FIGURE_SIZE = (5.6, 8.2)
XLABEL = r"$\lambda_\mathrm{rest}\ [\mu\mathrm{m}]$"
YLABEL = r"Relative extinction $A_\lambda - A_\mathrm{ref}$ [mag]"
RIGHT_YLABEL = r"Color excess relative to reference band [mag]"
SUBPLOTS_ADJUST = {
    "left": 0.16,
    "right": 0.88,
    "top": 0.93,
    "bottom": 0.095,
    "hspace": 0.045,
}
LEGEND_KWARGS = {"loc": "upper left", "frameon": False, "fontsize": 8}
ZERO_LINE_KWARGS = {"color": "0.74", "lw": 0.7, "ls": (0, (5, 5)), "zorder": 0}
DATA_KWARGS = {
    "fmt": "o",
    "color": "0.05",
    "ecolor": "0.15",
    "elinewidth": 0.9,
    "capsize": 2.5,
    "ms": 3.5,
    "zorder": 3,
}
REFERENCE_MARKER_KWARGS = {
    "marker": "x",
    "color": "0.05",
    "ms": 5.0,
    "mew": 0.9,
    "zorder": 4,
}
LINEWIDTH = 1.35
BIN_LABEL_XY = (0.07, 0.12)
FIT_TEXT_XY = (0.50, 0.84)
FIT_TEXT_DY = 0.1
BIN_LABEL_STYLE = {"ha": "left", "va": "bottom", "fontsize": 9}
FIT_TEXT_STYLE = {"ha": "left", "va": "top", "fontsize": 7.2}
GRID_PAD_UM = 0.025
GRID_POINTS = 400


@dataclass(frozen=True)
class ChromaticityConfig:
    """Resolved options for the chromaticity figure."""

    enabled: bool = True
    bands: tuple[str, ...] = DEFAULT_BANDS
    reference_band: str = DEFAULT_REFERENCE_BAND
    point_bands: tuple[str, ...] = DEFAULT_POINT_BANDS
    radial_pivot_kpc: float = DEFAULT_RADIAL_PIVOT_KPC
    fit_bin_indices: tuple[int, ...] | None = None
    laws: tuple[Mapping[str, Any], ...] = DEFAULT_LAWS
    wavelengths_um: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_FILTER_WAVELENGTHS_UM)
    )
    extension: tuple[str, ...] = ("png", "pdf")
    dpi: int = 240


@dataclass(frozen=True)
class RadialPowerLawFit:
    """Amplitude and slope of ``A_V(r)`` for one extinction law."""

    amplitude: float
    alpha: float
    amplitude_error: float
    alpha_error: float
    chi2: float
    dof: int
    optimizer_success: bool


def parse_chromaticity_options(raw: Any) -> ChromaticityConfig:
    """Parse a ``chromaticity`` block with defaults matching the paper figure."""

    if raw is False:
        return ChromaticityConfig(enabled=False)
    if raw is True or raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("chromaticity must be a mapping or boolean")
    if not bool(raw.get("enabled", True)):
        return ChromaticityConfig(enabled=False)

    wavelengths = dict(DEFAULT_FILTER_WAVELENGTHS_UM)
    wavelengths.update(
        {str(k): float(v) for k, v in (raw.get("wavelengths_um") or {}).items()}
    )
    fit_bin_indices = raw.get("fit_bin_indices")

    return ChromaticityConfig(
        bands=_str_tuple(raw.get("bands", DEFAULT_BANDS)),
        reference_band=str(raw.get("reference_band", DEFAULT_REFERENCE_BAND)),
        point_bands=_str_tuple(raw.get("point_bands", DEFAULT_POINT_BANDS)),
        radial_pivot_kpc=float(raw.get("radial_pivot_kpc", DEFAULT_RADIAL_PIVOT_KPC)),
        fit_bin_indices=(
            None if fit_bin_indices is None else tuple(int(i) for i in fit_bin_indices)
        ),
        laws=tuple(raw.get("laws", DEFAULT_LAWS)),
        wavelengths_um=wavelengths,
        extension=_str_tuple(raw.get("extension", ("png", "pdf"))),
        dpi=int(raw.get("dpi", 240)),
    )


def band_relative_chain(
    bands: Sequence[str],
    reference_band: str,
    target_band: str,
) -> tuple[tuple[str, float], ...]:
    """Return the signed adjacent colors summing to ``target - reference``.

    The stack measures adjacent-band colors (``g-r``, ``r-i``, ``i-z``), so a
    band-relative excess is a walk along that chain. Walking bluewards adds the
    colors as measured; walking redwards subtracts them, because each stored
    color is ``bluer - redder``.

    Parameters
    ----------
    bands : sequence of str
        Bands in wavelength order.
    reference_band, target_band : str
        Endpoints of the walk.

    Returns
    -------
    tuple of (str, float)
        ``(color_name, sign)`` pairs; empty when the endpoints coincide.
    """

    try:
        start = bands.index(reference_band)
        stop = bands.index(target_band)
    except ValueError as exc:
        raise ValueError(
            f"chromaticity bands {tuple(bands)} must contain both "
            f"{reference_band!r} and {target_band!r}"
        ) from exc

    if stop < start:
        return tuple((f"{bands[i]}-{bands[i + 1]}", 1.0) for i in range(stop, start))
    return tuple((f"{bands[i]}-{bands[i + 1]}", -1.0) for i in range(start, stop))


def band_relative_points(
    results: StackResults,
    config: ChromaticityConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return band-relative excesses and jackknife errors, per radial bin.

    Returns
    -------
    values, errors : ndarray
        Both shaped ``(n_radial_bins, n_point_bands)``.
    """

    chains = [
        band_relative_chain(config.bands, config.reference_band, band)
        for band in config.point_bands
    ]

    values = np.column_stack(
        [
            sum(sign * results.require(f"{color}_avg") for color, sign in chain)
            for chain in chains
        ]
    )

    # Propagate through the jackknife samples rather than adding errors in
    # quadrature: the adjacent colors share background objects, so their errors
    # are correlated and quadrature would overstate them.
    samples = np.stack(
        [
            sum(
                sign * results.require(f"{color}_jackknife_samples")
                for color, sign in chain
            )
            for chain in chains
        ],
        axis=-1,
    )
    n_patches = samples.shape[0]
    centered = samples - samples.mean(axis=0, keepdims=True)
    variance = (1.0 - 1.0 / n_patches) * np.sum(centered**2, axis=0)
    return values, np.sqrt(np.clip(variance, 0.0, np.inf))


def fit_radial_power_law(
    radii: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    coefficients: np.ndarray,
    *,
    pivot_kpc: float,
    fit_bin_indices: Sequence[int] | None = None,
) -> RadialPowerLawFit:
    """Fit ``A_V(r) = A (r/pivot)^alpha`` with the law's color shape held fixed.

    The chromatic shape is fixed by ``coefficients``, so the only freedom is the
    overall amplitude and its radial slope. That is what makes the figure a test
    of the law rather than a fit to it.
    """

    from scipy.optimize import least_squares

    fit_bin_mask = np.zeros(len(radii), dtype=bool)
    if fit_bin_indices is None:
        fit_bin_mask[:] = True
    else:
        fit_bin_mask[list(fit_bin_indices)] = True

    finite = (
        np.isfinite(radii[:, np.newaxis])
        & np.isfinite(values)
        & np.isfinite(errors)
        & (radii[:, np.newaxis] > 0)
        & (errors > 0)
        & fit_bin_mask[:, np.newaxis]
    )

    def model(params: np.ndarray) -> np.ndarray:
        amplitude, alpha = params
        radial = amplitude * (radii / pivot_kpc) ** alpha
        return radial[:, np.newaxis] * coefficients[np.newaxis, :]

    def residual(params: np.ndarray) -> np.ndarray:
        return ((values - model(params)) / errors)[finite]

    optimized = least_squares(
        residual,
        x0=np.array([0.03, -1.0], dtype=float),
        bounds=([0.0, -5.0], [np.inf, 1.0]),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=2000,
    )
    parameter_covariance = np.linalg.pinv(optimized.jac.T @ optimized.jac)
    parameter_errors = np.sqrt(np.clip(np.diag(parameter_covariance), 0.0, np.inf))
    return RadialPowerLawFit(
        amplitude=float(optimized.x[0]),
        alpha=float(optimized.x[1]),
        amplitude_error=float(parameter_errors[0]),
        alpha_error=float(parameter_errors[1]),
        chi2=float(np.sum(residual(optimized.x) ** 2)),
        dof=max(int(np.count_nonzero(finite)) - len(optimized.x), 0),
        optimizer_success=bool(optimized.success),
    )


def save_stack_chromaticity_figure(
    results: StackResults,
    output_dir: str | Path,
    *,
    config: ChromaticityConfig | None = None,
    foreground_redshift: float | None = None,
    radial_bin_edges: Sequence[float] | None = None,
) -> tuple[Path, ...]:
    """Draw and save the chromaticity comparison figure.

    Returns an empty tuple when the stack lacks the adjacent colors the
    requested bands need, matching how the other fit stages decline to run.
    """

    import matplotlib.pyplot as plt

    config = config or ChromaticityConfig()
    needed = {
        f"{color}_{suffix}"
        for band in config.point_bands
        for color, _ in band_relative_chain(config.bands, config.reference_band, band)
        for suffix in ("avg", "jackknife_samples")
    }
    if not config.enabled or not needed <= results.arrays.keys():
        return ()

    redshift = 0.0 if foreground_redshift is None else float(foreground_redshift)
    use_matplotlib_style()

    values, errors = band_relative_points(results, config)
    first_color = f"{config.bands[0]}-{config.bands[1]}"
    radii = np.asarray(results.require(f"{first_color}_bin_centers"), dtype=float)

    wavelengths = config.wavelengths_um
    lambda_ref = wavelengths[config.reference_band]
    x_points = np.array(
        [wavelengths[band] / (1.0 + redshift) for band in config.point_bands]
    )
    lambda_grid = np.linspace(
        min(wavelengths[band] for band in config.bands) - GRID_PAD_UM,
        max(wavelengths[band] for band in config.bands) + GRID_PAD_UM,
        GRID_POINTS,
    )
    x_grid = lambda_grid / (1.0 + redshift)

    # One (spec, grid coefficients, fit) triple per law being compared. The
    # chromatic shape is fixed by the law, so only the amplitude and slope fit.
    curves = []
    for spec in config.laws:
        law = _dust_law(str(spec["name"]), float(spec.get("rv", 3.1)))
        curves.append(
            (
                spec,
                _curve_coefficients(
                    law, lambda_grid, lambda_ref=lambda_ref, redshift=redshift
                ),
                fit_radial_power_law(
                    radii,
                    values,
                    errors,
                    _curve_coefficients(
                        law,
                        [wavelengths[band] for band in config.point_bands],
                        lambda_ref=lambda_ref,
                        redshift=redshift,
                    ),
                    pivot_kpc=config.radial_pivot_kpc,
                    fit_bin_indices=config.fit_bin_indices,
                ),
            )
        )

    fig, axes = plt.subplots(
        nrows=len(radii),
        ncols=1,
        figsize=FIGURE_SIZE,
        sharex=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)

    edges = None if radial_bin_edges is None else np.asarray(radial_bin_edges, float)
    if edges is not None and len(edges) == len(radii) + 1:
        bin_labels = [
            rf"${edges[i]:.0f} < R_p < {edges[i + 1]:.0f}\,\mathrm{{kpc}}$"
            for i in range(len(radii))
        ]
    else:
        bin_labels = [rf"$R_p \simeq {r:.0f}\,\mathrm{{kpc}}$" for r in radii]

    for index, ax in enumerate(axes):
        ax.axhline(0.0, **ZERO_LINE_KWARGS)
        plotted = []
        for spec, coefficients, fit in curves:
            av = fit.amplitude * (radii[index] / config.radial_pivot_kpc) ** fit.alpha
            model = av * coefficients
            plotted.append(model)
            ax.plot(
                x_grid,
                model,
                color=spec.get("color"),
                ls=spec.get("linestyle", "-"),
                lw=LINEWIDTH,
                label=spec.get("label", spec["name"]),
            )
        ax.errorbar(x_points, values[index], yerr=errors[index], **DATA_KWARGS)
        ax.plot(lambda_ref / (1.0 + redshift), 0.0, **REFERENCE_MARKER_KWARGS)

        # Pad the limits around the data, the curves, and the zero line together.
        stacked = np.concatenate(
            [
                values[index] - errors[index],
                values[index] + errors[index],
                *plotted,
                [0.0],
            ]
        )
        finite = stacked[np.isfinite(stacked)]
        ymin, ymax = float(finite.min()), float(finite.max())
        pad = max(0.12 * (ymax - ymin), 0.0008)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlim(x_grid.min(), x_grid.max())
        ax.text(
            *BIN_LABEL_XY,
            bin_labels[index],
            transform=ax.transAxes,
            **BIN_LABEL_STYLE,
        )

        if index == 0:
            for text_index, (spec, _, fit) in enumerate(curves):
                reduced = fit.chi2 / fit.dof if fit.dof else float("nan")
                ax.text(
                    FIT_TEXT_XY[0],
                    FIT_TEXT_XY[1] - FIT_TEXT_DY * text_index,
                    rf"{spec.get('short', spec['name'])}: $A={fit.amplitude:.4f}$, "
                    rf"$\alpha={fit.alpha:.2f}$, $\chi^2_\nu={reduced:.2f}$",
                    transform=ax.transAxes,
                    color=spec.get("color"),
                    **FIT_TEXT_STYLE,
                )

        # Mirror the y axis on the right, where the same numbers read as the
        # measured color excess rather than as relative extinction.
        right = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
        right.minorticks_on()
        right.tick_params(axis="y", which="both", direction="in")
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in", top=True)

    axes[0].legend(**LEGEND_KWARGS)
    top = axes[0].secondary_xaxis("top")
    top.set_xticks([wavelengths[band] / (1.0 + redshift) for band in config.bands])
    top.set_xticklabels([rf"${band}$" for band in config.bands])
    top.tick_params(direction="in")

    fig.supxlabel(XLABEL, y=0.04)
    fig.supylabel(YLABEL, x=0.055)
    fig.text(0.985, 0.5, RIGHT_YLABEL, va="center", ha="center", rotation=-90)
    # Escape underscores so analysis ids do not become mathtext subscripts.
    title = f"{results.stack_dir.name} {results.mode}".replace("_", r"\_")
    fig.suptitle(
        f"{title} band-relative extinction curve comparison",
        y=0.985,
        fontsize=11,
    )
    fig.subplots_adjust(**SUBPLOTS_ADJUST)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{results.stack_dir.name}_{results.mode}_extinction_curve_comparison"
    paths = []
    for extension in config.extension:
        path = out_dir / f"{stem}.{extension.lstrip('.')}"
        fig.savefig(path, dpi=config.dpi)
        paths.append(path)
    plt.close(fig)
    return tuple(paths)


def _curve_coefficients(
    law: Any,
    lambda_obs_um: float | Sequence[float] | np.ndarray,
    *,
    lambda_ref: float,
    redshift: float,
) -> np.ndarray:
    """Return ``A(lambda) - A(ref)`` per unit ``A_V`` at dust-frame wavelengths.

    The dust sits at the foreground redshift, so the observed filters must be
    blueshifted into the dust frame before the law is evaluated.
    """

    import astropy.units as u

    lambda_rest = np.asarray(lambda_obs_um, dtype=float) / (1.0 + redshift)
    lambda_ref_rest = float(lambda_ref) / (1.0 + redshift)
    ratios = np.asarray(law((1.0 / lambda_rest) * u.micron**-1), dtype=float)
    return ratios - float(law((1.0 / lambda_ref_rest) * u.micron**-1))


def _str_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


@register("chromaticity")
def _stage(context: PostRunContext, mode: str) -> tuple[Path, ...]:
    """Draw the band-relative extinction curve comparison for one stack mode."""

    redshift, _ = context.foreground_redshift
    return save_stack_chromaticity_figure(
        context.results(mode),
        context.stack_dir,
        config=parse_chromaticity_options(context.options("chromaticity")),
        foreground_redshift=redshift,
        radial_bin_edges=context.stack_config.get("r_bin_edges"),
    )


__all__ = [
    "ChromaticityConfig",
    "RadialPowerLawFit",
    "band_relative_chain",
    "band_relative_points",
    "fit_radial_power_law",
    "parse_chromaticity_options",
    "save_stack_chromaticity_figure",
]
