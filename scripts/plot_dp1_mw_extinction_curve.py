"""Plot DP1 band-relative color excesses against reference extinction curves."""

from __future__ import annotations

from pathlib import Path
import re
import sys

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from dust_extinction.averages import G03_SMCBar
from dust_extinction.parameter_averages import F99
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.dust_extinction_fit import DEFAULT_FILTER_WAVELENGTHS_UM
from dusty_colors.plotting import use_matplotlib_style


STACK_DIR = ROOT / "results" / "stacks" / "dp1_default"
STACK_PATH = STACK_DIR / "stack_fcolors.npz"
REPORT_PATH = STACK_DIR / "dust_extinction_fit_fcolors.txt"
PNG_PATH = STACK_DIR / "dp1_default_fcolors_extinction_curve_comparison.png"
PDF_PATH = STACK_DIR / "dp1_default_fcolors_extinction_curve_comparison.pdf"

# Analysis choices.
REFERENCE_BAND = "r"
PLOT_BANDS = ("g", "r", "i", "z")
POINT_BANDS = ("g", "i", "z")
RADIAL_BIN_EDGES_KPC = (10.0, 15.0, 40.0, 120.0, 1000.0)
RADIAL_PIVOT_KPC = 20.0
RADIAL_FIT_BIN_INDICES = (0, 1, 2)
CURVE_SPECS = (
    {
        "name": "MW",
        "label": r"F99 MW $R_V=3.1$",
        "law": F99(Rv=3.1),
        "color": "#1f5bd8",
        "linestyle": "-",
        "linewidth": 1.35,
    },
    {
        "name": "SMC",
        "label": r"G03 SMC Bar",
        "law": G03_SMCBar(),
        "color": "#2e7d32",
        "linestyle": "--",
        "linewidth": 1.35,
    },
)

# Aesthetic choices. These are intentionally centralized so the figure is easy
# to tune without touching the measurement or fitting logic.
FIGURE_SIZE = (5.6, 8.2)
TITLE = "DP1 default band-relative extinction curve comparison"
TITLE_Y = 0.985
XLABEL = r"$\lambda_\mathrm{rest}\ [\mu\mathrm{m}]$"
YLABEL = r"Relative extinction $A_\lambda - A_r$ [mag]"
RIGHT_YLABEL = r"Color excess relative to $r$ [mag]"
RIGHT_YLABEL_X = 0.985
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
BIN_LABEL_XY = (0.07, 0.12)
FIT_TEXT_XY = (0.50, 0.84)
FIT_TEXT_DY = 0.1
BIN_LABEL_STYLE = {"ha": "left", "va": "bottom", "fontsize": 9}
FIT_TEXT_STYLE = {"ha": "left", "va": "top", "fontsize": 7.2}


def main() -> None:
    use_matplotlib_style()

    data = np.load(STACK_PATH)
    foreground_redshift = _foreground_redshift(REPORT_PATH)
    wavelengths = dict(DEFAULT_FILTER_WAVELENGTHS_UM)

    lambda_r = wavelengths[REFERENCE_BAND]
    lambda_r_rest = _rest_wavelength(lambda_r, foreground_redshift)
    x_band = np.array(
        [
            _rest_wavelength(wavelengths[band], foreground_redshift)
            for band in PLOT_BANDS
        ]
    )
    x_points = np.array(
        [
            _rest_wavelength(wavelengths[band], foreground_redshift)
            for band in POINT_BANDS
        ]
    )

    lambda_grid = np.linspace(wavelengths["g"] - 0.025, wavelengths["z"] + 0.025, 400)
    x_grid = _rest_wavelength(lambda_grid, foreground_redshift)
    curve_specs = _prepare_curve_specs(
        wavelengths,
        lambda_grid,
        lambda_r=lambda_r,
        foreground_redshift=foreground_redshift,
    )

    radii = np.asarray(data["g-r_bin_centers"], dtype=float)
    bin_labels = _bin_labels(radii)
    y_by_bin, yerr_by_bin = _plot_points_by_bin(data, len(radii))
    for spec in curve_specs:
        spec["radial_fit"] = _fit_radial_power_law(
            radii,
            y_by_bin,
            yerr_by_bin,
            spec["point_coefficients"],
            fit_bin_indices=RADIAL_FIT_BIN_INDICES,
        )

    fig, axes = plt.subplots(
        nrows=len(radii),
        ncols=1,
        figsize=FIGURE_SIZE,
        sharex=True,
        constrained_layout=False,
    )
    axes = np.atleast_1d(axes)

    for index, ax in enumerate(axes):
        y, yerr = y_by_bin[index], yerr_by_bin[index]

        ax.axhline(0.0, **ZERO_LINE_KWARGS)
        plotted_curves = []
        for spec in curve_specs:
            fit = spec["radial_fit"]
            av = fit["amplitude"] * (radii[index] / RADIAL_PIVOT_KPC) ** fit["alpha"]
            curve = av * spec["grid_coefficients"]
            plotted_curves.append(curve)
            ax.plot(
                x_grid,
                curve,
                color=spec["color"],
                ls=spec["linestyle"],
                lw=spec["linewidth"],
                label=spec["label"],
            )
        ax.errorbar(
            x_points,
            y,
            yerr=yerr,
            **DATA_KWARGS,
        )
        ax.plot(
            lambda_r_rest,
            0.0,
            **REFERENCE_MARKER_KWARGS,
        )

        _set_y_limits(ax, y, yerr, plotted_curves)
        ax.set_xlim(x_grid.min(), x_grid.max())

        ax.text(
            *BIN_LABEL_XY,
            bin_labels[index],
            transform=ax.transAxes,
            **BIN_LABEL_STYLE,
        )
        if index == 0:
            _draw_radial_fit_summary(ax, curve_specs)
        _add_color_excess_axis(ax)
        ax.minorticks_on()
        ax.tick_params(axis="both", which="both", direction="in", top=True)

    axes[0].legend(**LEGEND_KWARGS)
    top = axes[0].secondary_xaxis("top")
    top.set_xticks(x_band)
    top.set_xticklabels([rf"${band}$" for band in PLOT_BANDS])
    top.tick_params(direction="in")

    fig.supxlabel(XLABEL, y=0.04)
    fig.supylabel(YLABEL, x=0.055)
    fig.text(
        RIGHT_YLABEL_X,
        0.5,
        RIGHT_YLABEL,
        va="center",
        ha="center",
        rotation=-90,
    )
    fig.suptitle(TITLE, y=TITLE_Y, fontsize=11)
    fig.subplots_adjust(**SUBPLOTS_ADJUST)

    fig.savefig(PNG_PATH, dpi=240)
    fig.savefig(PDF_PATH)
    for spec in curve_specs:
        fit = spec["radial_fit"]
        print(
            f"{spec['name']}: "
            f"A_V({RADIAL_PIVOT_KPC:.0f} kpc)={fit['amplitude']:.6g}, "
            f"alpha={fit['alpha']:.6g}, "
            f"chi2/dof={fit['chi2']:.6g}/{fit['dof']}"
        )
    print(PNG_PATH.relative_to(ROOT))
    print(PDF_PATH.relative_to(ROOT))


def _foreground_redshift(path: Path) -> float:
    match = re.search(
        r"^  foreground_redshift:\s*([0-9.eE+-]+)",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return float(match.group(1)) if match else 0.0


def _rest_wavelength(
    lambda_obs_um: float | np.ndarray,
    foreground_redshift: float,
) -> float | np.ndarray:
    lambda_rest = np.asarray(lambda_obs_um, dtype=float) / (1.0 + foreground_redshift)
    if lambda_rest.ndim == 0:
        return float(lambda_rest)
    return lambda_rest


def _bin_labels(radii: np.ndarray) -> list[str]:
    r_edges = np.asarray(RADIAL_BIN_EDGES_KPC, dtype=float)
    if len(r_edges) != len(radii) + 1:
        return [rf"$R_p \simeq {radius:.0f}\,\mathrm{{kpc}}$" for radius in radii]
    return [
        rf"${r_edges[index]:.0f} < R_p < {r_edges[index + 1]:.0f}\,\mathrm{{kpc}}$"
        for index in range(len(radii))
    ]


def _prepare_curve_specs(
    wavelengths: dict[str, float],
    lambda_grid: np.ndarray,
    *,
    lambda_r: float,
    foreground_redshift: float,
) -> list[dict[str, object]]:
    prepared = []
    point_wavelengths = [wavelengths[band] for band in POINT_BANDS]
    for spec in CURVE_SPECS:
        curve = dict(spec)
        curve["point_coefficients"] = _curve_coefficients(
            curve["law"],
            point_wavelengths,
            lambda_r=lambda_r,
            foreground_redshift=foreground_redshift,
        )
        curve["grid_coefficients"] = _curve_coefficients(
            curve["law"],
            lambda_grid,
            lambda_r=lambda_r,
            foreground_redshift=foreground_redshift,
        )
        prepared.append(curve)
    return prepared


def _curve_coefficients(
    law: object,
    lambda_obs_um: float | np.ndarray,
    *,
    lambda_r: float,
    foreground_redshift: float,
) -> np.ndarray:
    """Return A(lambda)-A(r), normalized by A_V, at dust-rest wavelengths."""

    lambda_obs = np.asarray(lambda_obs_um, dtype=float)
    lambda_rest = lambda_obs / (1.0 + foreground_redshift)
    lambda_r_rest = float(lambda_r) / (1.0 + foreground_redshift)
    ratios = np.asarray(law((1.0 / lambda_rest) * u.micron**-1), dtype=float)
    r_ratio = float(law((1.0 / lambda_r_rest) * u.micron**-1))
    return ratios - r_ratio


def _relative_color_points(
    data: np.lib.npyio.NpzFile,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return g-r, i-r, and z-r points with jackknife-propagated errors."""

    g_r = float(data["g-r_avg"][index])
    r_i = float(data["r-i_avg"][index])
    i_z = float(data["i-z_avg"][index])
    y = np.array([g_r, -r_i, -(r_i + i_z)], dtype=float)

    samples = np.column_stack(
        [
            data["g-r_jackknife_samples"][:, index],
            -data["r-i_jackknife_samples"][:, index],
            -(
                data["r-i_jackknife_samples"][:, index]
                + data["i-z_jackknife_samples"][:, index]
            ),
        ]
    )
    centered = samples - samples.mean(axis=0)
    covariance = (1.0 - 1.0 / samples.shape[0]) * centered.T @ centered
    yerr = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    return y, yerr


def _plot_points_by_bin(
    data: np.lib.npyio.NpzFile,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = []
    errors = []
    for index in range(n_bins):
        y, yerr = _relative_color_points(data, index)
        points.append(y)
        errors.append(yerr)
    return np.asarray(points, dtype=float), np.asarray(errors, dtype=float)


def _fit_radial_power_law(
    radii: np.ndarray,
    y_by_bin: np.ndarray,
    yerr_by_bin: np.ndarray,
    model_coefficients: np.ndarray,
    *,
    fit_bin_indices: tuple[int, ...],
) -> dict[str, float | int | bool]:
    fit_bin_mask = np.zeros(len(radii), dtype=bool)
    fit_bin_mask[list(fit_bin_indices)] = True
    finite = (
        np.isfinite(radii[:, np.newaxis])
        & np.isfinite(y_by_bin)
        & np.isfinite(yerr_by_bin)
        & (radii[:, np.newaxis] > 0)
        & (yerr_by_bin > 0)
        & fit_bin_mask[:, np.newaxis]
    )

    def model(params: np.ndarray) -> np.ndarray:
        amplitude, alpha = params
        radial = amplitude * (radii / RADIAL_PIVOT_KPC) ** alpha
        return radial[:, np.newaxis] * model_coefficients[np.newaxis, :]

    def residual(params: np.ndarray) -> np.ndarray:
        return ((y_by_bin - model(params)) / yerr_by_bin)[finite]

    optimized = least_squares(
        residual,
        x0=np.array([0.03, -1.0], dtype=float),
        bounds=([0.0, -5.0], [np.inf, 1.0]),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=2000,
    )
    chi2 = float(np.sum(residual(optimized.x) ** 2))
    dof = max(int(np.count_nonzero(finite)) - len(optimized.x), 0)
    parameter_covariance = np.linalg.pinv(optimized.jac.T @ optimized.jac)
    parameter_errors = np.sqrt(np.clip(np.diag(parameter_covariance), 0.0, np.inf))
    return {
        "amplitude": float(optimized.x[0]),
        "alpha": float(optimized.x[1]),
        "amplitude_error": float(parameter_errors[0]),
        "alpha_error": float(parameter_errors[1]),
        "chi2": chi2,
        "dof": dof,
        "optimizer_success": bool(optimized.success),
    }


def _draw_radial_fit_summary(
    ax: plt.Axes,
    curve_specs: list[dict[str, object]],
) -> None:
    for text_index, spec in enumerate(curve_specs):
        fit = spec["radial_fit"]
        ax.text(
            FIT_TEXT_XY[0],
            FIT_TEXT_XY[1] - FIT_TEXT_DY * text_index,
            rf"{spec['name']}: $A_{{20}}={fit['amplitude']:.4f}$, "
            + rf"$\alpha={fit['alpha']:.2f}$, "
            + rf"$\chi^2_\nu={fit['chi2'] / fit['dof']:.2f}$",
            transform=ax.transAxes,
            color=spec["color"],
            **FIT_TEXT_STYLE,
        )


def _add_color_excess_axis(ax: plt.Axes) -> None:
    right = ax.secondary_yaxis("right", functions=(lambda y: y, lambda y: y))
    right.minorticks_on()
    right.tick_params(axis="y", which="both", direction="in")


def _set_y_limits(
    ax: plt.Axes,
    y: np.ndarray,
    yerr: np.ndarray,
    curves: list[np.ndarray],
) -> None:
    y_all = np.concatenate([y - yerr, y + yerr, *curves, [0.0]])
    finite = y_all[np.isfinite(y_all)]
    ymin, ymax = float(finite.min()), float(finite.max())
    pad = max(0.12 * (ymax - ymin), 0.0008)
    ax.set_ylim(ymin - pad, ymax + pad)


if __name__ == "__main__":
    main()
