"""Fit direct radial power laws to stacked color excess profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from .plotting import StackResults, load_stack_results


@dataclass(frozen=True)
class ColorPowerLawFit:
    """Best-fit ``E(color) = A * (r / pivot)^alpha`` for one color."""

    color: str
    amplitude_mag: float
    alpha: float
    parameter_errors: np.ndarray
    parameter_covariance: np.ndarray
    chi2: float
    dof: int
    p_value: float
    n_data: int
    optimizer_success: bool
    optimizer_message: str

    @property
    def reduced_chi2(self) -> float:
        if self.dof <= 0:
            return np.nan
        return self.chi2 / self.dof


def save_stack_color_power_law_fits(
    source: StackResults | str | Path,
    output_dir: str | Path | None = None,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    stack_config: Mapping[str, Any] | None = None,
) -> Path | None:
    """Fit every available stack color and write a compact text report."""

    config = _power_law_config(stack_config)
    if not bool(config.get("enabled", True)):
        return None

    results = (
        source
        if isinstance(source, StackResults)
        else _load_stack_source(source, mode=mode, root=root)
    )
    pivot_kpc = float(config.get("radial_pivot_kpc", 100.0))
    colors = _configured_colors(config, results.colors)

    fits = [
        fit_color_power_law(results, color, pivot_kpc=pivot_kpc, config=config)
        for color in colors
        if _has_color_profile(results, color)
    ]
    fits = [fit for fit in fits if fit is not None]
    if not fits:
        return None

    out_dir = results.stack_dir if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"color_power_law_fits_{results.mode}.txt"
    path.write_text(
        format_color_power_law_fits(results.mode, fits, pivot_kpc=pivot_kpc),
        encoding="utf-8",
    )
    return path


def fit_color_power_law(
    results: StackResults,
    color: str,
    *,
    pivot_kpc: float = 100.0,
    config: Mapping[str, Any] | None = None,
) -> ColorPowerLawFit | None:
    """Fit one color profile in linear color-excess space."""

    if pivot_kpc <= 0:
        raise ValueError("color_power_law_fit.radial_pivot_kpc must be positive")

    radius = np.asarray(results.require(f"{color}_bin_centers"), dtype=float)
    signal = np.asarray(results.require(f"{color}_avg"), dtype=float)
    error = _profile_errors(results, color, len(radius))
    good = (
        np.isfinite(radius)
        & (radius > 0)
        & np.isfinite(signal)
        & np.isfinite(error)
        & (error > 0)
    )
    if np.count_nonzero(good) < 3:
        return None

    radius = radius[good]
    signal = signal[good]
    error = error[good]
    bounds = _fit_bounds(config)
    p0 = _initial_parameters(radius, signal, pivot_kpc, bounds)

    def residual(params: np.ndarray) -> np.ndarray:
        return (signal - _model(radius, params[0], params[1], pivot_kpc)) / error

    optimized = least_squares(
        residual,
        p0,
        bounds=bounds,
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=1000,
    )
    covariance = _parameter_covariance(optimized.jac)
    errors = np.sqrt(np.clip(np.diag(covariance), 0.0, np.inf))
    chi2 = float(np.sum(residual(optimized.x) ** 2))
    dof = max(len(signal) - 2, 0)
    p_value = float(chi2_distribution.sf(chi2, dof)) if dof > 0 else np.nan
    return ColorPowerLawFit(
        color=color,
        amplitude_mag=float(optimized.x[0]),
        alpha=float(optimized.x[1]),
        parameter_errors=errors,
        parameter_covariance=covariance,
        chi2=chi2,
        dof=dof,
        p_value=p_value,
        n_data=len(signal),
        optimizer_success=bool(optimized.success),
        optimizer_message=str(optimized.message),
    )


def format_color_power_law_fits(
    mode: str,
    fits: Sequence[ColorPowerLawFit],
    *,
    pivot_kpc: float,
) -> str:
    """Format direct color power-law fits as a text report."""

    lines = [
        "Direct color power-law fits",
        f"created_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"mode: {mode}",
        "",
        "model:",
        f"  E(color, r) = A * (r / {pivot_kpc:g} kpc)^alpha",
        "",
        "fits:",
    ]
    for fit in fits:
        lines.extend(
            [
                f"  {fit.color}:",
                f"    amplitude_at_pivot_mag: {fit.amplitude_mag:.10g} "
                f"+/- {fit.parameter_errors[0]:.3g}",
                f"    alpha: {fit.alpha:.10g} +/- {fit.parameter_errors[1]:.3g}",
                f"    chi2: {fit.chi2:.10g}",
                f"    dof: {fit.dof}",
                f"    reduced_chi2: {fit.reduced_chi2:.10g}",
                f"    p_value: {fit.p_value:.10g}",
                f"    n_data: {fit.n_data}",
                f"    optimizer_success: {fit.optimizer_success}",
                f"    optimizer_message: {fit.optimizer_message}",
            ]
        )
    return "\n".join(lines) + "\n"


def _power_law_config(stack_config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    stack_config = stack_config or {}
    raw = stack_config.get("color_power_law_fit", {})
    if raw is False:
        return {"enabled": False}
    if raw is True or raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("stack.color_power_law_fit must be a mapping or boolean")
    return raw


def _configured_colors(
    config: Mapping[str, Any],
    default_colors: Sequence[str],
) -> tuple[str, ...]:
    colors = config.get("colors", default_colors)
    if isinstance(colors, str):
        return (colors,)
    return tuple(str(color) for color in colors)


def _has_color_profile(results: StackResults, color: str) -> bool:
    return f"{color}_bin_centers" in results.arrays and f"{color}_avg" in results.arrays


def _load_stack_source(
    source: str | Path,
    *,
    mode: str | None,
    root: str | Path | None,
) -> StackResults:
    path = Path(source)
    if path.is_dir():
        return load_stack_results(stack_dir=path, mode=mode, root=root)
    return load_stack_results(path, mode=mode, root=root)


def _profile_errors(results: StackResults, color: str, size: int) -> np.ndarray:
    for key in (f"{color}_jackknife_err", f"{color}_err", f"{color}_analytic_err"):
        if key not in results.arrays:
            continue
        error = np.asarray(results.arrays[key], dtype=float)
        if error.shape == (size,):
            return error
    return np.full(size, np.nan)


def _fit_bounds(config: Mapping[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    config = config or {}
    amp_bounds = config.get("amplitude_bounds", [-np.inf, np.inf])
    alpha_bounds = config.get("alpha_bounds", [-10.0, 10.0])
    lower = np.array([float(amp_bounds[0]), float(alpha_bounds[0])], dtype=float)
    upper = np.array([float(amp_bounds[1]), float(alpha_bounds[1])], dtype=float)
    if np.any(lower >= upper):
        raise ValueError("color_power_law_fit parameter bounds must be increasing")
    return lower, upper


def _initial_parameters(
    radius: np.ndarray,
    signal: np.ndarray,
    pivot_kpc: float,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    nearest = int(np.argmin(np.abs(radius - pivot_kpc)))
    amplitude = float(signal[nearest])
    if not np.isfinite(amplitude) or amplitude == 0:
        amplitude = float(np.nanmedian(signal))
    if not np.isfinite(amplitude) or amplitude == 0:
        amplitude = 1.0e-6

    lower, upper = bounds
    amplitude = float(np.clip(amplitude, lower[0], upper[0]))
    alpha = float(np.clip(-1.0, lower[1], upper[1]))
    return np.array([amplitude, alpha], dtype=float)


def _model(
    radius: np.ndarray,
    amplitude: float,
    alpha: float,
    pivot_kpc: float,
) -> np.ndarray:
    return amplitude * (radius / pivot_kpc) ** alpha


def _parameter_covariance(jacobian: np.ndarray) -> np.ndarray:
    fisher = jacobian.T @ jacobian
    return np.linalg.pinv(fisher)


__all__ = [
    "ColorPowerLawFit",
    "fit_color_power_law",
    "format_color_power_law_fits",
    "save_stack_color_power_law_fits",
]
