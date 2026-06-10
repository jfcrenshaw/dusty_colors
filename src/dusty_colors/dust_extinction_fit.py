"""Fit radial dust-extinction laws to stacked color excess profiles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_distribution

from .observables import parse_color
from .plotting import StackResults, load_stack_results

DEFAULT_FIT_COLORS = ("g-r", "r-i", "i-z")
DEFAULT_FILTER_WAVELENGTHS_UM = {
    "u": 0.3671,
    "g": 0.4827,
    "r": 0.6223,
    "i": 0.7546,
    "z": 0.8691,
    "y": 0.9712,
}


@dataclass(frozen=True)
class DustExtinctionFitConfig:
    """Configuration for a stack-level dust-extinction fit."""

    enabled: bool = True
    colors: tuple[str, ...] = DEFAULT_FIT_COLORS
    law: str = "F99"
    radial_pivot_kpc: float = 100.0
    foreground_redshift: float | None = None
    wavelengths_um: Mapping[str, float] | None = None
    amplitude_bounds: tuple[float, float] = (0.0, np.inf)
    alpha_bounds: tuple[float, float] = (-5.0, 1.0)
    rv_bounds: tuple[float, float] = (2.0, 6.0)
    fixed_rv: float | None = None
    initial_amplitude: float | None = None
    initial_alpha: float = -0.8
    initial_rv: float = 3.1

    @property
    def filter_wavelengths_um(self) -> dict[str, float]:
        wavelengths = dict(DEFAULT_FILTER_WAVELENGTHS_UM)
        if self.wavelengths_um is not None:
            wavelengths.update(
                {str(k): float(v) for k, v in self.wavelengths_um.items()}
            )
        return wavelengths


@dataclass(frozen=True)
class StackFitData:
    """Flattened stack data vector and covariance."""

    colors: tuple[str, ...]
    row_colors: tuple[str, ...]
    radius_kpc: np.ndarray
    signal_mag: np.ndarray
    covariance: np.ndarray
    covariance_method: str


@dataclass(frozen=True)
class DustExtinctionFitResult:
    """Best-fit parameters and metadata for one stack mode."""

    mode: str
    colors: tuple[str, ...]
    law: str
    foreground_redshift: float
    foreground_redshift_source: str
    radial_pivot_kpc: float
    wavelengths_um: Mapping[str, float]
    amplitude_av_mag: float
    alpha: float
    rv: float
    rv_fixed: bool
    parameter_covariance: np.ndarray
    parameter_errors: np.ndarray
    chi2: float
    dof: int
    p_value: float
    covariance_rank: int
    covariance_method: str
    optimizer_success: bool
    optimizer_message: str
    data: StackFitData
    model_mag: np.ndarray

    @property
    def reduced_chi2(self) -> float:
        if self.dof <= 0:
            return np.nan
        return self.chi2 / self.dof


def parse_dust_extinction_fit_config(
    stack_config: Mapping[str, Any] | None,
) -> DustExtinctionFitConfig:
    """Parse ``stack.dust_extinction_fit`` with conservative defaults."""

    stack_config = stack_config or {}
    raw = stack_config.get("dust_extinction_fit", {})
    if raw is False:
        return DustExtinctionFitConfig(enabled=False)
    if raw is True or raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("stack.dust_extinction_fit must be a mapping or boolean")

    if bool(raw.get("enabled", True)) is False:
        return DustExtinctionFitConfig(enabled=False)

    return DustExtinctionFitConfig(
        colors=_tuple(raw.get("colors", DEFAULT_FIT_COLORS)),
        law=str(raw.get("law", "F99")),
        radial_pivot_kpc=float(raw.get("radial_pivot_kpc", 100.0)),
        foreground_redshift=_optional_float(raw.get("foreground_redshift")),
        wavelengths_um=raw.get("wavelengths_um"),
        amplitude_bounds=_bounds(raw.get("amplitude_bounds"), (0.0, np.inf)),
        alpha_bounds=_bounds(raw.get("alpha_bounds"), (-5.0, 1.0)),
        rv_bounds=_bounds(raw.get("rv_bounds"), (2.0, 6.0)),
        fixed_rv=_optional_float(raw.get("fixed_rv")),
        initial_amplitude=_optional_float(raw.get("initial_amplitude")),
        initial_alpha=float(raw.get("initial_alpha", -0.8)),
        initial_rv=float(raw.get("initial_rv", 3.1)),
    )


def save_stack_dust_extinction_fit(
    source: StackResults | str | Path,
    output_dir: str | Path | None = None,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
    stack_config: Mapping[str, Any] | None = None,
    foreground_redshift: float | None = None,
    foreground_redshift_source: str = "provided",
) -> Path | None:
    """Fit and write one dust-extinction report.

    Returns ``None`` when the selected stack mode does not contain all required
    colors. Other fit failures are raised so the post-run orchestrator can
    decide whether to warn or fail.
    """

    config = parse_dust_extinction_fit_config(stack_config)
    if not config.enabled:
        return None

    results = (
        source
        if isinstance(source, StackResults)
        else _load_stack_source(source, mode=mode, root=root)
    )
    if not _has_required_colors(results, config.colors):
        return None

    redshift = (
        config.foreground_redshift
        if config.foreground_redshift is not None
        else foreground_redshift
    )
    if redshift is None:
        redshift = 0.0
        foreground_redshift_source = "default_zero"
    elif config.foreground_redshift is not None:
        foreground_redshift_source = "stack_config"

    fit = fit_dust_extinction_law(
        results,
        config=config,
        foreground_redshift=float(redshift),
        foreground_redshift_source=foreground_redshift_source,
    )
    out_dir = results.stack_dir if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"dust_extinction_fit_{results.mode}.txt"
    path.write_text(format_dust_extinction_fit(fit), encoding="utf-8")
    return path


def fit_dust_extinction_law(
    results: StackResults,
    *,
    config: DustExtinctionFitConfig | None = None,
    foreground_redshift: float = 0.0,
    foreground_redshift_source: str = "provided",
) -> DustExtinctionFitResult:
    """Fit ``A_V(r) = amplitude * (r / pivot)^alpha`` and ``R_V``."""

    fit_config = config or DustExtinctionFitConfig()
    if fit_config.radial_pivot_kpc <= 0:
        raise ValueError("Dust-extinction radial_pivot_kpc must be positive")
    if foreground_redshift < 0:
        raise ValueError("Dust-extinction foreground_redshift must be non-negative")

    data = stack_fit_data(results, fit_config.colors)
    lower, upper = _fit_bounds(fit_config)
    if np.any(lower >= upper):
        raise ValueError("Dust-extinction parameter bounds must be increasing")
    if fit_config.fixed_rv is not None and fit_config.fixed_rv <= 0:
        raise ValueError("Dust-extinction fixed_rv must be positive")

    whitener, covariance_rank = _covariance_whitener(data.covariance)
    n_parameters = len(lower)
    if covariance_rank < n_parameters:
        diagonal = np.diag(np.clip(np.diag(data.covariance), 0.0, np.inf))
        whitener, covariance_rank = _covariance_whitener(_regularize_diagonal(diagonal))

    p0 = _initial_parameters(data, fit_config, foreground_redshift, lower, upper)

    def residual(params: np.ndarray) -> np.ndarray:
        amplitude, alpha, rv = _fit_parameter_values(params, fit_config)
        try:
            model = dust_color_excess_model(
                data.row_colors,
                data.radius_kpc,
                amplitude_av_mag=amplitude,
                alpha=alpha,
                rv=rv,
                radial_pivot_kpc=fit_config.radial_pivot_kpc,
                wavelengths_um=fit_config.filter_wavelengths_um,
                foreground_redshift=foreground_redshift,
                law=fit_config.law,
            )
        except (ValueError, TypeError):
            return np.full(covariance_rank, 1.0e30)
        return whitener(data.signal_mag - model)

    optimized = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        xtol=1.0e-10,
        ftol=1.0e-10,
        gtol=1.0e-10,
        max_nfev=2000,
    )

    parameter_covariance = _parameter_covariance(optimized.jac)
    parameter_errors = np.sqrt(np.clip(np.diag(parameter_covariance), 0.0, np.inf))
    amplitude, alpha, rv = _fit_parameter_values(optimized.x, fit_config)
    model = dust_color_excess_model(
        data.row_colors,
        data.radius_kpc,
        amplitude_av_mag=amplitude,
        alpha=alpha,
        rv=rv,
        radial_pivot_kpc=fit_config.radial_pivot_kpc,
        wavelengths_um=fit_config.filter_wavelengths_um,
        foreground_redshift=foreground_redshift,
        law=fit_config.law,
    )
    chi2 = float(np.sum(residual(optimized.x) ** 2))
    dof = max(int(covariance_rank) - n_parameters, 0)
    p_value = float(chi2_distribution.sf(chi2, dof)) if dof > 0 else np.nan

    return DustExtinctionFitResult(
        mode=results.mode,
        colors=fit_config.colors,
        law=fit_config.law,
        foreground_redshift=float(foreground_redshift),
        foreground_redshift_source=foreground_redshift_source,
        radial_pivot_kpc=fit_config.radial_pivot_kpc,
        wavelengths_um=fit_config.filter_wavelengths_um,
        amplitude_av_mag=float(amplitude),
        alpha=float(alpha),
        rv=float(rv),
        rv_fixed=fit_config.fixed_rv is not None,
        parameter_covariance=parameter_covariance,
        parameter_errors=parameter_errors,
        chi2=chi2,
        dof=dof,
        p_value=p_value,
        covariance_rank=covariance_rank,
        covariance_method=data.covariance_method,
        optimizer_success=bool(optimized.success),
        optimizer_message=str(optimized.message),
        data=data,
        model_mag=model,
    )


def stack_fit_data(
    results: StackResults,
    colors: Sequence[str] = DEFAULT_FIT_COLORS,
) -> StackFitData:
    """Build the flattened color/radius vector used by the fit."""

    selected_colors = tuple(str(color) for color in colors)
    if not _has_required_colors(results, selected_colors):
        missing = [
            color
            for color in selected_colors
            if f"{color}_bin_centers" not in results.arrays
            or f"{color}_avg" not in results.arrays
        ]
        raise KeyError(f"Stack results are missing dust-fit colors: {missing}")

    row_colors: list[str] = []
    radii: list[np.ndarray] = []
    signals: list[np.ndarray] = []
    sample_blocks: list[np.ndarray] = []
    covariance_blocks: list[np.ndarray] = []
    error_blocks: list[np.ndarray] = []

    for color in selected_colors:
        radius = np.asarray(results.require(f"{color}_bin_centers"), dtype=float)
        signal = np.asarray(results.require(f"{color}_avg"), dtype=float)
        if radius.shape != signal.shape:
            raise ValueError(f"{color} radius and signal arrays have different shapes")
        if np.any(radius <= 0):
            raise ValueError(f"{color} contains non-positive radial bins")

        row_colors.extend([color] * len(radius))
        radii.append(radius)
        signals.append(signal)

        samples_key = f"{color}_jackknife_samples"
        if samples_key in results.arrays:
            samples = np.asarray(results.arrays[samples_key], dtype=float)
            if samples.ndim == 2 and samples.shape[1] == len(radius):
                sample_blocks.append(samples)

        cov_key = f"{color}_cov"
        if cov_key in results.arrays:
            cov = np.asarray(results.arrays[cov_key], dtype=float)
            if cov.shape == (len(radius), len(radius)):
                covariance_blocks.append(cov)

        err = _profile_errors(results, color, len(radius))
        error_blocks.append(np.diag(err**2))

    radius_vector = np.concatenate(radii)
    signal_vector = np.concatenate(signals)
    finite = np.isfinite(radius_vector) & np.isfinite(signal_vector)

    covariance_method = "diagonal_errors"
    if len(sample_blocks) == len(selected_colors) and _same_rows(sample_blocks):
        samples = np.column_stack(sample_blocks)
        finite &= np.all(np.isfinite(samples), axis=0)
        covariance = _jackknife_covariance(samples)
        covariance_method = "full_jackknife"
    elif len(covariance_blocks) == len(selected_colors):
        covariance = _block_diag(covariance_blocks)
        covariance_method = "per_color_covariance"
    else:
        covariance = _block_diag(error_blocks)

    finite &= np.isfinite(np.diag(covariance)) & (np.diag(covariance) > 0)
    if np.count_nonzero(finite) < 4:
        raise ValueError("Dust-extinction fit needs at least four finite data points")

    covariance = covariance[np.ix_(finite, finite)]
    row_color_vector = tuple(
        color for color, keep in zip(row_colors, finite, strict=True) if keep
    )
    return StackFitData(
        colors=selected_colors,
        row_colors=row_color_vector,
        radius_kpc=radius_vector[finite],
        signal_mag=signal_vector[finite],
        covariance=_regularize_covariance(covariance),
        covariance_method=covariance_method,
    )


def dust_color_excess_model(
    colors: Sequence[str],
    radius_kpc: Sequence[float] | np.ndarray,
    *,
    amplitude_av_mag: float,
    alpha: float,
    rv: float,
    radial_pivot_kpc: float = 100.0,
    wavelengths_um: Mapping[str, float] | None = None,
    foreground_redshift: float = 0.0,
    law: str = "F99",
) -> np.ndarray:
    """Return model ``E(color)`` in magnitudes for each row color/radius."""

    radius = np.asarray(radius_kpc, dtype=float)
    if np.any(radius <= 0):
        raise ValueError("Dust-extinction model radii must be positive")
    if radial_pivot_kpc <= 0:
        raise ValueError("Dust-extinction radial pivot must be positive")

    color_coefficients = [
        color_excess_per_av(
            color,
            rv=rv,
            wavelengths_um=wavelengths_um,
            foreground_redshift=foreground_redshift,
            law=law,
        )
        for color in colors
    ]
    radial = amplitude_av_mag * (radius / radial_pivot_kpc) ** alpha
    return radial * np.asarray(color_coefficients, dtype=float)


def color_excess_per_av(
    color: str | Sequence[str],
    *,
    rv: float = 3.1,
    wavelengths_um: Mapping[str, float] | None = None,
    foreground_redshift: float = 0.0,
    law: str = "F99",
) -> float:
    """Return ``E(X-Y) / A_V`` for a Rubin-filter color."""

    wavelengths = dict(DEFAULT_FILTER_WAVELENGTHS_UM)
    if wavelengths_um is not None:
        wavelengths.update({str(k): float(v) for k, v in wavelengths_um.items()})
    ratios = band_extinction_ratios(
        wavelengths,
        rv=rv,
        foreground_redshift=foreground_redshift,
        law=law,
    )
    band1, band2 = parse_color(color)
    try:
        return float(ratios[band1] - ratios[band2])
    except KeyError as exc:
        known = ", ".join(sorted(ratios))
        raise ValueError(
            f"Unknown filter in color {band1}-{band2}; known filters: {known}"
        ) from exc


def color_excess_to_av(
    color_excess_mag: float | Sequence[float] | np.ndarray,
    color: str | Sequence[str],
    *,
    rv: float = 3.1,
    wavelengths_um: Mapping[str, float] | None = None,
    foreground_redshift: float = 0.0,
    law: str = "F99",
) -> float | np.ndarray:
    """Convert ``E(X-Y)`` color excess in magnitudes to ``A_V``."""

    coefficient = color_excess_per_av(
        color,
        rv=rv,
        wavelengths_um=wavelengths_um,
        foreground_redshift=foreground_redshift,
        law=law,
    )
    if coefficient == 0.0:
        band1, band2 = parse_color(color)
        raise ValueError(
            f"Color {band1}-{band2} has zero extinction contrast; "
            "cannot convert E(color) to A_V"
        )

    values = np.asarray(color_excess_mag, dtype=float)
    av = values / coefficient
    if values.ndim == 0:
        return float(av)
    return av


def band_extinction_ratios(
    wavelengths_um: Mapping[str, float],
    *,
    rv: float,
    foreground_redshift: float = 0.0,
    law: str = "F99",
) -> dict[str, float]:
    """Evaluate ``A_lambda / A_V`` at dust-rest-frame band wavelengths."""

    law_model = _dust_law(law, rv)
    bands = tuple(str(band) for band in wavelengths_um)
    wavelengths = np.array([float(wavelengths_um[band]) for band in bands], dtype=float)
    if np.any(wavelengths <= 0):
        raise ValueError("Filter effective wavelengths must be positive")

    rest_wavelengths = wavelengths / (1.0 + float(foreground_redshift))
    x_inverse_micron = 1.0 / rest_wavelengths
    values = np.asarray(
        law_model(_inverse_micron_quantity(x_inverse_micron)),
        dtype=float,
    )
    if values.shape != x_inverse_micron.shape:
        values = np.reshape(values, x_inverse_micron.shape)
    return dict(zip(bands, values, strict=True))


def format_dust_extinction_fit(fit: DustExtinctionFitResult) -> str:
    """Format a fit result as a compact text report."""

    errors = fit.parameter_errors
    lines = [
        "Dust extinction law fit",
        f"created_at_utc: {datetime.now(timezone.utc).isoformat()}",
        f"mode: {fit.mode}",
        f"colors: {', '.join(fit.colors)}",
        "",
        "model:",
        "  E(b1-b2, r) = A_V(pivot) * (r / pivot)^alpha "
        "* [(A_b1/A_V) - (A_b2/A_V)]",
        f"  law: {fit.law}",
        f"  radial_pivot_kpc: {fit.radial_pivot_kpc:.10g}",
        f"  foreground_redshift: {fit.foreground_redshift:.10g}",
        f"  foreground_redshift_source: {fit.foreground_redshift_source}",
        "  wavelength_note: observed effective wavelengths are divided by "
        "1 + foreground_redshift before evaluating the dust law",
        "",
        "filter_wavelengths_um_observed:",
    ]
    for band, wavelength in fit.wavelengths_um.items():
        lines.append(f"  {band}: {wavelength:.10g}")

    lines.extend(
        [
            "",
            "parameters:",
            f"  amplitude_Av_at_pivot_mag: {fit.amplitude_av_mag:.10g} "
            f"+/- {errors[0]:.3g}",
            f"  radial_power_law_index_alpha: {fit.alpha:.10g} "
            f"+/- {errors[1]:.3g}",
            _rv_parameter_line(fit),
            "",
            "goodness_of_fit:",
            f"  chi2: {fit.chi2:.10g}",
            f"  dof: {fit.dof}",
            f"  reduced_chi2: {fit.reduced_chi2:.10g}",
            f"  p_value: {fit.p_value:.10g}",
            f"  n_data: {len(fit.data.signal_mag)}",
            f"  n_free_parameters: {len(fit.parameter_errors)}",
            f"  covariance_rank: {fit.covariance_rank}",
            f"  covariance_method: {fit.covariance_method}",
            f"  optimizer_success: {fit.optimizer_success}",
            f"  optimizer_message: {fit.optimizer_message}",
            "",
            "parameter_covariance:",
        ]
    )
    for row in fit.parameter_covariance:
        lines.append("  " + " ".join(f"{value:.10e}" for value in row))

    sigma = np.sqrt(np.clip(np.diag(fit.data.covariance), 0.0, np.inf))
    residual = fit.data.signal_mag - fit.model_mag
    lines.extend(
        [
            "",
            "data:",
            "  color radius_kpc signal_mag sigma_mag model_mag residual_mag",
        ]
    )
    for color, radius, signal, err, model, resid in zip(
        fit.data.row_colors,
        fit.data.radius_kpc,
        fit.data.signal_mag,
        sigma,
        fit.model_mag,
        residual,
        strict=True,
    ):
        lines.append(
            f"  {color} {radius:.10g} {signal:.10e} {err:.10e} "
            f"{model:.10e} {resid:.10e}"
        )
    return "\n".join(lines) + "\n"


def _has_required_colors(results: StackResults, colors: Sequence[str]) -> bool:
    return all(
        f"{color}_bin_centers" in results.arrays and f"{color}_avg" in results.arrays
        for color in colors
    )


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


def _inverse_micron_quantity(values: np.ndarray) -> Any:
    try:
        import astropy.units as u
    except ModuleNotFoundError:
        return values
    return values / u.micron


def _rv_parameter_line(fit: DustExtinctionFitResult) -> str:
    if fit.rv_fixed:
        return f"  R_V: {fit.rv:.10g} (fixed)"
    return f"  R_V: {fit.rv:.10g} +/- {fit.parameter_errors[2]:.3g}"


def _tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _optional_float(value: Any) -> float | None:
    if value is None or value == "auto":
        return None
    return float(value)


def _bounds(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        return default
    lo, hi = value
    return _bound_value(lo), _bound_value(hi)


def _bound_value(value: Any) -> float:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return np.inf
        if lowered in {"-inf", "-infinity"}:
            return -np.inf
    return float(value)


def _profile_errors(results: StackResults, color: str, size: int) -> np.ndarray:
    for key in (f"{color}_jackknife_err", f"{color}_err", f"{color}_analytic_err"):
        if key not in results.arrays:
            continue
        err = np.asarray(results.arrays[key], dtype=float)
        if err.shape == (size,):
            return np.where(np.isfinite(err) & (err > 0), err, np.nan)
    return np.full(size, np.nan)


def _same_rows(blocks: Sequence[np.ndarray]) -> bool:
    rows = {block.shape[0] for block in blocks}
    return len(rows) == 1 and next(iter(rows)) >= 2


def _jackknife_covariance(samples: np.ndarray) -> np.ndarray:
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    covariance = (1.0 - 1.0 / samples.shape[0]) * centered.T @ centered
    return np.real_if_close(covariance)


def _block_diag(blocks: Sequence[np.ndarray]) -> np.ndarray:
    size = sum(block.shape[0] for block in blocks)
    out = np.zeros((size, size), dtype=float)
    start = 0
    for block in blocks:
        block = np.asarray(block, dtype=float)
        stop = start + block.shape[0]
        out[start:stop, start:stop] = block
        start = stop
    return out


def _regularize_covariance(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    cov = 0.5 * (cov + cov.T)
    diagonal = np.diag(cov)
    positive = diagonal[np.isfinite(diagonal) & (diagonal > 0)]
    if len(positive) == 0:
        raise ValueError("Dust-extinction covariance has no positive diagonal terms")
    floor = max(float(np.median(positive)) * 1.0e-10, 1.0e-16)
    cov = cov.copy()
    cov[np.diag_indices_from(cov)] += floor
    return cov


def _regularize_diagonal(covariance: np.ndarray) -> np.ndarray:
    diagonal = np.diag(np.asarray(covariance, dtype=float))
    positive = diagonal[np.isfinite(diagonal) & (diagonal > 0)]
    if len(positive) == 0:
        positive = np.array([1.0])
    floor = max(float(np.median(positive)) * 1.0e-10, 1.0e-16)
    return np.diag(np.where(np.isfinite(diagonal) & (diagonal > 0), diagonal, floor))


def _covariance_whitener(covariance: np.ndarray) -> tuple[Any, int]:
    cov = _regularize_covariance(covariance)
    values, vectors = np.linalg.eigh(cov)
    scale = float(np.max(values)) if len(values) else 0.0
    threshold = max(scale * 1.0e-10, 1.0e-16)
    keep = values > threshold
    if not np.any(keep):
        raise ValueError("Dust-extinction covariance is singular")
    transform = vectors[:, keep].T / np.sqrt(values[keep])[:, np.newaxis]

    def whiten(residual: np.ndarray) -> np.ndarray:
        return transform @ residual

    return whiten, int(np.count_nonzero(keep))


def _parameter_covariance(jacobian: np.ndarray) -> np.ndarray:
    fisher = jacobian.T @ jacobian
    return np.linalg.pinv(fisher)


def _fit_bounds(config: DustExtinctionFitConfig) -> tuple[np.ndarray, np.ndarray]:
    lower = [config.amplitude_bounds[0], config.alpha_bounds[0]]
    upper = [config.amplitude_bounds[1], config.alpha_bounds[1]]
    if config.fixed_rv is None:
        lower.append(config.rv_bounds[0])
        upper.append(config.rv_bounds[1])
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _fit_parameter_values(
    params: np.ndarray,
    config: DustExtinctionFitConfig,
) -> tuple[float, float, float]:
    amplitude = float(params[0])
    alpha = float(params[1])
    if config.fixed_rv is not None:
        return amplitude, alpha, float(config.fixed_rv)
    return amplitude, alpha, float(params[2])


def _initial_parameters(
    data: StackFitData,
    config: DustExtinctionFitConfig,
    foreground_redshift: float,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    alpha0 = float(np.clip(config.initial_alpha, lower[1], upper[1]))
    if config.fixed_rv is None:
        rv0 = float(np.clip(config.initial_rv, lower[2], upper[2]))
    else:
        rv0 = float(config.fixed_rv)
    if config.initial_amplitude is None:
        unit_model = dust_color_excess_model(
            data.row_colors,
            data.radius_kpc,
            amplitude_av_mag=1.0,
            alpha=alpha0,
            rv=rv0,
            radial_pivot_kpc=config.radial_pivot_kpc,
            wavelengths_um=config.filter_wavelengths_um,
            foreground_redshift=foreground_redshift,
            law=config.law,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            estimates = data.signal_mag / unit_model
        good = np.isfinite(estimates) & (estimates > 0)
        amplitude0 = float(np.median(estimates[good])) if np.any(good) else 0.01
    else:
        amplitude0 = config.initial_amplitude

    amplitude0 = _clip_initial(amplitude0, lower[0], upper[0])
    if config.fixed_rv is not None:
        return np.array([amplitude0, alpha0], dtype=float)
    return np.array([amplitude0, alpha0, rv0], dtype=float)


def _clip_initial(value: float, lower: float, upper: float) -> float:
    if np.isinf(upper):
        upper_for_clip = max(value, lower + 1.0)
    else:
        upper_for_clip = upper
    clipped = float(np.clip(value, lower, upper_for_clip))
    if clipped == lower and lower == 0:
        return min(1.0e-4, upper_for_clip)
    return clipped


def _dust_law(name: str, rv: float) -> Any:
    module = import_module("dust_extinction.parameter_averages")
    law_class = getattr(module, name, None)
    if law_class is None:
        raise ValueError(f"Unknown dust_extinction.parameter_averages law: {name}")
    return law_class(Rv=float(rv))


__all__ = [
    "DEFAULT_FILTER_WAVELENGTHS_UM",
    "DEFAULT_FIT_COLORS",
    "DustExtinctionFitConfig",
    "DustExtinctionFitResult",
    "StackFitData",
    "band_extinction_ratios",
    "color_excess_per_av",
    "color_excess_to_av",
    "dust_color_excess_model",
    "fit_dust_extinction_law",
    "format_dust_extinction_fit",
    "parse_dust_extinction_fit_config",
    "save_stack_dust_extinction_fit",
    "stack_fit_data",
]
