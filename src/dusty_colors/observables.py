"""Observable construction for canonical photometry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

ObservableMode = Literal["fcolors", "mcolors"]


def parse_color(color: str | Sequence[str]) -> tuple[str, str]:
    """Parse a color specification such as ``g-r``."""
    if isinstance(color, str):
        parts = color.split("-")
    else:
        parts = list(color)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Color must contain exactly two bands: {color}")
    return str(parts[0]), str(parts[1])


def flux_ratio_observable(
    catalog: pd.DataFrame,
    color: str | Sequence[str],
    snr_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flux-ratio values and propagated errors for a color."""
    band1, band2 = parse_color(color)
    if _has_flux(catalog, band1) and _has_flux(catalog, band2):
        f1 = catalog[f"flux_{band1}"].to_numpy(float)
        f2 = catalog[f"flux_{band2}"].to_numpy(float)
        e1 = _with_flux_error_floor(f1, catalog[f"fluxerr_{band1}"], snr_max)
        e2 = _with_flux_error_floor(f2, catalog[f"fluxerr_{band2}"], snr_max)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = f1 / f2
            rel = np.sqrt((e1 / f1) ** 2 + (e2 / f2) ** 2)
            error = np.abs(value) * rel
        invalid = (f1 <= 0) | (f2 <= 0) | (e1 <= 0) | (e2 <= 0)
        return _nan_invalid(value, error, invalid)

    value, color_err = magnitude_color_observable(catalog, color, snr_max=snr_max)
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = 10 ** (-0.4 * value)
        ratio_err = np.abs(ratio) * 0.4 * np.log(10) * color_err
    return ratio, ratio_err


def magnitude_color_observable(
    catalog: pd.DataFrame,
    color: str | Sequence[str],
    snr_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return magnitude-color values and propagated errors for a color."""
    band1, band2 = parse_color(color)
    if _has_mag(catalog, band1) and _has_mag(catalog, band2):
        m1 = catalog[f"mag_{band1}"].to_numpy(float)
        m2 = catalog[f"mag_{band2}"].to_numpy(float)
        e1 = _with_mag_error_floor(catalog[f"magerr_{band1}"], snr_max)
        e2 = _with_mag_error_floor(catalog[f"magerr_{band2}"], snr_max)
        value = m1 - m2
        error = np.sqrt(e1**2 + e2**2)
        invalid = (e1 <= 0) | (e2 <= 0)
        return _nan_invalid(value, error, invalid)

    if not (_has_flux(catalog, band1) and _has_flux(catalog, band2)):
        raise KeyError(
            "Catalog must contain either flux or magnitude pairs for "
            f"both bands in {band1}-{band2}"
        )

    f1 = catalog[f"flux_{band1}"].to_numpy(float)
    f2 = catalog[f"flux_{band2}"].to_numpy(float)
    e1 = _with_flux_error_floor(f1, catalog[f"fluxerr_{band1}"], snr_max)
    e2 = _with_flux_error_floor(f2, catalog[f"fluxerr_{band2}"], snr_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = -2.5 * np.log10(f1 / f2)
        rel = np.sqrt((e1 / f1) ** 2 + (e2 / f2) ** 2)
        error = 2.5 / np.log(10) * rel
    invalid = (f1 <= 0) | (f2 <= 0) | (e1 <= 0) | (e2 <= 0)
    return _nan_invalid(value, error, invalid)


def build_observable(
    catalog: pd.DataFrame,
    color: str | Sequence[str],
    mode: ObservableMode,
    snr_max: float | None = None,
) -> pd.DataFrame:
    """Build a value/error table for a TreeCorr color mode."""
    if mode == "fcolors":
        value, error = flux_ratio_observable(catalog, color, snr_max=snr_max)
    elif mode == "mcolors":
        value, error = magnitude_color_observable(catalog, color, snr_max=snr_max)
    else:
        raise ValueError(f"Unsupported observable mode: {mode}")
    return pd.DataFrame({"value": value, "error": error}, index=catalog.index)


def observable_column_names(
    color: str | Sequence[str],
    mode: ObservableMode,
) -> tuple[str, str]:
    """Return derived column names for a color observable."""
    band1, band2 = parse_color(color)
    label = f"{band1}_{band2}"
    if mode == "fcolors":
        return f"fcolor_{label}", f"fcolorerr_{label}"
    if mode == "mcolors":
        return f"mcolor_{label}", f"mcolorerr_{label}"
    raise ValueError(f"Unsupported observable mode: {mode}")


def add_observable_columns(
    catalog: pd.DataFrame,
    color: str | Sequence[str],
    mode: ObservableMode,
    snr_max: float | None = None,
    overwrite: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """Add derived observable columns without changing raw photometry."""
    out = catalog.copy() if copy else catalog
    value_col, err_col = observable_column_names(color, mode)
    if not overwrite and (value_col in out or err_col in out):
        raise ValueError(f"Observable columns already exist for {color} {mode}")
    observable = build_observable(out, color, mode, snr_max=snr_max)
    out[value_col] = observable["value"].to_numpy()
    out[err_col] = observable["error"].to_numpy()
    return out


def construct_observables(
    catalog: pd.DataFrame,
    colors: Sequence[str | Sequence[str]],
    modes: Sequence[ObservableMode] = ("fcolors", "mcolors"),
    snr_max: float | None = None,
    overwrite: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    """Add several derived color observables to a catalog."""
    out = catalog.copy() if copy else catalog
    for mode in modes:
        for color in colors:
            out = add_observable_columns(
                out,
                color,
                mode,
                snr_max=snr_max,
                overwrite=overwrite,
                copy=False,
            )
    return out


def _has_flux(catalog: pd.DataFrame, band: str) -> bool:
    return f"flux_{band}" in catalog and f"fluxerr_{band}" in catalog


def _has_mag(catalog: pd.DataFrame, band: str) -> bool:
    return f"mag_{band}" in catalog and f"magerr_{band}" in catalog


def _with_flux_error_floor(
    flux: np.ndarray,
    error: pd.Series | np.ndarray,
    snr_max: float | None,
) -> np.ndarray:
    err = np.asarray(error, dtype=float)
    if snr_max is None or snr_max <= 0:
        return err
    floor = np.abs(np.asarray(flux, dtype=float)) / float(snr_max)
    return np.maximum(err, floor)


def _with_mag_error_floor(
    error: pd.Series | np.ndarray,
    snr_max: float | None,
) -> np.ndarray:
    err = np.asarray(error, dtype=float)
    if snr_max is None or snr_max <= 0:
        return err
    floor = 2.5 / np.log(10) / float(snr_max)
    return np.maximum(err, floor)


def _nan_invalid(
    value: np.ndarray,
    error: np.ndarray,
    invalid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    invalid = invalid | ~np.isfinite(value) | ~np.isfinite(error)
    value = np.asarray(value, dtype=float).copy()
    error = np.asarray(error, dtype=float).copy()
    value[invalid] = np.nan
    error[invalid] = np.nan
    return value, error
