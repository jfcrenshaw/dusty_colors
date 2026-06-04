"""Explicit optional cleaning helpers for prepared samples."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


def apply_minimal_cleaning(
    catalog: pd.DataFrame,
    *,
    bands: list[str] | None = None,
    photometry: str | None = None,
) -> pd.DataFrame:
    """Filter invalid canonical rows without changing raw photometry columns."""
    cleaned = catalog.copy()
    required = ["ra", "dec", "z_phot", "is_galaxy", "mask_ok", "quality_ok"]
    mask = np.ones(len(cleaned), dtype=bool)
    for col in required:
        if col not in cleaned:
            raise ValueError(f"Catalog missing required cleaning column: {col}")
    mask &= np.isfinite(cleaned["ra"].to_numpy(float))
    mask &= np.isfinite(cleaned["dec"].to_numpy(float))
    mask &= np.isfinite(cleaned["z_phot"].to_numpy(float))
    mask &= cleaned["is_galaxy"].astype(bool).to_numpy()
    mask &= cleaned["mask_ok"].astype(bool).to_numpy()
    mask &= cleaned["quality_ok"].astype(bool).to_numpy()

    for band in bands or []:
        if photometry == "flux":
            cols = [f"flux_{band}", f"fluxerr_{band}"]
        elif photometry == "mag":
            cols = [f"mag_{band}", f"magerr_{band}"]
        else:
            cols = [
                col
                for col in (
                    f"flux_{band}",
                    f"fluxerr_{band}",
                    f"mag_{band}",
                    f"magerr_{band}",
                )
                if col in cleaned
            ]
        for col in cols:
            if col not in cleaned:
                raise ValueError(f"Catalog missing requested photometry column: {col}")
            mask &= np.isfinite(cleaned[col].to_numpy(float))
        err_cols = [col for col in cols if "err" in col]
        for col in err_cols:
            mask &= cleaned[col].to_numpy(float) > 0

    return cleaned.loc[mask].reset_index(drop=True)


def add_diagnostic_columns(
    catalog: pd.DataFrame,
    *,
    bands: list[str] | None = None,
) -> pd.DataFrame:
    """Add simple diagnostic columns without modifying raw photometry."""
    out = catalog.copy()
    for band in bands or []:
        flux = f"flux_{band}"
        err = f"fluxerr_{band}"
        if flux in out and err in out:
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"diagnostic_snr_{band}"] = out[flux] / out[err]
    return out


def clean_sample(catalog: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Apply optional sample cleaning without overwriting raw photometry."""
    cleaned = catalog.copy()
    if not config or not bool(config.get("enabled", False)):
        return cleaned

    finite_columns = list(config.get("finite_columns", []))
    if finite_columns:
        mask = np.ones(len(cleaned), dtype=bool)
        for col in finite_columns:
            if col not in cleaned:
                raise ValueError(f"Cleaning requested missing column: {col}")
            mask &= np.isfinite(cleaned[col].to_numpy(float))
        cleaned = cleaned.loc[mask].reset_index(drop=True)

    clip = config.get("robust_clip")
    if clip:
        cleaned = _apply_robust_clip(cleaned, _cleaning_mapping(clip, "robust_clip"))

    trend = config.get("redshift_trend", config.get("redshift_trend_removal"))
    if trend:
        cleaned = remove_redshift_trends(
            cleaned,
            _cleaning_mapping(trend, "redshift_trend"),
        )

    forest = config.get("isolation_forest", config.get("isolationforest"))
    if forest:
        cleaned = apply_isolation_forest(
            cleaned,
            _cleaning_mapping(forest, "isolation_forest"),
        )

    return cleaned


def remove_redshift_trends(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Add polynomial redshift-trend and trend-removed diagnostic columns."""
    if not bool(config.get("enabled", True)):
        return catalog.copy()

    columns = _required_columns(config, "redshift_trend")
    redshift_col = str(
        config.get("redshift_col", config.get("redshift_column", "z_phot"))
    )
    if redshift_col not in catalog:
        raise ValueError(f"Redshift trend requested missing column: {redshift_col}")

    degree = int(config.get("degree", 1))
    if degree < 0:
        raise ValueError("redshift_trend.degree must be non-negative")

    output_suffix = str(config.get("output_suffix", "_z_detrended"))
    trend_suffix = str(config.get("trend_suffix", "_z_trend"))
    out = catalog.copy()
    redshift = out[redshift_col].to_numpy(float)

    for col in columns:
        if col not in out:
            raise ValueError(f"Redshift trend requested missing column: {col}")
        trend_col = f"{col}{trend_suffix}"
        output_col = f"{col}{output_suffix}"
        _ensure_new_columns(out, [trend_col, output_col], source=col)

        values = out[col].to_numpy(float)
        trend = np.full(len(out), np.nan, dtype=float)
        detrended = np.full(len(out), np.nan, dtype=float)
        finite = np.isfinite(redshift) & np.isfinite(values)
        if np.any(finite):
            trend_fit = _polynomial_trend(
                redshift[finite],
                values[finite],
                degree=degree,
            )
            center = _trend_center(values[finite], config)
            trend[finite] = trend_fit
            detrended[finite] = values[finite] - trend_fit + center
        out[trend_col] = trend
        out[output_col] = detrended

    return out


def apply_isolation_forest(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Remove multivariate outliers using scikit-learn IsolationForest."""
    if not bool(config.get("enabled", True)):
        return catalog.copy()

    try:
        from sklearn.ensemble import IsolationForest
    except ImportError as exc:
        raise ImportError(
            "IsolationForest cleaning requires scikit-learn. Install the project "
            "dependencies or add scikit-learn to your environment."
        ) from exc

    columns = _required_columns(config, "isolation_forest")
    out = catalog.copy()
    for col in columns:
        if col not in out:
            raise ValueError(f"IsolationForest requested missing column: {col}")

    drop_nonfinite = bool(config.get("drop_nonfinite", True))
    features = out[columns].to_numpy(float)
    finite = np.isfinite(features).all(axis=1)
    if drop_nonfinite:
        out = out.loc[finite].reset_index(drop=True)
        features = out[columns].to_numpy(float)
        finite = np.ones(len(out), dtype=bool)

    min_samples = int(config.get("min_samples", 2))
    if min_samples < 1:
        raise ValueError("isolation_forest.min_samples must be positive")
    if np.sum(finite) < min_samples:
        return out

    scaled = _scaled_features(features[finite], config)
    model = IsolationForest(**_isolation_forest_kwargs(config))
    labels = model.fit_predict(scaled)
    scores = model.decision_function(scaled)

    score_col = config.get("score_col", "isolation_forest_score")
    label_col = config.get("label_col", "isolation_forest_label")
    if score_col:
        out[str(score_col)] = np.nan
        out.loc[finite, str(score_col)] = scores
    if label_col:
        out[str(label_col)] = 0
        out.loc[finite, str(label_col)] = labels

    keep = ~finite if not drop_nonfinite else np.zeros(len(out), dtype=bool)
    keep[finite] = labels == 1
    return out.loc[keep].reset_index(drop=True)


def _apply_robust_clip(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    cleaned = catalog.copy()
    columns = _required_columns(config, "robust_clip")
    sigma = float(config.get("sigma", 5.0))
    if sigma <= 0:
        raise ValueError("robust_clip.sigma must be positive")

    for col in columns:
        if col not in cleaned:
            raise ValueError(f"Robust clip requested missing column: {col}")
        values = cleaned[col].to_numpy(float)
        med = np.nanmedian(values)
        mad = np.nanmedian(np.abs(values - med))
        if not np.isfinite(mad) or mad == 0:
            continue
        keep = np.abs(values - med) <= sigma * 1.4826 * mad
        cleaned = cleaned.loc[keep].reset_index(drop=True)
    return cleaned


def _cleaning_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"cleaning.{name} must be a mapping")
    return value


def _required_columns(config: Mapping[str, Any], name: str) -> list[str]:
    columns = list(config.get("columns", []))
    if not columns:
        raise ValueError(f"cleaning.{name}.columns must not be empty")
    return [str(col) for col in columns]


def _ensure_new_columns(
    catalog: pd.DataFrame,
    columns: list[str],
    *,
    source: str,
) -> None:
    for col in columns:
        if col == source:
            raise ValueError(f"Cleaning would overwrite source column: {source}")
        if col in catalog:
            raise ValueError(f"Cleaning output column already exists: {col}")


def _polynomial_trend(
    redshift: np.ndarray,
    values: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    unique_redshifts = np.unique(redshift).size
    fit_degree = min(degree, len(values) - 1, unique_redshifts - 1)
    if fit_degree <= 0:
        return np.full(len(values), np.nanmedian(values))
    coeffs = np.polyfit(redshift, values, fit_degree)
    return np.polyval(coeffs, redshift)


def _trend_center(values: np.ndarray, config: Mapping[str, Any]) -> float:
    center = config.get("center", 0.0)
    if center in (None, False, "none"):
        return 0.0
    if center == "median":
        return float(np.nanmedian(values)) if len(values) else 0.0
    return float(center)


def _isolation_forest_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "n_estimators": int(config.get("n_estimators", 100)),
        "contamination": _validate_contamination(config.get("contamination", "auto")),
        "max_samples": config.get("max_samples", "auto"),
        "random_state": config.get("random_state", 42),
    }
    for key in ("max_features", "bootstrap", "n_jobs", "warm_start"):
        if key in config:
            kwargs[key] = config[key]
    return kwargs


def _validate_contamination(value: Any) -> float | str:
    if isinstance(value, str):
        if value != "auto":
            raise ValueError("IsolationForest contamination must be a float or 'auto'")
        return value
    contamination = float(value)
    if not 0.0 < contamination <= 0.5:
        raise ValueError("IsolationForest contamination must be in (0, 0.5]")
    return contamination


def _scaled_features(features: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    scale = config.get("scale", "robust")
    if scale in (False, None, "none"):
        return features
    if scale != "robust":
        raise ValueError("isolation_forest.scale must be 'robust', 'none', or false")

    center = np.nanmedian(features, axis=0)
    mad = np.nanmedian(np.abs(features - center), axis=0)
    sigma = 1.4826 * mad
    sigma[~np.isfinite(sigma) | (sigma == 0)] = 1.0
    return (features - center) / sigma
