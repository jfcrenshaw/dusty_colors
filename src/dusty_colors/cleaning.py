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


class CleaningPipeline:
    """Apply the optional cleaning steps for one prepared sample."""

    def __init__(self, catalog: pd.DataFrame, config: Mapping[str, Any]) -> None:
        self.catalog = catalog
        self.config = config

    def clean(self) -> pd.DataFrame:
        cleaned = self.catalog.copy()
        if not self.config or not bool(self.config.get("enabled", False)):
            return cleaned

        cleaned = self._drop_nonfinite_columns(cleaned)
        cleaned = self._apply_step(cleaned, "robust_clip", RobustClipCleaner)
        cleaned = self._apply_step(
            cleaned,
            "redshift_trend",
            PolynomialRedshiftTrendCleaner,
        )
        cleaned = self._apply_step(
            cleaned,
            "column_redshift_trend",
            ColumnRedshiftTrendCleaner,
        )
        cleaned = self._apply_step(
            cleaned,
            "isolation_forest",
            IsolationForestCleaner,
        )
        cleaned = self._apply_step(
            cleaned,
            "column_isolation_forest",
            ColumnIsolationForestCleaner,
        )
        return cleaned

    def _drop_nonfinite_columns(self, catalog: pd.DataFrame) -> pd.DataFrame:
        finite_columns = list(self.config.get("finite_columns", []))
        if not finite_columns:
            return catalog

        mask = np.ones(len(catalog), dtype=bool)
        for col in finite_columns:
            if col not in catalog:
                raise ValueError(f"Cleaning requested missing column: {col}")
            mask &= np.isfinite(catalog[col].to_numpy(float))
        return catalog.loc[mask].reset_index(drop=True)

    def _apply_step(
        self,
        catalog: pd.DataFrame,
        name: str,
        cleaner_cls: type["ConfiguredCleaner"],
    ) -> pd.DataFrame:
        step_config = self._step_config(name)
        if step_config is None:
            return catalog
        return cleaner_cls(step_config).apply(catalog)

    def _step_config(
        self,
        name: str,
    ) -> Mapping[str, Any] | None:
        value = self.config.get(name)
        if not value:
            return None
        if not isinstance(value, Mapping):
            raise ValueError(f"cleaning.{name} must be a mapping")
        return value


class ConfiguredCleaner:
    """Base class for one configured optional cleaning operation."""

    config_name = "cleaning"

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    def apply(self, catalog: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def enabled(self) -> bool:
        return bool(self.config.get("enabled", True))

    def required_columns(self) -> list[str]:
        columns = list(self.config.get("columns", []))
        if not columns:
            raise ValueError(f"cleaning.{self.config_name}.columns must not be empty")
        return [str(col) for col in columns]

    @staticmethod
    def ensure_new_columns(
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


class RobustClipCleaner(ConfiguredCleaner):
    """Remove univariate outliers with median absolute deviation clipping."""

    config_name = "robust_clip"

    def apply(self, catalog: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled():
            return catalog.copy()

        cleaned = catalog.copy()
        sigma = float(self.config.get("sigma", 5.0))
        if sigma <= 0:
            raise ValueError("robust_clip.sigma must be positive")

        for col in self.required_columns():
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


class PolynomialRedshiftTrendCleaner(ConfiguredCleaner):
    """Add polynomial redshift-trend and trend-removed diagnostic columns."""

    config_name = "redshift_trend"

    def apply(self, catalog: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled():
            return catalog.copy()

        redshift_col = str(self.config.get("redshift_col", "z_phot"))
        if redshift_col not in catalog:
            raise ValueError(f"Redshift trend requested missing column: {redshift_col}")

        degree = int(self.config.get("degree", 1))
        if degree < 0:
            raise ValueError("redshift_trend.degree must be non-negative")

        output_suffix = str(self.config.get("output_suffix", "_z_detrended"))
        trend_suffix = str(self.config.get("trend_suffix", "_z_trend"))
        out = catalog.copy()
        redshift = out[redshift_col].to_numpy(float)

        for col in self.required_columns():
            if col not in out:
                raise ValueError(f"Redshift trend requested missing column: {col}")
            trend_col = f"{col}{trend_suffix}"
            output_col = f"{col}{output_suffix}"
            self.ensure_new_columns(out, [trend_col, output_col], source=col)

            values = out[col].to_numpy(float)
            trend = np.full(len(out), np.nan, dtype=float)
            detrended = np.full(len(out), np.nan, dtype=float)
            finite = np.isfinite(redshift) & np.isfinite(values)
            if np.any(finite):
                trend_fit = self.polynomial_trend(
                    redshift[finite],
                    values[finite],
                    degree=degree,
                )
                center = self.trend_center(values[finite])
                trend[finite] = trend_fit
                detrended[finite] = values[finite] - trend_fit + center
            out[trend_col] = trend
            out[output_col] = detrended

        return out

    @staticmethod
    def polynomial_trend(
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

    def trend_center(self, values: np.ndarray) -> float:
        center = self.config.get("center", 0.0)
        if center in (None, False, "none"):
            return 0.0
        if center == "median":
            return float(np.nanmedian(values)) if len(values) else 0.0
        return float(center)


class ColumnRedshiftTrendCleaner(ConfiguredCleaner):
    """Remove binned redshift trends from selected columns."""

    config_name = "column_redshift_trend"

    def apply(self, catalog: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled():
            return catalog.copy()

        method = str(self.config.get("method", "binned_median")).lower()
        if method != "binned_median":
            raise ValueError("column_redshift_trend.method must be 'binned_median'")

        redshift_col = str(self.config.get("redshift_col", "z_phot"))
        if redshift_col not in catalog:
            raise ValueError(
                f"Column redshift trend requested missing column: {redshift_col}"
            )

        output = self.config.get("output")
        if output is not None:
            output = str(output).lower()
            if output not in {"overwrite", "suffix"}:
                raise ValueError(
                    "column_redshift_trend.output must be 'overwrite' or 'suffix'"
                )
            overwrite = output == "overwrite"
        else:
            overwrite = bool(self.config.get("overwrite", False))

        output_suffix = str(self.config.get("output_suffix", "_z_detrended"))
        bin_width = float(self.config.get("bin_width", 0.04))
        out = catalog.copy()
        redshift = out[redshift_col].to_numpy(float)

        for col in self.required_columns():
            if col not in out:
                raise ValueError(
                    f"Column redshift trend requested missing column: {col}"
                )
            output_col = col if overwrite else f"{col}{output_suffix}"
            if not overwrite:
                self.ensure_new_columns(out, [output_col], source=col)

            out[output_col] = self.binned_median_redshift_detrended(
                out[col].to_numpy(float),
                redshift,
                flux_like=self.trend_mode(col) == "multiplicative",
                bin_width=bin_width,
            )

        return out

    def trend_mode(self, column: str) -> str:
        column_modes = self.config.get("column_modes", {})
        if column_modes is None:
            column_modes = {}
        if not isinstance(column_modes, Mapping):
            raise ValueError("column_redshift_trend.column_modes must be a mapping")

        mode = column_modes.get(column, self.config.get("mode", "auto"))
        mode = str(mode).lower()
        if mode == "auto":
            return "multiplicative" if self.flux_like(column) else "additive"

        aliases = {
            "flux": "multiplicative",
            "ratio": "multiplicative",
            "multiplicative": "multiplicative",
            "mag": "additive",
            "magnitude": "additive",
            "color": "additive",
            "additive": "additive",
        }
        if mode not in aliases:
            raise ValueError(
                "column_redshift_trend.mode must be 'auto', 'flux', 'mag', "
                "'color', 'multiplicative', or 'additive'"
            )
        return aliases[mode]

    @staticmethod
    def flux_like(column: str) -> bool:
        lower = column.lower()
        return lower.startswith("flux_") or "_flux_" in lower

    @staticmethod
    def binned_median_redshift_detrended(
        values: np.ndarray,
        redshift: np.ndarray,
        *,
        flux_like: bool,
        bin_width: float,
    ) -> np.ndarray:
        if bin_width <= 0:
            raise ValueError("column_redshift_trend.bin_width must be positive")

        out = values.astype(float, copy=True)
        finite = np.isfinite(values) & np.isfinite(redshift)
        if np.sum(finite) < 2:
            return out

        try:
            from scipy.stats import binned_statistic
        except ImportError as exc:
            raise ImportError(
                "column_redshift_trend redshift detrending requires scipy"
            ) from exc

        bins = np.arange(
            np.nanmin(redshift) - 2.0 * bin_width,
            np.nanmax(redshift) + 2.0 * bin_width,
            bin_width,
        )
        if len(bins) < 2:
            return out

        medians, bin_edges, _ = binned_statistic(
            redshift,
            values,
            statistic=np.nanmedian,
            bins=bins,
        )
        centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        trend = np.interp(redshift[finite], centers, medians)
        center = np.nanmedian(values[finite])
        if flux_like:
            good_trend = np.isfinite(trend) & (trend != 0)
            finite_indices = np.where(finite)[0]
            out[finite_indices[good_trend]] = (
                values[finite_indices[good_trend]] / trend[good_trend] * center
            )
        else:
            out[finite] = values[finite] - trend + center
        return out


class BaseIsolationForestCleaner(ConfiguredCleaner):
    """Shared setup for IsolationForest-based cleaning operations."""

    sklearn_error_message = "IsolationForest cleaning requires scikit-learn"

    def isolation_forest_cls(self) -> Any:
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            raise ImportError(self.sklearn_error_message) from exc
        return IsolationForest

    def isolation_forest_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "n_estimators": int(self.config.get("n_estimators", 100)),
            "contamination": self.validate_contamination(
                self.config.get("contamination", "auto")
            ),
            "max_samples": self.config.get("max_samples", "auto"),
            "random_state": self.config.get("random_state", 42),
        }
        for key in ("max_features", "bootstrap", "n_jobs", "warm_start"):
            if key in self.config:
                kwargs[key] = self.config[key]
        return kwargs

    @staticmethod
    def validate_contamination(value: Any) -> float | str:
        if isinstance(value, str):
            if value != "auto":
                raise ValueError(
                    "IsolationForest contamination must be a float or 'auto'"
                )
            return value
        contamination = float(value)
        if not 0.0 < contamination <= 0.5:
            raise ValueError("IsolationForest contamination must be in (0, 0.5]")
        return contamination

    def scaled_features(self, features: np.ndarray) -> np.ndarray:
        scale = self.config.get("scale", "robust")
        if scale in (False, None, "none"):
            return features
        if scale != "robust":
            raise ValueError(
                "isolation_forest.scale must be 'robust', 'none', or false"
            )

        center = np.nanmedian(features, axis=0)
        mad = np.nanmedian(np.abs(features - center), axis=0)
        sigma = 1.4826 * mad
        sigma[~np.isfinite(sigma) | (sigma == 0)] = 1.0
        return (features - center) / sigma


class IsolationForestCleaner(BaseIsolationForestCleaner):
    """Remove multivariate outlier rows using scikit-learn IsolationForest."""

    config_name = "isolation_forest"
    sklearn_error_message = (
        "IsolationForest cleaning requires scikit-learn. Install the project "
        "dependencies or add scikit-learn to your environment."
    )

    def apply(self, catalog: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled():
            return catalog.copy()

        IsolationForest = self.isolation_forest_cls()
        columns = self.required_columns()
        out = catalog.copy()
        for col in columns:
            if col not in out:
                raise ValueError(f"IsolationForest requested missing column: {col}")

        drop_nonfinite = bool(self.config.get("drop_nonfinite", True))
        features = out[columns].to_numpy(float)
        finite = np.isfinite(features).all(axis=1)
        if drop_nonfinite:
            out = out.loc[finite].reset_index(drop=True)
            features = out[columns].to_numpy(float)
            finite = np.ones(len(out), dtype=bool)

        min_samples = int(self.config.get("min_samples", 2))
        if min_samples < 1:
            raise ValueError("isolation_forest.min_samples must be positive")
        if np.sum(finite) < min_samples:
            return out

        scaled = self.scaled_features(features[finite])
        model = IsolationForest(**self.isolation_forest_kwargs())
        labels = model.fit_predict(scaled)
        scores = model.decision_function(scaled)

        score_col = self.config.get("score_col", "isolation_forest_score")
        label_col = self.config.get("label_col", "isolation_forest_label")
        if score_col:
            out[str(score_col)] = np.nan
            out.loc[finite, str(score_col)] = scores
        if label_col:
            out[str(label_col)] = 0
            out.loc[finite, str(label_col)] = labels

        keep = ~finite if not drop_nonfinite else np.zeros(len(out), dtype=bool)
        keep[finite] = labels == 1
        return out.loc[keep].reset_index(drop=True)


class ColumnIsolationForestCleaner(BaseIsolationForestCleaner):
    """Mask per-column outliers with NaN while preserving sample rows."""

    config_name = "column_isolation_forest"
    sklearn_error_message = (
        "Column IsolationForest cleaning requires scikit-learn. Install the "
        "project dependencies or add scikit-learn to your environment."
    )

    def apply(self, catalog: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled():
            return catalog.copy()

        IsolationForest = self.isolation_forest_cls()
        columns = self.required_columns()
        context_columns = self.context_columns()
        out = catalog.copy()
        for col in columns + context_columns:
            if col not in out:
                raise ValueError(
                    f"Column IsolationForest requested missing column: {col}"
                )

        min_samples = int(self.config.get("min_samples", 2))
        if min_samples < 1:
            raise ValueError("column_isolation_forest.min_samples must be positive")

        for col in columns:
            feature_cols = [col] + context_columns
            features = out[feature_cols].to_numpy(float)
            finite = np.isfinite(features).all(axis=1)
            if np.sum(finite) < min_samples:
                continue

            scaled = self.scaled_features(features[finite])
            model = IsolationForest(**self.isolation_forest_kwargs())
            keep = model.fit_predict(scaled) == 1
            finite_indices = np.where(finite)[0]
            out.loc[finite_indices[~keep], col] = np.nan

        return out

    def context_columns(self) -> list[str]:
        columns = self.config.get("context_columns", [])
        if isinstance(columns, str):
            columns = [columns]
        redshift_col = self.config.get("redshift_col")
        if redshift_col is not None:
            columns = [*columns, str(redshift_col)]
        unique = []
        seen = set()
        for col in columns:
            col = str(col)
            if col not in seen:
                unique.append(col)
                seen.add(col)
        return unique


def clean_sample(catalog: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Apply optional sample cleaning."""
    return CleaningPipeline(catalog, config).clean()
