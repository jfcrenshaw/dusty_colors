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
    """Apply simple diagnostic cleaning without overwriting raw photometry."""
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
        columns = list(clip.get("columns", []))
        sigma = float(clip.get("sigma", 5.0))
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
