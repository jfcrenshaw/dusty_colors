"""Utilities for data processing."""

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from sklearn.ensemble import IsolationForest

fields = {
    "ECDFS": {"ra": 53.13, "dec": -28.10},
    "EDFS": {"ra": 59.10, "dec": -48.73},
    "Rubin SV 95 -25": {"ra": 95.00, "dec": -25.00},
}


def flux_to_mag(cat: pd.DataFrame, band: str, ftype: str = "cModel"):
    """Convert fluxes in nJy to AB mags."""
    # Get fluxes and errors
    flux = cat[f"{band}_{ftype}Flux"]
    flux_err = cat[f"{band}_{ftype}FluxErr"]

    # Convert to AB mags
    with np.errstate(divide="ignore"):
        mag = -2.5 * np.log10(np.clip(flux, 0, None)) + 31.4
        mag_err = 2.5 / np.log(10) * flux_err / np.clip(flux, 0, None)

    return mag, mag_err


def select_ecdfs(cat: pd.DataFrame) -> pd.DataFrame:
    """Select only objects in the ECDFS region."""
    return cat.query(
        "coord_ra > 52 and coord_ra < 54 and coord_dec > -29 and coord_dec < -27"
    )


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean data by removing redshift trends and rejecting outliers."""
    data = data.copy()
    for col in data.columns:
        if (
            "Err" in col
            or ".mask" in col
            or not ("Mag" in col or "-" in col or "Flux" in col)
        ):
            continue

        # Remove median trend
        dz = 0.04
        bins = np.arange(data.z_phot.min() - 2 * dz, data.z_phot.max() + 2 * dz, dz)
        stat, bin_edges, _ = binned_statistic(
            data.z_phot,
            data[col],
            statistic=np.nanmedian,
            bins=bins,
        )
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Subtract off trend from mag data
        if "Mag" in col or "-" in col:
            data[col] = (
                data[col]
                - np.interp(data.z_phot, bin_centers, stat)
                + np.nanmedian(data[col])
            )
        # Divide out trend from flux data
        elif "Flux" in col:
            data[col] = (
                data[col]
                / np.interp(data.z_phot, bin_centers, stat)
                * np.nanmedian(data[col])
            )

        # Remove outliers with isolation forest
        X = np.column_stack([data[col].values, data.z_phot.values])
        iso = IsolationForest(random_state=42)
        mask = iso.fit_predict(X) == 1
        data.loc[mask == False, col] = np.nan

    return data
