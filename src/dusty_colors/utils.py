"""Utilities for data processing."""

import matplotlib.pyplot as plt
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


def load_stack(stack, stack_type, r_norm=3):
    stack_dict = {}

    # Load both data files
    try:
        data = np.load(stack / f"stack_{stack_type}.npz")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Stack file not found: {stack / f'stack_{stack_type}.npz'}"
        )
    try:
        data_flipped = np.load(stack / f"stack_{stack_type}_flipped.npz")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Flipped stack file not found: {stack / f'stack_{stack_type}_flipped.npz'}"
        )

    # Get the keys
    keys = [key.split("_")[0] for key in data.keys() if key.endswith("_avg")]

    for key in keys:
        # Bin centers
        r = data[f"{key}_bin_centers"]

        # Load averages and errors
        x = data[f"{key}_avg"]
        err = data[f"{key}_err"]
        x_f = data_flipped[f"{key}_avg"]
        err_f = data_flipped[f"{key}_err"]

        if stack_type == "mags" or stack_type == "mcolors":
            # Correct signal using flipped data
            x = x - x_f
            err = np.sqrt(err**2 + err_f**2)

            # Normalize to large radii
            norm = np.nanmean(x[r > r_norm])
            x -= norm
        else:
            # Correct signal using flipped data
            x = x / x_f
            err = x / x_f * np.sqrt((err / x) ** 2 + (err_f / x_f) ** 2)

            # Normalize to large radii
            norm = np.nanmean(x[r > r_norm])
            x /= norm
            err /= norm

        # Store in dictionary
        stack_dict[f"{key}_bin_centers"] = r
        stack_dict[f"{key}_avg"] = x
        stack_dict[f"{key}_err"] = err

    return stack_dict


def plot_stack(stack, stack_type, r_norm=3):
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)

    stack_data = load_stack(stack, stack_type=stack_type, r_norm=r_norm)
    keys = [key.split("_")[0] for key in stack_data.keys() if key.endswith("_avg")]

    for key in keys:
        ax.errorbar(
            stack_data[f"{key}_bin_centers"],
            stack_data[f"{key}_avg"],
            yerr=stack_data[f"{key}_err"],
            label=key,
            alpha=0.7,
        )

    # Label the y axis
    if stack_type == "fluxes":
        ax.set_ylabel("Normalized flux")
    elif stack_type == "mags":
        ax.set_ylabel("Magnitude difference")
    elif stack_type == "fcolors":
        ax.set_ylabel("Normalized flux ratio")
    elif stack_type == "mcolors":
        ax.set_ylabel("Color difference")

    # Reference lines
    if stack_type == "fluxes" or stack_type == "fcolors":
        ax.axhline(1, color="k", ls="--", lw=0.8, alpha=0.5)
    else:
        ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)

    ax.legend(frameon=False, fontsize=8, handlelength=1)
    ax.set(xlabel="Impact parameter [Mpc]")

    return fig
