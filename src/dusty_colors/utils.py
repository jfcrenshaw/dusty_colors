"""Utilities for data processing."""

import numpy as np
import pandas as pd

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
