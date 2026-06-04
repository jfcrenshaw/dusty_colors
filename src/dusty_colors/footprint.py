"""Footprint, field, jackknife, and random-position helpers."""

from __future__ import annotations

from collections.abc import Mapping

import healpy as hp
import numpy as np
import pandas as pd


def assign_healpix_pixels(
    catalog: pd.DataFrame,
    *,
    nside: int,
    ra_col: str = "ra",
    dec_col: str = "dec",
    pixel_col: str = "pixel",
) -> pd.DataFrame:
    catalog = catalog.copy()
    catalog[pixel_col] = hp.ang2pix(
        nside,
        catalog[ra_col].to_numpy(float),
        catalog[dec_col].to_numpy(float),
        nest=True,
        lonlat=True,
    )
    return catalog


def assign_fields(
    catalog: pd.DataFrame,
    fields: Mapping[str, Mapping[str, float]],
    *,
    radius_deg: float = 2.0,
) -> pd.DataFrame:
    """Assign field labels from configured centers using a simple sky radius."""
    catalog = catalog.copy()
    assigned = np.full(len(catalog), "", dtype=object)
    ra = catalog["ra"].to_numpy(float)
    dec = catalog["dec"].to_numpy(float)
    for name, center in fields.items():
        dist2 = (ra - float(center["ra"])) ** 2 + (dec - float(center["dec"])) ** 2
        assigned[dist2 < radius_deg**2] = name
    if np.any(assigned == ""):
        assigned[assigned == ""] = "unknown"
    catalog["field"] = assigned
    return catalog


def assign_jackknife_regions(
    catalog: pd.DataFrame,
    *,
    regions_per_field: int,
    field_col: str = "field",
    output_col: str = "jackknife_region",
) -> pd.DataFrame:
    """Assign angular-sector jackknife regions independently per field."""
    if regions_per_field < 1:
        raise ValueError("regions_per_field must be positive")

    catalog = catalog.copy()
    regions = np.full(len(catalog), -1, dtype=int)
    offset = 0
    for field in sorted(catalog[field_col].dropna().unique()):
        use = catalog[field_col].to_numpy() == field
        sub = catalog.loc[use]
        center_ra = np.nanmedian(sub["ra"].to_numpy(float))
        center_dec = np.nanmedian(sub["dec"].to_numpy(float))
        theta = np.arctan2(
            sub["dec"].to_numpy(float) - center_dec,
            sub["ra"].to_numpy(float) - center_ra,
        )
        local = np.digitize(
            theta,
            np.linspace(-np.pi, np.pi, regions_per_field + 1),
        ) - 1
        local = np.clip(local, 0, regions_per_field - 1)
        regions[np.where(use)[0]] = local + offset
        offset += regions_per_field

    catalog[output_col] = regions
    return catalog


def footprint_table(catalog: pd.DataFrame) -> pd.DataFrame:
    columns = ["ra", "dec", "field", "pixel", "jackknife_region"]
    columns = [col for col in columns if col in catalog.columns]
    return catalog[columns].copy()


def sample_positions_in_pixels(
    pixels: np.ndarray,
    *,
    nside: int,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random sky positions inside a set of HEALPix pixels."""
    pixels = np.asarray(pixels, dtype=int)
    if len(pixels) == 0:
        raise ValueError("Cannot sample positions from an empty pixel set")
    pixel_set = set(pixels.tolist())
    lon, lat = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
    pixel_radius = np.rad2deg(np.sqrt(hp.nside2pixarea(nside)))
    ra_min = np.min(lon) - pixel_radius
    ra_max = np.max(lon) + pixel_radius
    dec_min = max(np.min(lat) - pixel_radius, -90.0)
    dec_max = min(np.max(lat) + pixel_radius, 90.0)

    ras: list[np.ndarray] = []
    decs: list[np.ndarray] = []
    have = 0
    while have < size:
        batch = max(1024, 4 * (size - have))
        ra = rng.uniform(ra_min, ra_max, size=batch)
        sin_dec = rng.uniform(
            np.sin(np.deg2rad(dec_min)),
            np.sin(np.deg2rad(dec_max)),
            size=batch,
        )
        dec = np.rad2deg(np.arcsin(sin_dec))
        pix = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
        keep = np.fromiter((p in pixel_set for p in pix), dtype=bool, count=batch)
        if np.any(keep):
            ras.append(ra[keep])
            decs.append(dec[keep])
            have += int(np.sum(keep))

    return np.concatenate(ras)[:size], np.concatenate(decs)[:size]
