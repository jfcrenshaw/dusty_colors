"""Run some necessary post-processing steps for the DP1 catalog."""

import numpy as np
import healpy as hp
from astropy.table import Table, join
from healsparse import HealSparseMap

from dusty_colors.utils import fields, flux_to_mag

# Load the raw catalog
cat = Table.read("data/dp1_catalog_raw.fits")

# Match photo-z's
pz = Table.read("data/dp1_photoz_no_mags.parquet")
cat = join(cat, pz, keys_left="objectID", keys_right="objectId")
cat.remove_column("objectId")  # Remove duplicated ID column

# Match ECDFS redshift reference catalog
# Load the redshift reference catalog
zref = Table.read("data/comcam_ecdfs_crossmatched_catalog_20250618.parquet")
zref = zref[np.isfinite(zref["redshift"]) & (zref["confidence"] >= 0.95)]
zref = zref[["objectId", "redshift", "confidence", "type", "source"]]
cat = join(cat, zref, join_type="left", keys_left="objectID", keys_right="objectId")
cat.remove_column("objectId")  # Remove duplicated ID column


# De-redden fluxes

# EBV coefficients
band_a_ebv = dict(
    u=4.81,
    g=3.64,
    r=2.70,
    i=2.06,
    z=1.58,
    y=1.31,
)

for band in "ugrizy":
    # Load the EBV coefficient
    a_ebv = band_a_ebv[band]

    # Every type of flux for this band
    for col in cat.columns:
        if col.startswith(f"{band}_") and "Flux" in col and "Err" not in col:
            cat[col] = cat[col] * 10 ** (a_ebv * cat["ebv"] / 2.5)

# Save mags for every flux
for band in "ugrizy":
    for col in list(cat.columns):
        if col.startswith(f"{band}_") and "Flux" in col and "Err" not in col:
            ftype = col[2:].replace("Flux", "")
            mag, mag_err = flux_to_mag(cat, band, ftype=ftype)
            cat[f"{band}_{ftype}Mag"] = mag
            cat[f"{band}_{ftype}MagErr"] = mag_err

# Save SNR for every band
for band in "ugrizy":
    cat[f"{band}_snr"] = cat[f"{band}_cModelFlux"] / cat[f"{band}_cModelFluxErr"]

# Determine shear cuts, using Table 4 of https://arxiv.org/pdf/1705.06745
e = np.sqrt(cat["i_hsmShapeRegauss_e1"] ** 2 + cat["i_hsmShapeRegauss_e2"] ** 2)
R2 = 1 - (cat["i_ixxPSF"] + cat["i_iyyPSF"]) / (cat["i_ixx"] + cat["i_iyy"])

shear_cut = (
    (e < 2)  # Remove spurious shapes
    & (cat["i_hsmShapeRegauss_sigma"] <= 0.4)  # Shape uncertainty not too high
    & (cat["i_hsmShapeRegauss_flag"] == 0)  # Shape measurement succeeded
    & (cat["i_blendedness"] < 0.42)  # Object not too blended
    & (cat["i_iPSF_flag"] == 0)  # Successfully measured moments of PSF
    & (R2 >= 0.3)  # Galaxy sufficiently resolved (removes small artifacts)
)
cat["use_shear"] = shear_cut.data

# Estimate shear from shapes
R = 0.85  # Shear responsivity for HSM Regaussianization; typical value from HSC
cat["shear_g1"] = cat["i_hsmShapeRegauss_e1"] / (2 * R)
cat["shear_g2"] = cat["i_hsmShapeRegauss_e2"] / (2 * R)

# Save info related to shear uncertainties
i_snr = cat["i_cModelFlux"] / cat["i_cModelFluxErr"]
shape_noise = 0.264 * (i_snr / 20) ** -0.891 * (R2 / 0.5) ** -1.015
measurement_err = 0.4
cat["shear_err"] = np.sqrt(shape_noise**2 + measurement_err**2)

# Calculate pixelized depths
# --------------------------
# Set resolutions for healsparse maps
nside_coverage = 32
nside_sparse = 131072

# Assign each galaxy to a pixel
pixel_indices = hp.ang2pix(
    nside_sparse,
    np.deg2rad(cat["coord_dec"]),
    np.deg2rad(cat["coord_ra"]),
    lonlat=True,
)
cat["pixel"] = pixel_indices

# Loop over each band
for band in "ugrizy":
    # Calculate depth per galaxy...
    snr = np.clip(cat[f"{band}_snr"], 1, None)
    m5 = cat[f"{band}_cModelMag"] - 2.5 * np.log10(5 / snr)
    cat[f"{band}5"] = m5

    # Instantiate the map
    m5_map = HealSparseMap.make_empty(
        nside_coverage=nside_coverage,
        nside_sparse=nside_sparse,
        dtype=float,
    )

    # Loop over each pixel
    pixelized = cat.group_by("pixel")
    for pixel, data in zip(pixelized.groups.keys, pixelized.groups):
        # Set the median
        median = np.median(data[f"{band}5"], keepdims=True).astype(float)
        m5_map.update_values_pix(pixel[0], median)

    cat[f"{band}5_pixel"] = m5_map.get_values_pix(cat["pixel"])

# Save depths from DM maps
for band in "ugrizy":
    depth_map = HealSparseMap.read(f"data/deepCoadd_psf_maglim_{band}.fits")
    cat[f"{band}5_pixel_DM"] = depth_map.get_values_pos(
        cat["coord_ra"], cat["coord_dec"]
    )

# Save field names
field = np.empty(len(cat), dtype="U15")
for name in fields:
    mask = (cat["coord_ra"] - fields[name]["ra"]) ** 2 + (
        cat["coord_dec"] - fields[name]["dec"]
    ) ** 2 < 2**2
    field[mask] = name
cat["field"] = field

# Add columns for colors
bands = "ugrizy"
for i in range(len(bands) - 1):
    band1 = bands[i]
    band2 = bands[i + 1]
    cat[f"{band1}-{band2}"] = cat[f"{band1}_gaap1p0Mag"] - cat[f"{band2}_gaap1p0Mag"]
    cat[f"{band1}-{band2}_Err"] = np.sqrt(
        cat[f"{band1}_gaap1p0MagErr"] ** 2 + cat[f"{band2}_gaap1p0MagErr"] ** 2
    )

# Save the processed catalog
cat.write("data/dp1_catalog_processed.parquet", overwrite=True)
