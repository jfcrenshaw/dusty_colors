"""Run some necessary post-processing steps for the DP1 catalog."""

import numpy as np
from astropy.table import Table, join

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

# Calculate pixelized depths in i band
# ------------------------------------

# First for individual galaxies...
i5 = cat["i_cModelMag"] - 2.5 * np.log10(5 / i_snr)
ra = cat["coord_ra"]
dec = cat["coord_dec"]

# Calculate 2D histogram of mean i5 values
# First, digitize the coordinates to get bin indices
# (-1 because digitize returns 1-based indices)
n_bins = 2000
ra_edges = np.linspace(ra.min(), ra.max(), n_bins + 1)
dec_edges = np.linspace(dec.min(), dec.max(), n_bins + 1)
ra_indices = np.digitize(ra, ra_edges) - 1
dec_indices = np.digitize(dec, dec_edges) - 1

# Clip indices to valid range (handle edge cases)
ra_indices = np.clip(ra_indices, 0, n_bins - 1)
dec_indices = np.clip(dec_indices, 0, n_bins - 1)

# Create arrays to store sum and count for each bin
bin_sum = np.zeros((n_bins, n_bins))
bin_count = np.zeros((n_bins, n_bins))

# Accumulate values in each bin
np.add.at(bin_sum, (ra_indices, dec_indices), i5)
np.add.at(bin_count, (ra_indices, dec_indices), 1)

# Calculate mean, avoiding division by zero
mean_i5_grid = np.divide(
    bin_sum,
    bin_count,
    out=np.full_like(bin_sum, np.nan),
    where=bin_count > 0,
)

# Assign mean i5 value to each galaxy based on its bin
galaxy_mean_i5 = mean_i5_grid[ra_indices, dec_indices]

# Save data
cat["i5"] = i5
cat["i5_pixel"] = galaxy_mean_i5

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
