"""Run some necessary post-processing steps for the DP1 catalog."""

import numpy as np
from astropy.table import Table, join

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

# Save the processed catalog
cat.write("data/dp1_catalog_processed.parquet", overwrite=True)
