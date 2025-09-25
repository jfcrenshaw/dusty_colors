"""Query the TAPS service for the raw DP1 catalog. Must be run on RSP!"""

from lsst.rsp import get_tap_service

# Open TAP service to query catalog
service = get_tap_service("tap")

# Programmatically build query
bands = "ugrizy"
query = "SELECT objectID, coord_ra, coord_dec, ebv, i_blendedness, refExtendedness"

# Lensing-related quantities
query += ", i_hsmShapeRegauss_e1, i_hsmShapeRegauss_e2"
query += ", i_hsmShapeRegauss_flag, i_hsmShapeRegauss_sigma"
query += ", i_iPSF_flag, i_ixxPSF, i_iyyPSF, i_ixx, i_iyy"

# Fluxes
for band in bands:
    query += f", {band}_cModelFlux,  {band}_cModelFluxErr"
for band in bands:
    query += f", {band}_free_cModelFlux,  {band}_free_cModelFluxErr"
for band in bands:
    query += f", {band}_gaap1p0Flux,  {band}_gaap1p0FluxErr"

# Specify the catalog
query += " FROM dp1.Object"

# Set selection criteria
query += " WHERE i_cModelFlux / i_cModelFluxErr > 5"
query += " AND i_centroid_flag = 0 AND i_blendedness_flag = 0"

for band in bands:
    query += f" AND {band}_cModel_flag = 0"

for band in bands:
    query += f" AND {band}_gaapFlux_flag = 0"

# Submit the job
job = service.submit_job(query)
job.run()
job.wait(phases=["COMPLETED", "ERROR"])
job.raise_if_error()
print("Job phase is", job.phase)

# Save catalog
results = job.fetch_result()
cat = results.to_table()
cat.write("data/dp1_catalog_raw.fits")
