# Dusty Colors YAML-First Refactor Design

## Goals

- Make TreeCorr the only stacking implementation.
- Remove legacy selector subclasses, the KDTree stacker, old script-driven analysis customization, and compatibility shims.
- Keep the code simple by separating three stages: catalog preparation, sample selection, and TreeCorr stacking.
- Make YAML describe science choices; use command-line flags only for execution behavior such as forcing a stage to rerun.
- Allow many stacking variants to reuse one prepared catalog or selected sample.
- Support DP1 and CLAUDS-HSC by adapting each input catalog into one canonical analysis schema.

## Architecture

The pipeline is a small three-stage graph:

```text
analysis.yaml -> sample.yaml -> catalog.yaml
```

Each stage has a stable `id`, a resolved config hash, expected outputs, and a `manifest.yaml`. A stage is skipped when its expected outputs exist and its manifest hash matches the resolved current config and relevant input hashes. If outputs exist but hashes differ, the runner should fail with a clear message unless the matching force flag is passed.

The output layout is:

```text
results/catalogs/<catalog_id>/
  catalog.parquet
  footprint.parquet
  manifest.yaml

results/samples/<sample_id>/
  foreground.parquet
  background.parquet
  manifest.yaml

results/stacks/<analysis_id>/
  stack_fcolors.npz
  stack_mcolors.npz
  config_resolved.yaml
  manifest.yaml
```

Catalog preparation owns raw input adaptation, field metadata, footprint pixels, and canonical schema validation. Sample selection owns foreground/background membership and optional diagnostic cleaning. TreeCorr stacking owns only the color-stack estimator and output stack files.

Catalog variants should be separate catalog YAML files when they change reusable
catalog products. For DP1, `configs/catalogs/dp1.yaml` uses the standard
kcorrect template set and `configs/catalogs/dp1_pai_blanton2024.yaml` uses the
expanded Pai & Blanton 2024 template set.

## YAML Interfaces

Catalog YAML owns raw input adaptation and footprint metadata:

```yaml
id: dp1
adapter: rubin_dp1
primary_source: objects
sources:
  objects:
    path: data/dp1_catalog_raw.fits
  photoz:
    path: data/dp1_photoz_no_mags.parquet
    join:
      left_key: objectID
      right_key: objectId
      how: inner
  specz:
    path: data/comcam_ecdfs_crossmatched_catalog_20250618.parquet
    query: confidence >= 0.95
    finite: [redshift]
    columns: [objectId, redshift, confidence, type, source]
    join:
      left_key: objectID
      right_key: objectId
      how: left
bands: [g, r, i, z]
photometry: flux
photoz:
  combine:
    estimates:
      - z: fzboost_z_mode
        err_low: fzboost_z_err68_low
        err_high: fzboost_z_err68_high
      - z: lephare_z_mode
        err_low: lephare_z_err68_low
        err_high: lephare_z_err68_high
extinction:
  enabled: true
  ebv_column: ebv
  bands: [u, g, r, i, z, y]
  coefficients: {u: 4.81, g: 3.64, r: 2.70, i: 2.06, z: 1.58, y: 1.31}
enrichments:
  kcorrect:
    enabled: true
    responses:
      - data/bandpasses/bandpass_u_v1p9.dat
      - data/bandpasses/bandpass_g_v1p9.dat
      - data/bandpasses/bandpass_r_v1p9.dat
      - data/bandpasses/bandpass_i_v1p9.dat
      - data/bandpasses/bandpass_z_v1p9.dat
      - data/bandpasses/bandpass_y_v1p9.dat
    response_bands: [u, g, r, i, z, y]
    max_redshift: 0.5
footprint:
  nside: 1024
  fields:
    ECDFS: {ra: 53.13, dec: -28.10}
    EDFS: {ra: 59.10, dec: -48.73}
    Rubin SV 95 -25: {ra: 95.00, dec: -25.00}
jackknife:
  regions_per_field: 3
```

Sample YAML owns reusable sample cuts:

```yaml
id: dp1_default
catalog: configs/catalogs/dp1.yaml
selection:
  foreground_z: [0.2, 0.5]
  background_z: [0.7, 1.4]
  photoz_max_sigma: 0.1
  shared_query: "mask_ok and quality_ok"
  foreground_query: null
  background_query: null
cleaning:
  enabled: false
```

Analysis YAML owns only stack settings:

```yaml
id: dp1_default
sample: configs/samples/dp1_default.yaml
stack:
  colors: [g-r, r-i, i-z, g-i]
  modes: [fcolors, mcolors]
  r_bin_edges:
    geomspace: {start: 5.0, stop: 1000.0, num: 6}
  reference_annulus: [2000.0, 4000.0]
  snr_max: 100
  random_correction: true
  random_multiplier: 5
  random_seed: 42
  jackknife: true
```

TreeCorr is the only stacker, so the analysis YAML should not include an `engine` field. Supported array specs should include explicit lists plus declarative NumPy-backed helpers: `geomspace`, `linspace`, and `logspace`. The loader should not evaluate Python or NumPy expressions from YAML strings.

## Canonical Catalog Schema

Catalog adapters convert source-specific columns into a common schema before any selection or stacking code runs.

Required canonical columns:

- `object_id`
- `ra`
- `dec`
- `field`
- `z_phot`
- `z_phot_err`
- `is_galaxy`
- `mask_ok`
- `quality_ok`

Optional but preferred canonical columns:

- `spec_z`
- `stellar_mass_log`
- `absmag_<band>`
- `halo_mass_log`
- `r200_mpc`
- `pixel`
- `jackknife_region`
- `depth5_<band>`

Photometry columns may be provided in either flux or magnitude form:

- Flux catalogs provide `flux_<band>` and `fluxerr_<band>`.
- Magnitude catalogs provide `mag_<band>` and `magerr_<band>`.
- The observable builder derives color observables internally.
- Raw photometry should not be overwritten by cleaning or observable construction.

The first adapters should be:

- `rubin_dp1`: maps assembled Rubin DP1 object, photo-z, and optional spec-z sources into canonical names.
- `clauds_sextractor`: maps CLAUDS-HSC SourceExtractor physical-parameter catalogs using columns such as `RA`, `DEC`, `ZPHOT`, `ZPDF_L68`, `ZPDF_U68`, `OBJ_TYPE`, `MASK`, `MASS_MED`, and HSC/CLAUDS magnitudes.

## Implementation Modules

- `config.py`: load YAML, resolve references, parse array specs, and compute stable config hashes.
- `sources.py`: load raw tables, apply local source filters, and join source tables from YAML.
- `catalogs.py`: adapter registry, canonical schema validation, and catalog preparation.
- `enrichments.py`: optional catalog-stage physical-property enrichments such as kcorrect stellar masses.
- `footprint.py`: HEALPix pixels, field assignment, jackknife regions, and random-position sampling.
- `selection.py`: foreground/background cuts and sample output.
- `cleaning.py`: optional diagnostic cleaning transforms.
- `observables.py`: flux-ratio and magnitude-color observable construction.
- `treecorr_stacker.py`: TreeCorr estimator only.
- `pipeline.py`: stage orchestration, manifest checks, and force behavior.
- `scripts/run_stack.py`: the single YAML-first runner.

The refactor should delete or replace the old selector and stacker structures rather than preserving import aliases. The public API should center on config loading, stage execution, and TreeCorr stack results.

## Cleaning Policy

Default cleaning should be minimal and explicit:

- require finite positions, redshifts, and requested photometry;
- apply catalog mask, object-type, and quality cuts;
- apply foreground/background redshift cuts;
- do not run IsolationForest by default;
- do not remove redshift trends by default;
- never overwrite raw photometry.

Optional diagnostic cleaning may add derived columns, but production stacking should rely on the estimator itself for systematics control: forward stack, flipped stack, random-footprint correction, reference-annulus subtraction, and jackknife covariance.

This replaces broad mutation-based cleaning with a small set of auditable cuts and optional derived diagnostics.

## Runner Behavior

The command shape should be:

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-stack
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-sample
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-catalog
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-all
```

Force behavior:

- `--force-stack`: recompute stack outputs only.
- `--force-sample`: recompute sample outputs and dependent stack outputs.
- `--force-catalog`: recompute catalog outputs plus dependent sample and stack outputs.
- `--force-all`: recompute every stage in the resolved graph.

The command line should not contain science parameters such as radial bins, color choices, sample cuts, or catalog column mappings. Those choices belong in YAML.

## Test Plan

- Config resolution and hash tests.
- Array parsing tests for explicit lists, `geomspace`, `linspace`, and `logspace`.
- DP1 adapter fixture test.
- CLAUDS SourceExtractor adapter fixture test.
- Sample reuse test with two stack YAMLs sharing one sample YAML.
- Manifest mismatch test requiring an appropriate force flag.
- Cleaning test proving raw photometry columns are not overwritten.
- TreeCorr smoke test on a tiny synthetic catalog.
- Random footprint test confirming sampled random positions remain inside allowed HEALPix pixels.
- Runner skip test confirming catalog and sample stages are skipped when only stack settings change.

## Assumptions

- This is a hard break; old imports, legacy scripts, selector subclasses, and the KDTree stacker can be deleted.
- TreeCorr is the only stacker, so no `engine` field is needed in analysis YAML.
- YAML config names are science-facing; CLI flags are operational only.
- CLAUDS-HSC first support targets the SourceExtractor plus Le Phare physical-parameter product.
- Project Python metadata should be aligned before tests are added, because `pyproject.toml` currently requires Python 3.13 while `environment.yml` uses Python 3.11.
