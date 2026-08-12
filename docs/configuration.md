# Configuration reference

Every scientific choice in this analysis lives in YAML.
Command-line flags control only execution behaviour, such as forcing a stage to rerun.

There are three config levels, one per pipeline stage.
The sections below are ordered the way the stages run:

```text
  1. configs/catalogs/*.yaml   ->  2. configs/samples/*.yaml   ->  3. configs/analyses/*.yaml
```

Each config *references* the one before it, in the opposite direction: an analysis config names its sample, and a sample config names its catalog.
You run the pipeline by naming an analysis config, and the runner resolves the chain back to the catalog before executing forwards.

Each file needs an `id`, which determines its output directory under `results/`.
Two configs sharing an `id` will overwrite each other's outputs, so keep ids unique.

## Catalog config

Catalog YAML loads raw tables, joins them, adapts them to the canonical schema, applies corrections and enrichments, and writes `catalog.parquet` and `footprint.parquet`.

### `adapter`

Either `rubin_dp1` or `clauds_sextractor`.
Adapter options include `bands`, `photometry` (`flux` or `mag`), and `columns` for canonical column mapping.
`rubin_dp1` additionally accepts `flux_type` and `extendedness_min`.
`clauds_sextractor` additionally accepts `mag_kind`, `band_prefix`, `band_map`, `field`, and `apply_aperture_offset`.

### `primary_source` and `sources`

Each source has a `path` plus optional `rename`, `query`, `finite`, `drop_duplicates`, and `columns`.
Non-primary sources add a `join` block with either `on` or `left_key`/`right_key`, plus optional `how`, `suffixes`, `validate`, and `drop_right_key`.

### `photoz.combine`

Combines `estimates`, each with a `z` column and either `err` or `err_low`/`err_high`.
Output names can be set with `z_col`, `err_col`, and `diff_col`.
Each estimate also produces `photoz_<label>` and `photoz_sigma_<label>` diagnostic columns, which sample configs can cut on.

### `extinction`

Accepts `enabled`, `ebv_column`, `bands`, and per-band `coefficients`.

### `enrichments`

Two enrichments are available, `kcorrect` and `halo_mass`, each gated by `enabled`.

`kcorrect` accepts `model` or `responses`, plus `responses_out`, `responses_map`, `redshift_range`, `nredshift`, `abcorrect`, `interpolate_templates`, `response_bands`, `absmag_bands`, `redshift_col`, `min_redshift`, `max_redshift`, `error_floor`, `stellar_mass_col`, and `linear_stellar_mass_col`.

`halo_mass` uses the Moster et al. (2013) stellar-to-halo-mass relation.
It accepts `stellar_mass_col`, `stellar_mass_is_log`, `redshift_col`, `min_redshift`, `max_redshift`, `halo_mass_col`, `r200_col`, `log_mass_min`, and `log_mass_max`.

### `footprint`

Accepts `fields`, `field_radius_deg`, and HEALPix `nside`.

## Canonical catalog schema

Adapters convert survey-specific columns into a common schema before any selection or stacking code runs.
This is what lets one estimator serve more than one survey.

Required columns: `object_id`, `ra`, `dec`, `field`, `z_phot`, `z_phot_err`, `is_galaxy`, `mask_ok`, `quality_ok`.

Optional but preferred: `spec_z`, `stellar_mass_log`, `absmag_<band>`, `halo_mass_log`, `r200_mpc`, `pixel`, `jackknife_region`, `depth5_<band>`.

Photometry arrives in either form, and the observable builder derives colors internally:

- flux catalogs provide `flux_<band>` and `fluxerr_<band>`;
- magnitude catalogs provide `mag_<band>` and `magerr_<band>`.

Raw photometry is never overwritten by cleaning or observable construction.

## Sample config

Sample YAML points at a catalog config, applies foreground and background cuts, and writes `foreground.parquet` and `background.parquet`.

Before any configured cut runs, the pipeline always applies minimal validity filters: finite positions and redshifts, galaxy/mask/quality flags, finite requested photometry, and positive photometry errors.

### `selection`

Available options are `foreground_z`, `background_z`, `photoz_max_sigma`, `photoz_max_sigma_norm`, `photoz_estimate_max_sigma`, `photoz_max_diff_norm`, `blendedness_max`, `magnitude_limits`, `shared_query`, `foreground_query`, and `background_query`.

`pixel_depth_cuts` computes per-pixel limiting magnitudes from `fluxerr_template` using `depth_sigma`, then applies `valid_range`, `min_occupancy`, `complete_to`, and `drop_shallowest`.
The sample footprint written after these cuts defines the accepted random-catalog footprint, so these cuts matter to the estimator and not only to the sample.

Any structured cut can be switched off with `enabled: false`.

### `jackknife`

`regions_per_field` assigns angular-sector jackknife regions after the sample cuts.

### `cleaning`

Cleaning can be configured globally, or separately under `foreground` and `background` keys.

Available transforms are `finite_columns`, `robust_clip`, `redshift_trend`, `column_redshift_trend`, `isolation_forest`, and `column_isolation_forest`.

- `robust_clip` accepts `columns` and `sigma`.
- `redshift_trend` accepts `columns`, `redshift_col`, polynomial `degree`, `output_suffix`, `trend_suffix`, and `center`, and adds derived trend and trend-removed columns.
- `column_redshift_trend` applies binned-median redshift detrending, and can either write suffixed columns or overwrite the selected ones.
- `isolation_forest` accepts `columns`, `contamination`, `n_estimators`, `max_samples`, `random_state`, `min_samples`, `drop_nonfinite`, `scale`, `score_col`, `label_col`, and the scikit-learn options `max_features`, `bootstrap`, `n_jobs`, and `warm_start`.
- `column_isolation_forest` uses the same model options but masks outliers in each selected column with `NaN`, preserving rows.

The intended default is minimal cleaning.
Production stacking should rely on the estimator for systematics control (forward stack, flipped stack, random-footprint correction, reference-annulus subtraction, jackknife covariance) rather than on aggressive sample cleaning.

## Analysis config

Analysis YAML points at a sample config and owns only stack settings.
TreeCorr is the only stacker, so there is no `engine` option.

### `stack`

Options are `colors`, `modes` (`fcolors` and/or `mcolors`), `r_bin_edges`, `reference_annulus`, `snr_max`, `bin_slop`, `num_threads`, `jackknife`, `patch_col`, `cross_patch_weight`, `random_correction`, `random_multiplier`, `random_seed`, `random_nside`, `random_weighting`, `flipped_correction`, `diagnostic_plots`, `diagnostic_photoz_bins`, and `diagnostic_color_bins`.

`r_bin_edges` takes an explicit list or a declarative spec.
The loader deliberately does not evaluate Python or NumPy expressions from YAML strings.

```yaml
r_bin_edges: [10, 15, 40, 120, 1000]
# or
r_bin_edges:
  geomspace: {start: 5.0, stop: 1000.0, num: 6}
```

`linspace` and `logspace` are also supported.

Set `flipped_correction: false` to measure the forward stack minus the random forward stack, with the same reference-annulus subtraction, but without subtracting foreground-color flipped stacks.

`random_weighting` makes the otherwise uniform random catalogs match the real foreground and background distribution in per-pixel depth or noise features:

```yaml
stack:
  random_weighting:
    enabled: true
    depth:
      bands: [g, r, i, z]
      fluxerr_template: cmodel_fluxerr_{band}
      depth_sigma: 5
    n_bins: 5
```

### Stack outputs

Each configured mode writes `stack_<mode>.npz`, containing the science signal, jackknife arrays, and the color-space component profiles needed to rebuild correction variants without rerunning the stack.

`stack_<mode>_provenance.npz` holds raw-space profiles, pair counts, and derived intermediate estimator terms.

With `diagnostic_plots` enabled, `stack_<mode>_diagnostics.npz` holds pair-weighted background photo-z and magnitude-color histograms per radial bin.

Every run also writes `config_resolved.yaml` and refreshes standard figures in `results/stacks/<analysis_id>/`: one square log-log jackknife-sample plot for the first configured color, and one square log-log full-signal plot for every color, per mode.

## Plotting from Python

The stack plotting helpers are importable for manual use.
They read the color order from the analysis YAML and apply the project Matplotlib style.

```python
from dusty_colors.plotting import save_stack_figures

save_stack_figures(
    "configs/analyses/dp1_default.yaml",
    "figures",
    mode="fcolors",
)
```
