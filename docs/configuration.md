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

## Inheriting from another config with `extends`

Most variants differ from a baseline in one or two settings, so they inherit the rest with `extends`:

```yaml
# configs/samples/dp1_no_ztrend.yaml
id: dp1_no_ztrend
extends: dp1_default.yaml
cleaning:
  background:
    column_redshift_trend:
      enabled: false
```

The path is relative to the extending file's own directory.
Chains are allowed, cycles are rejected, and a config using `extends` must declare its own `id` — inheriting the parent's would silently overwrite the parent's outputs.

Nested mappings merge key by key, so the example above changes one flag and leaves every other cleaning setting intact.
Everything else is replaced wholesale: lists, strings, and numbers take the override's value entirely.
That means a query string cannot be extended, only restated, which is why `configs/samples/dp1_red.yaml` repeats the whole `foreground_query` to add its color cut.

`extends` is consumed during loading, so a merged config produces exactly the same hash as the equivalent standalone file.
Rewriting a config to use `extends` therefore does not invalidate results already on disk.

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

Settings for fits and figures belong in [`postrun`](#postrun), not here.

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

Every run also writes `config_resolved.yaml` and refreshes the post-run analysis products described below.

## `postrun`

Analyses that run automatically after a stack completes, reading it back off disk.

This block is **not** part of the stack config hash.
Editing it therefore leaves the stack valid, and an ordinary run regenerates the derived products without recomputing anything:

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml
# catalog: skip / sample: skip / stack: skip, then the fits and figures are rewritten
```

Post-run analyses rerun on every invocation, including when every stage skips, so no flag is needed to pick up a change here.
`--only-postrun` is available when you want a guarantee that no stage can run: it refuses rather than building a missing catalog or sample, which is useful while iterating on a figure.

Each key names one registered analysis.
Set it to `false` (or `{enabled: false}`) to switch the analysis off, to `true` to run it with defaults, or to a mapping of its options.
All analyses are on by default.

```yaml
postrun:
  figures: true
  dust_extinction_fit:
    radial_pivot_kpc: 20.0
    covariance: diagonal_errors
    fixed_rv: 3.1
  color_power_law_fit: false
```

Settings nested inside `stack:` are still read as a fallback for configs written before this block existed, but settings there *are* hashed into the stack, so a change forces `--force-stack`.
Prefer `postrun:`.

### `figures`

Writes, per mode, one square log-log jackknife-sample plot for the first configured color and one square log-log full-signal plot for every color.
When the stack ran with `diagnostic_plots`, also writes pair-weighted photo-z and per-color radial distribution figures.

### `dust_extinction_fit`

Fits `E(b1-b2, r) = A_V(pivot) * (r/pivot)^alpha * [(A_b1/A_V) - (A_b2/A_V)]` and writes `dust_extinction_fit_<mode>.txt`.

Options are `colors`, `law` (any name in `dust_extinction.parameter_averages` or `.averages`, such as `F99`, `CCM89`, `G23`, `G03_SMCBar`), `radial_pivot_kpc`, `covariance` (`auto`, `full_jackknife`, `per_color_covariance`, `diagonal_errors`), `foreground_redshift`, `wavelengths_um`, `fixed_rv`, `amplitude_bounds`, `alpha_bounds`, `rv_bounds`, `initial_amplitude`, `initial_alpha`, and `initial_rv`.

The fit de-redshifts the filter wavelengths by `1 + z_fg` before evaluating the law, using the median foreground `z_phot` when available and the midpoint of `selection.foreground_z` otherwise.
Declines to run when the stack lacks any of the requested colors.

### `color_power_law_fit`

Fits `E(color) = A (r/pivot)^alpha` independently per color and writes `color_power_law_fits_<mode>.txt`.
Options are `colors`, `radial_pivot_kpc`, and the fit bounds.

### `chromaticity`

Writes `<analysis_id>_<mode>_extinction_curve_comparison.{png,pdf}`: one panel per radial bin showing band-relative color excesses against reference extinction curves, with the chromatic shape of each law held fixed so the figure tests the law rather than fitting it.

Options are `bands` (in wavelength order, default `[g, r, i, z]`), `reference_band`, `point_bands`, `radial_pivot_kpc`, `fit_bin_indices`, `laws`, `wavelengths_um`, `extension`, and `dpi`.

```yaml
postrun:
  chromaticity:
    radial_pivot_kpc: 20.0
    fit_bin_indices: [0, 1, 2]
    laws:
      - {name: F99, rv: 3.1, short: MW, label: "F99 MW $R_V=3.1$", color: "#1f5bd8"}
      - {name: G03_SMCBar, short: SMC, label: "G03 SMC Bar", color: "#2e7d32"}
```

Band-relative excesses are built by walking the chain of adjacent measured colors, so this needs every adjacent color between `reference_band` and each of `point_bands`; it declines to run otherwise.
Errors are propagated through the jackknife samples rather than added in quadrature, because adjacent colors share background objects.

### `analysis_catalog_stats`

Writes `analysis_catalog_stats.{txt,json}` into the *sample* directory, since the statistics describe the sample rather than one stack mode.

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
