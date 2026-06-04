# dusty_colors

Studying impact of foreground dust on background galaxy colors

## Installation

If running on the RSP, everything should already be installed.

If running locally:

```bash
conda env create -f environment.yml
conda activate dusty_colors
python -m ipykernel install --user --name dusty_colors
```

Note you can replace conda -> mamba if you are a mamba user.

## Running an analysis

Analyses are YAML-first.
Catalog preparation, sample selection, and TreeCorr stacking are described by
config files under `configs/`.
The default DP1 catalog uses the standard kcorrect template set;
`configs/catalogs/dp1_pai_blanton2024.yaml` uses the expanded Pai & Blanton 2024
template set.

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml
python scripts/run_stack.py configs/analyses/dp1_pai_blanton2024_default.yaml
```

The runner skips stages whose outputs and manifests already match the resolved
YAML graph.
Use operational force flags to recompute stages:

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-stack
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-sample
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-catalog
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-all
```

## YAML pipeline

The config graph is:

```text
analysis.yaml -> sample.yaml -> catalog.yaml
```

Each stage has an `id`, a config hash, expected outputs, and a `manifest.yaml`.
Catalog outputs are written to `results/catalogs/<catalog_id>/`, samples to
`results/samples/<sample_id>/`, and stacks to `results/stacks/<analysis_id>/`.

### Catalog preparation

Catalog YAML files load raw tables, join sources, adapt them to the canonical
schema, add optional catalog-level corrections/enrichments, and write
`catalog.parquet` plus `footprint.parquet`.

Useful catalog options:

- `adapter`: currently `dp1` or `clauds_sextractor`; adapter options include
  `bands`, `photometry` (`flux` or `mag`), `columns` for canonical column
  mapping, DP1 `flux_type`/`extendedness_min`, and CLAUDS `mag_kind`,
  `band_prefix`, `band_map`, `field`, and `apply_aperture_offset`.
- `primary_source` and `sources`: each source has `path` plus optional `rename`,
  `query`, `finite`, `drop_duplicates`, and `columns`. Non-primary sources add
  `join` with either `on` or `left_key`/`right_key`, plus optional `how`,
  `suffixes`, `validate`, and `drop_right_key`.
- `photoz.combine`: combines `estimates` with `z` and either `err` or
  `err_low`/`err_high`; optional output names are `z_col`, `err_col`, and
  `diff_col`. Each estimate also gets a `photoz_<label>` and
  `photoz_sigma_<label>` diagnostic column for stricter sample cuts.
- `extinction`: `enabled`, `ebv_column`, `bands`, and per-band `coefficients`.
- `enrichments`: `kcorrect` and `halo_mass`, each with `enabled`. `kcorrect`
  accepts `model` or `responses`, plus `responses_out`,
  `responses_map`, `redshift_range`, `nredshift`, `abcorrect`,
  `interpolate_templates`, `response_bands`, `absmag_bands`, `redshift_col`,
  `min_redshift`, `max_redshift`, `error_floor`, `stellar_mass_col`, and
  `linear_stellar_mass_col`. `halo_mass` uses the Moster et al. (2013)
  stellar-to-halo-mass relation and accepts `stellar_mass_col`,
  `stellar_mass_is_log`, `redshift_col`, `min_redshift`, `max_redshift`,
  `halo_mass_col`, `r200_col`, `log_mass_min`, and `log_mass_max`.
- `footprint`: `fields`, `field_radius_deg`, and HEALPix `nside`.

### Sample selection and cleaning

Sample YAML files point at a catalog YAML, apply foreground/background cuts, and
write `foreground.parquet` and `background.parquet`.

Selection options include `foreground_z`, `background_z`, `photoz_max_sigma`,
`photoz_max_sigma_norm`, `photoz_estimate_max_sigma`, `photoz_max_diff_norm`,
`blendedness_max`, `magnitude_limits`, `shared_query`,
`foreground_query`, and `background_query`. `pixel_depth_cuts` computes
per-pixel limiting magnitudes from `fluxerr_template` using `depth_sigma`, then
applies `valid_range`, `min_occupancy`, `complete_to`, and
`drop_shallowest`; the sample footprint written after these cuts defines the
accepted random-catalog footprint. `jackknife.regions_per_field` assigns
angular-sector jackknife regions after sample cuts. Optional structured cuts can
be turned off with `enabled: false`. Before these cuts are written, the pipeline
always applies minimal validity filters: finite positions/redshifts,
galaxy/mask/quality flags, finite requested photometry, and positive photometry
errors.

Cleaning can be configured globally or separately for `foreground` and
`background`. The cleaning block can use `finite_columns`, `robust_clip`,
`redshift_trend`, `column_redshift_trend`, `isolation_forest`, and
`column_isolation_forest`. `robust_clip` accepts `columns` and `sigma`.
`redshift_trend` accepts `columns`, `redshift_col`,
polynomial `degree`, `output_suffix`, `trend_suffix`, and `center`; it adds
derived trend and trend-removed columns. `column_redshift_trend` applies
binned-median redshift detrending to selected columns and can either write
suffixed columns or overwrite the selected columns. `isolation_forest` accepts
`columns`, `contamination`, `n_estimators`, `max_samples`, `random_state`,
`min_samples`, `drop_nonfinite`, `scale`, `score_col`, `label_col`, and the
scikit-learn options `max_features`, `bootstrap`, `n_jobs`, and `warm_start`.
`column_isolation_forest` uses the same model options but masks outliers in each
selected column with `NaN` while preserving rows.

### TreeCorr stacking

Analysis YAML files point at a sample YAML and own only stack settings. TreeCorr
is the only stacker; there is no `engine` option. The stack stage consumes the
prepared sample and catalog footprint, then writes one `stack_<mode>.npz` per
configured mode plus `config_resolved.yaml`.

Stack options include `colors`, `modes` (`fcolors` and/or `mcolors`),
`r_bin_edges` as an explicit list or `geomspace`/`linspace`/`logspace`,
`reference_annulus`, `snr_max`, `bin_slop`, `num_threads`, `jackknife`,
`patch_col`, `cross_patch_weight`, `random_correction`, `random_multiplier`,
`random_seed`, `random_nside`, `flipped_correction`, `diagnostic_plots`,
`diagnostic_photoz_bins`, and `diagnostic_color_bins`. Set
`flipped_correction: false` to measure the forward stack minus the random forward
stack, with the same reference-annulus subtraction, without subtracting
foreground-color flipped stacks.

Each analysis run also refreshes standard stack figures in
`results/stacks/<analysis-id>`: one square log-log jackknife-sample plot for the
first configured color, and one square log-log full-signal plot for every color,
for each configured stack mode. With `diagnostic_plots` enabled, stack outputs
also include pair-weighted background photo-z and magnitude-color histograms for
each radial bin; the pipeline saves one photo-z diagnostic and one color
diagnostic per configured color.

Stack plotting helpers also live in `dusty_colors.plotting` for manual use. They
load the analysis YAML color order and apply the project Matplotlib style:

```python
from dusty_colors.plotting import save_stack_figures

save_stack_figures(
    "configs/analyses/dp1_pai24_default.yaml",
    "figures",
    mode="fcolors",
)
```
