# dusty_colors

Measuring circumgalactic dust reddening by stacking background-galaxy colors around foreground galaxies.

This is the analysis code for [Crenshaw, McQuinn & Werk (2026)](https://arxiv.org/abs/2606.24159), *A First Measurement of Circumgalactic Dust Reddening from Only 1.7 deg² of the Rubin Observatory's DP1*.

The measurement works by stacking the colors of background galaxies as a function of projected separation from foreground galaxies.
Dust in the foreground galaxies' circumgalactic medium reddens the background light, so the stacked color excess as a function of radius traces the projected dust profile.
The estimator controls systematics by differencing a forward stack against a foreground-color-flipped stack, subtracting a random-position stack, and subtracting a large-radius reference annulus, with jackknife errors from angular sub-regions.

## Installation

The dependencies are declared in `pyproject.toml`.
Install them into a conda environment, since `healpy`, `treecorr`, and `kcorrect` are much easier to obtain from conda-forge than from PyPI.

```bash
conda create -n dusty_colors python=3.13
conda activate dusty_colors
pip install -e ".[dev,notebooks,data]"
python -m ipykernel install --user --name dusty_colors
```

Replace `conda` with `mamba` if you prefer.
The optional dependency groups are `dev` (formatting, linting, tests), `notebooks` (JupyterLab and plotting extras), and `data` (only needed for the download scripts in `scripts/`).

Verify the install:

```bash
pytest
```

## Running an analysis

Analyses are described entirely by YAML.
Nothing scientific is passed on the command line.

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml
```

That single command runs three stages, in this order:

```text
  data/                                   raw DP1 tables
    |
    |   STAGE 1: CATALOG   <--- configs/catalogs/rubin_dp1.yaml
    v
  results/catalogs/rubin_dp1/
      catalog.parquet, footprint.parquet
    |
    |   STAGE 2: SAMPLE    <--- configs/samples/dp1_default.yaml
    v
  results/samples/dp1_default/
      foreground.parquet, background.parquet, sample_report.md
    |
    |   STAGE 3: STACK     <--- configs/analyses/dp1_default.yaml
    v
  results/stacks/dp1_default/
      stack_fcolors.npz, config_resolved.yaml, figures
```

Note that you launch the pipeline by naming the **analysis** config, which belongs to the *last* stage.
That is because the configs reference each other backwards: an analysis config names the sample config it needs, and a sample config names the catalog config it needs.
The runner follows that chain of references down to the catalog, then executes forwards from the catalog back up to the stack.

Each stage writes a `manifest.yaml` recording a hash of its resolved config and its inputs.
A stage is skipped when its outputs already exist and its hash still matches, so rerunning an analysis that only changes stack settings will not rebuild the catalog.
If outputs exist but the hash has changed, the runner stops and tells you which force flag to pass:

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-stack
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-sample
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-catalog
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-all
```

Forcing a stage also forces everything downstream of it.

What each stage does:

1. **Catalog** loads the raw tables, joins them, adapts survey-specific columns into one canonical schema, applies extinction corrections and physical-property enrichments, and writes `catalog.parquet` plus `footprint.parquet`.
2. **Sample** reads that catalog, applies foreground and background cuts, optional cleaning, and jackknife region assignment, and writes `foreground.parquet` and `background.parquet` plus a human-readable `sample_report.md`.
3. **Stack** reads those two samples, runs the TreeCorr estimator, and writes one `stack_<mode>.npz` per configured mode, alongside diagnostic arrays and standard figures.

Every YAML key is documented in [docs/configuration.md](docs/configuration.md).
The stage machinery is described in [ARCHITECTURE.md](ARCHITECTURE.md).

## Reproducing the paper figures

The figures are produced by three notebooks, which between them need eleven analyses on disk.

The main measurement, the red/blue split, and the catalog-properties figures:

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml
python scripts/run_stack.py configs/analyses/dp1_red.yaml
python scripts/run_stack.py configs/analyses/dp1_blue.yaml
python scripts/run_stack.py configs/analyses/dp1_default_pai24.yaml
```

The systematics variants, used only by the sensitivity figure:

```bash
for v in uniform_random no_random no_flip smaller_ap larger_ap smaller_dz larger_dz; do
    python scripts/run_stack.py "configs/analyses/dp1_${v}.yaml"
done
```

Then run these notebooks, which write into `figures/`:

| Figure | Notebook | Reads |
| --- | --- | --- |
| `fig_photoz_distribution.pdf` | `plot_catalogs.ipynb` | sample `dp1_pai24` |
| `fig_foreground_cmd.pdf` | `plot_catalogs.ipynb` | sample `dp1_pai24` |
| `fig_foreground_properties.pdf` | `plot_catalogs.ipynb` | sample `dp1_pai24` |
| `fig_result_compare.pdf` | `plot_main_results.ipynb` | stack `dp1_default` |
| `fig_result_jackknife.pdf` | `plot_main_results.ipynb` | stack `dp1_default` |
| `fig_result_colors.pdf` | `plot_main_results.ipynb` | stack `dp1_default` |
| `fig_result_red_vs_blue.pdf` | `plot_main_results.ipynb` | stacks `dp1_red`, `dp1_blue` |
| `fig_result_sensitivity.pdf` | `plot_main_results.ipynb` | the 7 systematics variants above |
| `fig_chromaticity.pdf` | `plot_wavelength_dependence.ipynb` | stack `dp1_default` |
| `fig_r_band_stamp_grid.pdf` | `appendix/` (see below) | stack `dp1_default` |

Note that the catalog-properties figures read the **`dp1_pai24`** sample, which uses the expanded Pai & Blanton (2024) kcorrect template set, not the default template set.

The appendix stamp grid is different from the rest.
It lives under `appendix/`, must be run inside the Rubin Science Platform because it fetches image cutouts.

The remaining notebooks produce supporting material rather than paper figures: `plot_cleaning.ipynb`, `plot_jackknife_regions.ipynb`, and `calculate_dust_mass.ipynb`.

## Repository layout

```text
src/dusty_colors/   the package: config, pipeline stages, estimator, fits, plotting
  appendix/         appendix diagnostics, lower maintenance tier
scripts/            the pipeline runner plus one-off data download and build scripts
configs/            analysis, sample, and catalog YAML
notebooks/          figure and diagnostic notebooks
appendix/           appendix notebooks and their build script, see appendix/README.md
tests/              pytest suite
data/               raw inputs (gitignored)
results/            pipeline outputs (gitignored)
figures/            paper figures (gitignored)
```

Code under `appendix/` and `src/dusty_colors/appendix/` supports the paper appendices and is deliberately held to a lower standard than the core.
It is excluded from linting and type checking, and should not be taken as a model for how the rest of the code is written.
See [appendix/README.md](appendix/README.md).

## Development

```bash
pytest                      # run the test suite
black src tests scripts     # format
isort src tests scripts     # sort imports
flake8 src tests scripts    # lint
mypy src                    # type check
```

Notebook outputs are stripped before commit.
After cloning, enable this once:

```bash
nbstripout --install
```

Do not commit anything in `data/`, `results/`, or `figures/`.
Large catalog files committed by accident are extremely difficult to remove from history later.
