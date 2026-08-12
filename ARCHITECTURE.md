# Architecture

How the code is organised and why.
For what every YAML key means, see [docs/configuration.md](docs/configuration.md).

## The central idea

The analysis is a three-stage pipeline in which each stage is a pure function of its config and the outputs of the stage before it.

Execution runs in this direction:

```text
  raw tables  -->  CATALOG  -->  SAMPLE  -->  STACK  -->  measured profiles
                      ^             ^            ^
                      |             |            |
             catalog.yaml    sample.yaml   analysis.yaml
```

Configuration references run in the *opposite* direction:

```text
  analysis.yaml  --names-->  sample.yaml  --names-->  catalog.yaml
```

You invoke the pipeline by naming an analysis config, which configures the last stage.
`load_resolved_config` walks the reference chain backwards to find the sample and catalog configs, and `run_pipeline` then executes forwards, catalog first.

The reason for the backwards references is that it makes variants cheap.
An analysis config is a few lines that name an existing sample, so trying a new radial binning costs one small file and reuses the catalog and sample already on disk.

Each stage writes its outputs plus a `manifest.yaml` recording a stable hash of its resolved config together with the hashes of its upstream stages.
On a rerun, a stage is skipped if its outputs exist and its hash still matches.
If the outputs exist but the hash has changed, the runner refuses to proceed and names the force flag that would overwrite them.

This is the most important design decision in the repo, and it buys three things.

- Changing a stack setting does not rebuild the catalog, which takes far longer.
- Many stacking variants share one prepared catalog and one selected sample, so the variant configs in `configs/analyses/` are cheap to run.
- Stale outputs cannot silently masquerade as current ones.

Output layout, listed in execution order:

```text
1. results/catalogs/<catalog_id>/   catalog.parquet, footprint.parquet, manifest.yaml
2. results/samples/<sample_id>/     foreground.parquet, background.parquet,
                                    sample_report.{md,json}, manifest.yaml
3. results/stacks/<analysis_id>/    stack_<mode>.npz, stack_<mode>_provenance.npz,
                                    config_resolved.yaml, manifest.yaml, figures
```

Note that the directory is named for the *config id* of the stage that produced it, so one catalog directory is typically shared by many sample directories, and one sample directory by many stack directories.

## Separation of concerns

Catalog preparation owns raw input adaptation, field metadata, footprint pixels, and canonical schema validation.
Sample selection owns foreground and background membership plus optional cleaning.
Stacking owns only the estimator and its output files.

Survey-specific column naming is confined to the catalog adapters.
Everything downstream sees one canonical schema, which is what allows a second survey to be added without touching the estimator.

Science choices live in YAML.
The command line carries only operational flags.
Radial bins, color choices, sample cuts, and column mappings are never passed as arguments.

## Modules

| Module | Responsibility |
| --- | --- |
| `paths.py` | `get_root()`, which locates the repository root by walking up to `pyproject.toml`. |
| `config.py` | Load YAML, resolve the three-stage reference graph, parse array specs, compute stable config hashes. |
| `pipeline.py` | Stage orchestration, manifest checks, force behaviour. |
| `sources.py` | Load raw tables, apply per-source filters, join source tables. |
| `catalogs.py` | Adapter registry, canonical schema validation, catalog preparation. |
| `enrichments.py` | Catalog-stage physical property enrichments: kcorrect stellar masses, halo masses. |
| `footprint.py` | HEALPix pixels, field assignment, jackknife regions, random position sampling. |
| `selection.py` | Foreground and background cuts, sample outputs and reports. |
| `cleaning.py` | Optional diagnostic cleaning transforms. |
| `observables.py` | Flux-ratio and magnitude-color observable construction. |
| `treecorr_stacker.py` | The TreeCorr estimator. |
| `plotting.py` | Stack result loading and figures. |
| `dust_extinction_fit.py` | Fit an extinction law to the measured color excess profiles. |
| `color_power_law_fit.py` | Fit a power law to a single color profile. |
| `analysis_stats.py` | Summary statistics for a completed analysis. |
| `postrun.py` | Post-run analyses invoked automatically after a stack. |

`scripts/run_stack.py` is the single pipeline entry point.
The other scripts are one-off data acquisition and model-building utilities, not part of the pipeline.

### Maintenance tiers

`src/dusty_colors/appendix/` holds the paper-appendix diagnostics and is deliberately held to a lower standard: excluded from linting and type checking, and not refactored alongside the core.
Its tests still run, because they are cheap and catch import breakage.

The dependency direction is enforced by convention rather than tooling: appendix modules may import from the core, but no core module may import from `appendix`.
Keeping that one-way is what allows the core to be refactored without consulting the appendix code.

## The estimator

The stack measures a color excess profile as a function of projected separation.
Systematics are controlled by construction rather than by cleaning the input samples:

- a **forward stack** of background colors around foreground positions;
- a **flipped stack**, which repeats the measurement with foreground colors flipped, and is subtracted to cancel color-independent selection effects (disable with `flipped_correction: false`);
- a **random-position stack**, subtracted to remove footprint and depth-driven artefacts, with optional depth-matched reweighting of the randoms;
- a **reference-annulus subtraction** at large radius, removing any residual constant offset;
- **jackknife covariance** from angular sub-regions assigned per field.

Two color modes exist.
`fcolors` works in flux-ratio space and is what the paper uses.
`mcolors` works in magnitude-color space and is retained as a cross-check.

## Known rough edges

Recorded honestly so they are not mistaken for intentional design.

- `treecorr_stacker.py` is a single 2,458-line class holding the estimator alongside random-catalog generation, radial rebinning, diagnostics, and TreeCorr plumbing.
  Splitting it is planned; see `CLEANUP_PLAN.md`.
- The radial rebinning and basis-cache machinery is not exercised by any config.
- `StackResults` and the stack-file loaders live in `plotting.py`, which is why several fit-module helpers are duplicated.
- Sample configs duplicate each other almost entirely, since there is no `extends:` mechanism yet.
- `pipeline.py` dispatches stages by dynamic string import, which is more indirection than three fixed stages need.
