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

Sample configs go further and inherit with `extends`, which deep-merges a config onto a named base, so a variant states only what makes it different.
`extends` is consumed while loading, so a merged config hashes exactly like the equivalent standalone file and rewriting one does not invalidate results already on disk.
See [docs/configuration.md](docs/configuration.md) for the merge rules.

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
| `utils.py` | `get_root()`, plus the Matplotlib style and the shared figure size. |
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
| `randoms.py` | Random catalogs inside the footprint, and depth-matched weights for them. |
| `pair_histograms.py` | Pair-weighted histograms of background properties by separation, computed during the stack. |
| `results.py` | `StackResults` and the loaders that read a finished stack back off disk. |
| `postrun/` | Analyses that run automatically after a stack; see below. |

`scripts/run_stack.py` is the single pipeline entry point.
The other scripts are one-off data acquisition and model-building utilities, not part of the pipeline.

## Post-run analyses

Everything that consumes a finished stack — figures, fits, summary tables — is a *post-run analysis* living in `src/dusty_colors/postrun/`.

| Module | Responsibility |
| --- | --- |
| `postrun/base.py` | `PostRunContext`, the `@register` decorator, and the runner. |
| `postrun/plot_stacks.py` | The science figures: color excess profiles and their jackknife. |
| `postrun/plot_diagnostics.py` | Draws the histograms that `pair_histograms.py` computed during the stack. |
| `postrun/plot_chromaticity.py` | Band-relative excesses against reference extinction curves. |
| `postrun/dust_extinction_fit.py` | Fit an extinction law to the measured color excess profiles. |
| `postrun/color_power_law_fit.py` | Fit a power law to a single color profile. |
| `postrun/analysis_stats.py` | Summary statistics for a completed analysis. |

Figure modules are named `plot_*.py`, one per kind of figure.
Each figure is a single linear function that makes its axes, pulls the arrays it needs out of the stack, draws, labels, and returns.
That is deliberate: these functions are read and copied into notebooks far more often than they are called, so a reader should be able to follow one top to bottom without chasing helpers.
Prefer inlining a short expression twice over introducing a helper used twice, and keep module-level constants to genuinely shared conventions such as `COLOR_STYLES`, which exists so a color-pair looks the same in every figure and in the paper.

The stack stage calls `write_post_run_analyses` once; the registry decides what runs.
Adding an analysis is two steps: a new module with a `@register("name")` function returning the paths it wrote, and one import line in `postrun/__init__.py`.
The runner supplies the shared context, resolves the config block, and owns the error policy — a failing analysis warns and is skipped rather than discarding a stack that may have taken hours.

`PostRunContext` caches the loaded stack per mode and the representative foreground redshift, so several analyses reading the same arrays cost one read rather than one each.

Options come from the `postrun` block of the analysis YAML, which is **deliberately excluded from the stack config hash**.
Post-run analyses read a stack back off disk and cannot change it, so hashing their settings would mean retuning a fit parameter invalidated the manifest and demanded a full re-stack to regenerate a text report.
Excluding the block is safe for stacks already on disk: configs without a `postrun` key hash through the identical code path they always did.
Post-run analyses rerun on every invocation, including the skip path, so editing this block and running normally is enough to regenerate the products.
`--only-postrun` additionally refuses to run any stage, for when a figure tweak must not be able to trigger a rebuild.

The older convention of nesting these settings inside `stack:` is still read as a fallback, but settings there *are* hashed, so prefer `postrun:`.

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

- `treecorr_stacker.py` is still a single 1,448-line class, now holding the estimator and its TreeCorr plumbing after random-catalog generation and diagnostics moved out.
  Its remaining bulk is the estimator itself, which is closer to an appropriate size.
- `selection.py` is 1,044 lines, of which roughly 120 are Markdown and JSON report rendering.
  Splitting that out was tried and reverted: the rendering is reached through a single call, but the report *construction* it sits next to is called throughout the cuts, so the separation bought less than the extra module cost.
- The three-line pseudo-inverse that turns a Jacobian into a parameter covariance is repeated in each of the three fitting modules under `postrun/`, and the two `_profile_errors` helpers each have their own copy.
  The `_profile_errors` pair are deliberately different: `dust_extinction_fit` sanitises non-finite errors inside the helper because they feed a covariance block before any filtering, while `color_power_law_fit` filters at the call site.
- `chromaticity.py` fits the same radial power law as `color_power_law_fit.py` but with the chromatic shape held fixed, so the two share a model without sharing code.
- `pipeline.py` dispatches stages by dynamic string import, which is more indirection than three fixed stages need.
- `prefer_observable_columns` is a stack option that no config sets, still wired through `treecorr_stacker.py`.
