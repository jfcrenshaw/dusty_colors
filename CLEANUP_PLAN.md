# Cleanup and restructure plan

Working document for tidying `dusty_colors` ahead of onboarding a student.
Delete this file once the work is done.

## Verdict

The core of this repo is in better shape than the directory listing suggests.
The three-stage YAML pipeline (`catalog -> sample -> stack`) with content-hashed manifests and skip/force semantics is a genuinely good design for reproducible analysis, and `config.py` and `pipeline.py` are clean and well documented.

The problems were concentrated in four places: repository hygiene, a 2,458-line god class, dead exploratory code that no config exercises, and documentation that described a refactor rather than the code.
None of this required an architectural rewrite.

Original size: 11,324 lines across 18 modules in `src/dusty_colors/`, 4,372 lines of tests, 1,714 lines of scripts.

**Progress so far.**
Hygiene, environment, and documentation are done, the appendix code is separated to its own tier, and the dead rebinning machinery is gone.
The stacker is down to 2,001 lines and the suite passes at 106 tests.
The remaining substantial work is splitting the stacker (Phase 4) and the config and pipeline tidying in Phase 5.

## What is working, and should not be touched

These are the parts worth defending during cleanup.

- The three-stage config graph and the `results/{catalogs,samples,stacks}/<id>/` output layout.
- Manifest hashing with explicit `--force-*` flags.
  This is the single most valuable thing in the repo for a student, because it makes reruns cheap and mistakes visible.
- The canonical catalog schema plus adapters, which keeps survey-specific column naming out of the science code.
- `config.py` (208 lines) and `pipeline.py` (584 lines), which are close to exemplary.
- Type hints and NumPy docstrings throughout.
- The fast, passing test suite.

## Findings

### 1. The `.git` directory was 19.4 GiB — RESOLVED 2026-08-12

`.git/objects/pack/pack-d881...pack` alone was 20.8 GB on disk.
It contained data blobs of 300 to 400 MB each (the DP1 catalog, photo-z, and cross-match parquet files) that were committed at some point and had become unreachable.
Total reachable content is under 1 MB.

Fixed by expiring the reflog and pruning:

```bash
git reflog expire --expire=now --all
git gc --prune=now
```

Result: 19.4 GiB down to 1.2 MB, a single 1.03 MiB pack with 854 objects.
Verified afterwards with `git fsck` (clean), 79 commits intact, 94 tracked files, working tree unchanged.

To keep it that way, make sure `data/`, `results*/`, and `figures/` stay gitignored, and add `nbstripout` so notebook image outputs do not re-inflate the history.

### 2. The environment is ambiguous and the documented setup is broken

Three competing stories exist:

- `README.md` says `conda env create -f environment.yml`, but `environment.yml` does not exist.
- `AGENTS.md` says to use the mamba environment `dusty_colors`.
- A `.venv/` running Python 3.14 is present in the repo root.

`pyproject.toml` also mixes development tooling (`black`, `isort`, `flake8`, `mypy`, `pytest`, `jupyterlab`) into runtime `dependencies`, so anyone installing the package gets the whole toolchain.

This is the number one practical blocker for a new person.
Pick one story, make it work from a clean checkout, and move the dev tools into an optional dependency group.

### 3. `treecorr_stacker.py` is a 2,458-line god class

`TreeCorrStacker` spans lines 89 to 2443, with 90 methods and 30 configuration fields on a single dataclass.
It tangles at least six responsibilities:

| Responsibility | Approximate lines | Size |
| --- | --- | --- |
| Sample loading and validation | 225-330 | ~105 |
| Jackknife setup | 286-332 | ~45 |
| Random catalogs and depth-matched reweighting | 332-835 | ~500 |
| Radial rebinning and basis caching | 891-1340 | ~450 |
| The estimator itself | 1342-2020 | ~680 |
| Diagnostic histograms | 2019-2180 | ~160 |
| TreeCorr plumbing | 2180-2450 | ~270 |

This is both the file a student is most likely to need to modify and the one that is hardest to enter.
Note that the estimator itself is only about 680 lines, which is reasonable.
The bulk of the file is support machinery that does not need to live in the same class.

### 4. Roughly 1,200 lines of code that no configuration exercises

These are all confirmed by grepping `configs/` and `results/`.

- **Radial rebinning and the basis cache**, about 450 lines, methods `_run_rebinned` through `_rebin_sum`.
  No config file sets `radial_rebinning`.
  Added 2026-06-12 in "Stacker saves mean r_perp; can cache fine bins".
- **The CLAUDS path**, about 615 lines total.
  `ClaudsSExtractorCatalogAdapter` (~140), `scripts/download_clauds.py` (175), `tests/test_clauds_smoke.py` (215), `tests/test_download_clauds.py` (83), plus 6 config files.
  `results/stacks/` contains no CLAUDS output and `paper.tex` never mentions CLAUDS.
- **The `mcolors` mode.**
  All 16 analysis configs set `modes: [fcolors]`, but the stacker, `observables.py`, and `pipeline.py` all carry the magnitude-color branch, and `pipeline.py` still defaults to both modes.
- **`prefer_observable_columns`**, set by no config.

### 5. Appendix analysis code lives in the core package — RESOLVED 2026-08-12

Moved to a lower maintenance tier rather than deleted, because the work is still paper-relevant.

- `color_split_bias.py` and `postage_stamps.py` moved to `src/dusty_colors/appendix/`.
- The four appendix notebooks and `build_dp1_inner_pair_table.py` moved to `appendix/`.
- Tests moved to `tests/appendix/` and still run.
- Excluded from `flake8` and `mypy` by configuration.
- `appendix/README.md` states the maintenance policy and the one-way dependency rule.

The notebooks each located the repo root with a different ad-hoc idiom that assumed they lived exactly one directory below the root.
All four now call `get_root()` from the new `paths.py`, so they work from any depth.

Three `src/dusty_colors` modules still hand-roll root resolution as `Path(__file__).resolve().parents[2]`: `sources.py`, `enrichments.py`, and `plotting.py`.
Those are correct today but encode the module's depth in the package, so anything moved into a subpackage silently resolves to the wrong root, exactly as the appendix move would have done.
Migrating them to `get_root(__file__)` is a good follow-up, but it needs care: they compute the root at import time, so a hard failure there would break importing the package rather than failing at the point of use.

Original finding follows.

### 5a. Original finding

`color_split_bias.py` (1,569 lines) and `postage_stamps.py` (616 lines) are inner-bin contamination diagnostics and RSP-only blend review.
Together they are 2,185 lines, or 19% of the package, and they are used only by three notebooks.

This is legitimate paper work, but mixing it into `dusty_colors/` makes the package look twice as large as its actual scientific core.
A student opening `src/dusty_colors/` should see the estimator, not the referee-response tooling.

### 6. `selection.py` mixes cuts with report formatting

Of its 1,044 lines, roughly 200 are Markdown and JSON report generation (`format_sample_report`, `_markdown_stage_table`, `_markdown_field_table`, `_json_ready`).
The selection logic and the reporting logic have no reason to share a file.

### 7. Config files duplicate each other almost entirely

There are 11 sample configs of about 66 lines each, several differing by a single line.
`configs/samples/dp1_no_ztrend.yaml` differs from `dp1_default.yaml` only in `id` and one `enabled: true` to `enabled: false`.

Adding a new cut currently means editing 11 files by hand.
A small `extends:` merge key in `config.py` would collapse these to one base plus ten short overrides.

Naming is also inconsistent.
`results/stacks/` contains both `dp1_red` and `dp1_red_default`, and the analysis `dp1_default_pai24` points at the sample `dp1_pai24`.
Several result directories (`dp1_default_500kpc`, `dp1_default_explore*`, `dp1_default_newbin`, `dp1_default_rand10`) have no corresponding config, because those configs were moved to `_old/`.
Those outputs can no longer be reproduced.

### 8. Documentation describes a refactor, not the code

- `README.md` is 191 lines and is almost entirely an exhaustive reference for every YAML key.
  It has no statement of what the project does scientifically, and its install instructions do not work.
  There is nothing that tells a reader how to get from raw data to a paper figure.
- `DESIGN.md` is a stale refactor plan written in the future tense: "The refactor should delete...", "The first adapters should be...".
  It describes work that is already finished, which makes it actively misleading.
- `AGENTS.md` is two lines and there is no `CLAUDE.md`.

### 9. Notebook problems

- `notebooks/plot_jackknives.ipynb` imports `from dusty_colors.utils import load_stack`.
  There is no `utils.py`.
  The notebook is broken.
- Notebook outputs are committed.
  `plot_jackknives.ipynb` is 199 KB and `dp1_magnification_bias_estimate.ipynb` is 124 KB, mostly base64 images.
- `notebooks/matplotlibrc` duplicates `src/dusty_colors/matplotlibrc`, differing only by a trailing newline.
- No notebook states which paper figure it produces.

### 10a. `paper.tex` was stale and actively misleading — RESOLVED 2026-08-12

`paper.tex` was last committed 2026-06-03 and included only 7 `\includegraphics` calls, omitting `fig_chromaticity.pdf` which is in the submitted manuscript.

This was a trap rather than harmless staleness: anyone reasoning about which figures matter by grepping it would silently get the wrong answer, which is exactly what happened during this cleanup.

Resolved by deleting the file.
The manuscript now lives only with the submission, and the README figure table is the repository's answer to "which figures are current".

### 10. Small duplication across modules

`_load_stack_source` and `_profile_errors` are byte-identical in `color_power_law_fit.py` and `dust_extinction_fit.py`.
`_parameter_covariance`, `_fit_bounds`, and `_initial_parameters` are near-identical.
`_require_columns`, `_stack_config`, `_catalog_nside`, and `_unique` each appear twice.

The root cause is that `StackResults` and `load_stack_results` live in `plotting.py`, so anything that needs to read a stack file has to either import from a plotting module or reimplement the loader.

### 11. Over-engineering in the stage dispatcher

`pipeline.py` uses `_handler_candidates()` to import stage functions by string name, `_wrap_domain_handler()` to build per-stage lambdas, and an injectable `StageHandlers` dataclass.
That is about 80 lines of indirection for a fixed three-stage pipeline whose stages are known at import time.
A module-level dict of three functions would be clearer, and the tests could import the stage functions directly.

### 12. Test suite shape

`tests/test_catalog_sample_slice.py` is 1,609 lines, 37% of all test code, in one file.
About 400 lines test one-off download scripts, which is a lot of investment relative to the estimator.
Test filenames do not always map to module names, for example `test_analysis_catalog_stats.py` covers `analysis_stats.py`.

### 13. Working-tree clutter

Untracked or ignored but present: `_old/` (9 files), `results_safe/` (3.6 GB), `build/`, `.mypy_cache/`, `.pytest_cache/`, `.venv/`, `paper.log`, three `.DS_Store` files.
`pzserver_token.txt` is correctly gitignored, but a credential sitting in the repo root is worth moving to `~/.config` or an environment variable so it cannot be committed by accident.

## Proposed structure

Keep the package flat.
Eighteen flat modules is slightly too many, but nested subpackages would be worse for someone learning the code: a flat directory where every filename says what it does is easier to navigate than a tree.
The one exception is the appendix code, which benefits from being visibly separated.

```text
src/dusty_colors/
  config.py            # unchanged, add `extends:` merge
  pipeline.py          # simplify stage dispatch
  sources.py           # unchanged
  catalogs.py          # drop CLAUDS adapter
  enrichments.py       # unchanged
  footprint.py         # unchanged
  selection.py         # cuts only
  reports.py           # NEW: sample report formatting, out of selection.py
  cleaning.py          # unchanged
  observables.py       # drop mcolors branch if confirmed
  stacker.py           # NEW: core estimator, ~700 lines
  randoms.py           # NEW: random catalogs + depth weighting, ~500 lines
  diagnostics.py       # NEW: diagnostic histograms, ~200 lines
  results.py           # NEW: StackResults + loaders, out of plotting.py
  plotting.py          # pure plotting
  dust_extinction_fit.py
  color_power_law_fit.py
  analysis_stats.py
  postrun.py
  appendix/            # NEW: paper appendix / referee-response tooling
    color_split_bias.py
    postage_stamps.py
```

The important moves are:

1. `treecorr_stacker.py` splits into `stacker.py`, `randoms.py`, and `diagnostics.py`, with the rebinning machinery deleted.
2. A new `results.py` owns `StackResults` and stack-file loading.
   This dissolves most of the cross-module duplication in finding 10 without inventing a utils grab-bag.
3. `appendix/` makes the boundary between core method and paper-specific diagnostics explicit.

## Phased plan

Phases are ordered so that each one leaves the repo working and the tests passing.
Phases 1 and 2 are pure wins with no science risk and are worth doing regardless of what you decide about the rest.

### Phase 1: hygiene (about half a day, no code changes)

- ~~Reclaim the 20 GB of git history.~~ Done 2026-08-12, see finding 1.
- ~~Delete `build/`, `.mypy_cache/`, `.pytest_cache/`, `paper.log`, and the `.DS_Store` files.~~ Done 2026-08-12.
- ~~`results_safe/` (3.6 GB).~~ Removed 2026-08-12.
- Still present locally, all gitignored so invisible to a fresh clone: `_old/` (88 KB), `.venv/` (301 MB), and 8 orphaned `results/stacks/` directories (1.5 MB) whose configs now live in `_old/`.
- Move `pzserver_token.txt` out of the repo.
- Add `nbstripout` so notebook outputs stop entering git.

### Phase 2: environment and documentation — DONE 2026-08-12

- Rewrote `pyproject.toml`: runtime dependencies separated from `dev`, `notebooks`, and `data` optional groups.
  Added the missing `pyyaml` dependency and dropped the unused `healsparse`.
- Rewrote `README.md`: science summary, working install instructions, an end-to-end run walkthrough, and a table mapping all 7 paper figures to the 2 notebooks that produce them.
- Moved the YAML key reference into `docs/configuration.md`.
- Replaced the stale future-tense `DESIGN.md` with a present-tense `ARCHITECTURE.md` that also records known rough edges.
- Merged `AGENTS.md` into `CLAUDE.md`, leaving `AGENTS.md` as a symlink so both agent conventions read one file.

Tests pass and all internal documentation links resolve.

### Phase 3: delete dead code — DONE 2026-08-12

- Removed radial rebinning and the basis cache: 451 lines from `treecorr_stacker.py`, plus two tests.
  The stacker went from 2,458 to 2,001 lines.
  `parse_array_spec` and `stable_hash` imports became unused and were dropped.
- CLAUDS and `mcolors` were kept by decision, so they are no longer counted as dead code.
- `prefer_observable_columns` is still present and still unused.

Verified no regression: mypy went from 77 errors to 73 (the removed code accounted for 4), `src/` is now flake8-clean, and the suite passes at 106 tests.

### Phase 4: split the stacker — LARGELY DONE 2026-08-12

Guarded throughout by `tests/test_stacker_characterization.py`, which pins all 284 output arrays bit-exactly.
Every step below left them unchanged.

- Added the characterization test first, closing a real gap: no test ran `stacker.run()` at all, so the estimator had no output-level coverage.
- Extracted `randoms.py` (649 lines): random catalog construction and depth-matched reweighting.
- Extracted `diagnostics.py` (180 lines): pair-weighted histograms, with no knowledge of colors or modes.

`treecorr_stacker.py` went 2,458 to 1,448 lines across Phases 3 and 4.
Both extractions use plain functions with explicit arguments rather than mixins, so the moved code can be read and tested without constructing a stacker.

The public interface is unchanged throughout: all 25 dataclass fields, `run_treecorr_stack`, the YAML schema, and `scripts/run_stack.py` are untouched.

Remaining, if wanted: `selection.py` reporting split, and the `results.py` extraction that would dissolve the duplicated fit helpers.

### Phase 4 original plan

- Extract `randoms.py`, then `diagnostics.py`, then leave `stacker.py` as the estimator.
- Add characterization tests first: run `dp1_default` before the split, save the output NPZ, and assert bit-identical output afterwards.
  This is the safety net that makes the refactor boring instead of frightening.
- Split reporting out of `selection.py` into `reports.py`.
- Create `results.py` and delete the duplicated loader helpers.

### Phase 5: configs and appendix (about a day)

- Add `extends:` support to `config.py` and collapse the 11 sample configs.
- Normalise config and result directory naming.
- Move `color_split_bias.py` and `postage_stamps.py` into `appendix/`.
- Simplify the `pipeline.py` stage dispatch.

### Phase 6: notebooks (about a day)

- Fix or delete `plot_jackknives.ipynb`.
- Delete `notebooks/matplotlibrc` in favour of the packaged one.
- Add a title cell to each notebook naming the paper figure it produces.
- Delete or explain the three unused figures.

## Decisions taken 2026-08-12

1. **Environment**: `pyproject.toml` is the single source of truth, installed into a conda environment.
   No `environment.yml`.
2. **`mcolors`**: keep.
   The magnitude-color branch stays in the stacker, `observables.py`, and `pipeline.py`.
3. **CLAUDS**: keep.
   The adapter, download script, tests, and configs all stay.
4. **Appendix code**: frozen, archive it.
   `color_split_bias.py` and `postage_stamps.py` move out of the installed package and out of the test suite, with no further cleanup investment.

Consequence: Phase 3 shrinks to radial rebinning plus `prefer_observable_columns`, roughly 470 lines rather than 1,200.

## Still open

1. **Radial rebinning (about 450 lines).**
   No config sets it, and it is now the only substantial dead code left.
   It is a caching and performance feature rather than a science option, so deleting it does not remove any measurement capability.
   Confirm before Phase 3.

2. **Are `fig_result_sensitivity.pdf` and any other figures current?**
   `fig_r_band_stamp_grid.pdf` is confirmed relevant and now documented.
   `fig_result_sensitivity.pdf` is listed in the README on the assumption it is current, since it is freshly produced from seven systematics variants.
   Confirm, and see finding 10a about `paper.tex` being an unreliable source for this.

3. **How much do you want to preserve exact reproducibility of the published numbers?**
   Assumed yes for now, so Phase 4 will be treated as a strict no-op guarded by characterization tests.
   Say otherwise if that is too conservative.
