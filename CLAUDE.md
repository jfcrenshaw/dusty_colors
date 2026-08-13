# Project instructions for dusty_colors

Analysis code for [arXiv:2606.24159](https://arxiv.org/abs/2606.24159), measuring circumgalactic dust reddening from Rubin DP1.

Read [ARCHITECTURE.md](ARCHITECTURE.md) before making structural changes.
YAML keys are documented in [docs/configuration.md](docs/configuration.md).

## Environment

Always run commands in the `dusty_colors` conda environment:

```bash
mamba run -n dusty_colors <command>
```

Dependencies are declared in `pyproject.toml` only.
There is no `environment.yml`.
Runtime dependencies go in `[project.dependencies]`; tooling goes in the `dev`, `notebooks`, or `data` optional groups.

## Commands

```bash
mamba run -n dusty_colors pytest                    # full suite, ~8s
mamba run -n dusty_colors black src tests scripts
mamba run -n dusty_colors isort src tests scripts
mamba run -n dusty_colors flake8 src tests scripts
mamba run -n dusty_colors mypy src
```

Run an analysis:

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml [--force-{stack,sample,catalog,all}]
```

## Conventions

- NumPy-style docstrings.
- Black with line length 88.
- Markdown and LaTeX: one sentence per line.
- **Comment any non-obvious code inline, explaining *why* rather than *what*.**
  A student should never have to reverse-engineer intent from the implementation.
  See `src/dusty_colors/randoms.py` for the intended density and style.
- Science choices belong in YAML, never in command-line arguments or hardcoded constants.
- Never count parent directories to find the repository root.
  Use `from dusty_colors import get_root`, optionally with `get_root(__file__)` when the result must not depend on the working directory.
- Notebooks that plot must call `use_matplotlib_style()` from `dusty_colors`.
  The style is not picked up implicitly, and `src/dusty_colors/matplotlibrc` is the only copy.
- Do not put `backend` in `matplotlibrc`; it is appearance-only.
  `rc_file()` ignores the key anyway, and a script needing a specific backend should call `mpl.use(...)` itself.
- Never overwrite raw photometry columns; cleaning adds derived columns instead.
- Adding a survey means adding a catalog adapter, not touching the estimator.
- **Do not fragment code into tiny helpers.**
  A helper earns its name by removing real duplication across several call sites, or by organising a large thematically unified block.
  A three-line function called once does neither; it just makes the reader jump.
  This applies everywhere, and doubly to figures: each figure is one linear function that makes the axes, pulls out the arrays, draws, labels, and returns, because these get copied into notebooks and must read top to bottom.
  Module-level constants are for conventions shared across figures, such as the per-color styles that keep a color-pair looking the same everywhere.
  Symmetric pairs are worth keeping even when small — `_has_flux`/`_has_mag` in `observables.py` read better together than either would inlined.

### What counts as non-obvious

Comment it when a reader could not recover the reason from the code alone.
In practice that means:

- **Statistical or geometric choices**, such as sampling uniformly in `sin(dec)` rather than `dec`, or taking quantile bin edges from one sample and applying them to another.
- **Physics or unit conversions** that look like bare arithmetic, such as a flux error becoming a limiting magnitude.
- **Non-obvious algorithms**, such as encoding an N-dimensional histogram as a single integer index.
- **Deliberate choices that look like bugs**, such as resampling values from real objects onto randoms, or zeroing a weight instead of leaving it at one.
- **Ordering that matters**, such as a single RNG driving two draws in a fixed sequence for reproducibility.
- **Padding, clipping, and guard values**, such as widening a bounding box because HEALPix returns pixel centres.

Do not comment what the code already states.
Put the *what* in the docstring and keep inline comments for the *why*, so they stay short enough to actually read.

## Hard rules

- **Never run `git commit` without asking first.**
  Propose the message and wait for approval, every time.
  Approval for one commit is not approval for the next, and a long task does not become an exception.
  The same applies to `git push`, `git rebase`, `git reset --hard`, and anything else that rewrites history or publishes work.
  Staging with `git add` is fine; leaving the working tree ready to review is the desired end state.
- **Never commit anything under `data/`, `results/`, or `figures/`.**
  The history previously reached 19 GB this way and had to be pruned.
- Notebook outputs are stripped by `nbstripout`; do not commit notebooks with embedded images.
- `pzserver_token.txt` is a credential.
  It is gitignored and must stay that way.

## Current state

A cleanup is largely complete; see [ARCHITECTURE.md](ARCHITECTURE.md) for the known rough edges.
`treecorr_stacker.py` is still one large class at 1,448 lines, after random-catalog generation moved to `randoms.py` and the pair histograms to `pair_histograms.py`.
What remains is mostly the estimator itself, so prefer adding new concerns as their own module rather than growing this class.

Anything that consumes a finished stack goes in [src/dusty_colors/postrun/](src/dusty_colors/postrun/) as a registered analysis, never in the stacker.
Adding one is a new module with a `@register("name")` function plus one import line in `postrun/__init__.py`.
Its options live under `postrun:` in the analysis YAML, which is excluded from the stack config hash so retuning a fit does not force a re-stack.

`tests/test_stacker_characterization.py` pins all 284 stack output arrays bit-exactly.
Run it after any change to the estimator.
If it fails, the change altered the numbers; only regenerate its reference when that is intended.
