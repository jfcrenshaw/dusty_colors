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
- Science choices belong in YAML, never in command-line arguments or hardcoded constants.
- Never count parent directories to find the repository root.
  Use `from dusty_colors import get_root`, optionally with `get_root(__file__)` when the result must not depend on the working directory.
- Never overwrite raw photometry columns; cleaning adds derived columns instead.
- Adding a survey means adding a catalog adapter, not touching the estimator.

## Hard rules

- **Never commit anything under `data/`, `results/`, or `figures/`.**
  The history previously reached 19 GB this way and had to be pruned.
- Notebook outputs are stripped by `nbstripout`; do not commit notebooks with embedded images.
- `pzserver_token.txt` is a credential.
  It is gitignored and must stay that way.

## Current state

A cleanup is in progress; see [CLEANUP_PLAN.md](CLEANUP_PLAN.md) for the phased plan and the known rough edges.
The largest is that `treecorr_stacker.py` is a single 2,458-line class due to be split.
Prefer changes that move toward that plan rather than adding to the existing god class.
