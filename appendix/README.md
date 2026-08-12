# Appendix analyses

Diagnostics supporting the paper appendices: inner-bin contamination checks, blend review, and the magnification-bias estimate.

**This code is relevant but deliberately held to a lower maintenance standard than the core package.**
It still produces paper material and must keep working, but it is not refactored, linted, or type-checked alongside `src/dusty_colors/`.
Do not treat it as an example of how the rest of the codebase is written.

## What lives where

| Path | Contents |
| --- | --- |
| `src/dusty_colors/appendix/color_split_bias.py` | Contamination-proxy splits and inner-bin exclusion scenarios. |
| `src/dusty_colors/appendix/postage_stamps.py` | Projected pair tables, TAP metadata queries, blend risk scoring, contact sheets. |
| `appendix/build_dp1_inner_pair_table.py` | Builds the inner-bin pair table the notebooks consume. |
| `appendix/notebooks/` | The four appendix notebooks. |
| `tests/appendix/` | Tests for the two modules above. |

The modules stay inside the installed package so the notebooks can import them without path juggling.
The dependency direction is one-way: appendix code may import from `dusty_colors` core, never the reverse.

## Running the notebooks

Two of these fetch image cutouts and query the DP1 TAP service, so they must run **inside the Rubin Science Platform**:

- `dp1_inner_bin_postage_stamps.ipynb`
- `dp1_inner_bin_appendix_stamp_grid.ipynb`

The other two run locally against existing `results/` outputs:

- `dp1_inner_bin_color_split_bias.ipynb`
- `dp1_magnification_bias_estimate.ipynb`

Each notebook calls `get_root()` from the package, so they work from any working directory:

```python
from dusty_colors import get_root

ROOT = get_root()
```

They expect `results/stacks/dp1_default/` and `results/samples/dp1_default/` to exist already.
`dp1_magnification_bias_estimate.ipynb` uses `dp1_default_pai24` instead.

## Paper output

`dp1_inner_bin_appendix_stamp_grid.ipynb` writes `appendix_r_band_stamp_grid.pdf` into `results/postage_stamps/<stack_id>/`.
Copy it to `figures/fig_r_band_stamp_grid.pdf` by hand for the manuscript.

## Maintenance policy

- Keep the tests in `tests/appendix/` passing; they are cheap and catch import breakage.
- Excluded from `flake8` and `mypy` by configuration.
- If a core refactor breaks something here, the minimum fix is fine.
  Do not spend time improving structure, naming, or coverage.
