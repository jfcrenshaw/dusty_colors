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
Catalog preparation, sample selection, and TreeCorr stacking are described by config files under `configs/`.

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml
```

The runner skips stages whose outputs and manifests already match the resolved YAML graph.
Use operational force flags to recompute stages:

```bash
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-stack
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-sample
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-catalog
python scripts/run_stack.py configs/analyses/dp1_default.yaml --force-all
```
