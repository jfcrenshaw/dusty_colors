"""Sample selection for canonical catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cleaning import apply_minimal_cleaning, clean_sample


def select_samples(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    bands: list[str] | None = None,
    photometry: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Return foreground/background samples from an in-memory catalog."""
    selection = dict(config.get("selection", {}))
    cleaning = dict(config.get("cleaning", {}))

    selected = apply_minimal_cleaning(
        catalog,
        bands=bands,
        photometry=photometry,
    )
    selected = selected.loc[_base_mask(selected, selection)].reset_index(drop=True)

    foreground = _redshift_slice(selected, selection["foreground_z"])
    background = _redshift_slice(selected, selection["background_z"])

    if selection.get("foreground_query"):
        foreground = foreground.query(str(selection["foreground_query"])).reset_index(
            drop=True
        )
    if selection.get("background_query"):
        background = background.query(str(selection["background_query"])).reset_index(
            drop=True
        )

    foreground = clean_sample(foreground, cleaning)
    background = clean_sample(background, cleaning)
    if len(foreground) == 0 or len(background) == 0:
        raise ValueError("Foreground and background samples must both be non-empty")
    return {"foreground": foreground, "background": background}


def write_sample_outputs(
    samples: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples["foreground"].to_parquet(output_dir / "foreground.parquet", index=False)
    samples["background"].to_parquet(output_dir / "background.parquet", index=False)


def prepare_sample(
    catalog_path_or_dir: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    bands: list[str] | None = None,
    photometry: str | None = None,
) -> None:
    """Select and write foreground/background samples from canonical parquet."""
    path = Path(catalog_path_or_dir)
    catalog_path = path / "catalog.parquet" if path.is_dir() else path
    catalog = pd.read_parquet(catalog_path)
    samples = select_samples(
        catalog,
        config,
        bands=bands,
        photometry=photometry,
    )
    write_sample_outputs(samples, output_dir)


def _base_mask(catalog: pd.DataFrame, selection: Mapping[str, Any]) -> np.ndarray:
    mask = np.ones(len(catalog), dtype=bool)
    if "photoz_max_sigma" in selection:
        mask &= catalog["z_phot_err"].to_numpy(float) < float(
            selection["photoz_max_sigma"]
        )
    if "photoz_max_sigma_norm" in selection:
        max_sigma = float(selection["photoz_max_sigma_norm"])
        mask &= catalog["z_phot_err"].to_numpy(float) < max_sigma * (
            1.0 + catalog["z_phot"].to_numpy(float)
        )
    if selection.get("shared_query"):
        query_mask = catalog.eval(str(selection["shared_query"])).astype(bool).to_numpy()
        mask &= query_mask
    return mask


def _redshift_slice(catalog: pd.DataFrame, bounds: list[float]) -> pd.DataFrame:
    if len(bounds) != 2:
        raise ValueError("Redshift bounds must be [min, max]")
    lo, hi = float(bounds[0]), float(bounds[1])
    return catalog.query("z_phot > @lo and z_phot < @hi").reset_index(drop=True)
