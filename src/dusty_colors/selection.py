"""Sample selection for canonical catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cleaning import apply_minimal_cleaning, clean_sample

MAGSYS_ZEROPOINT = 31.4


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
    photoz_estimate_sigma = _enabled_option(selection, "photoz_estimate_max_sigma")
    if photoz_estimate_sigma is not None:
        mask &= _photoz_estimate_sigma_mask(catalog, photoz_estimate_sigma)
    photoz_diff_norm = _enabled_option(selection, "photoz_max_diff_norm")
    if photoz_diff_norm is not None:
        mask &= _photoz_diff_norm_mask(catalog, photoz_diff_norm)
    blendedness_max = _enabled_option(selection, "blendedness_max")
    if blendedness_max is not None:
        mask &= _max_column_mask(
            catalog,
            blendedness_max,
            default_column="blendedness_i",
            option_name="blendedness_max",
        )
    magnitude_limits = _enabled_option(
        selection,
        "magnitude_limits",
        fallback_name="magnitude_limit",
    )
    if magnitude_limits is not None:
        mask &= _magnitude_limit_mask(catalog, magnitude_limits)
    if "photoz_max_sigma_norm" in selection:
        max_sigma = float(selection["photoz_max_sigma_norm"])
        mask &= catalog["z_phot_err"].to_numpy(float) < max_sigma * (
            1.0 + catalog["z_phot"].to_numpy(float)
        )
    if selection.get("shared_query"):
        query_mask = (
            catalog.eval(str(selection["shared_query"])).astype(bool).to_numpy()
        )
        mask &= query_mask
    return mask


def _redshift_slice(catalog: pd.DataFrame, bounds: list[float]) -> pd.DataFrame:
    if len(bounds) != 2:
        raise ValueError("Redshift bounds must be [min, max]")
    lo, hi = float(bounds[0]), float(bounds[1])
    return catalog.query("z_phot > @lo and z_phot < @hi").reset_index(drop=True)


def _enabled_option(
    config: Mapping[str, Any],
    name: str,
    *,
    fallback_name: str | None = None,
) -> Any:
    value = config.get(name)
    if value is None and fallback_name is not None:
        value = config.get(fallback_name)
    if value is None or value is False:
        return None
    if isinstance(value, Mapping) and not bool(value.get("enabled", True)):
        return None
    return value


def _photoz_estimate_sigma_mask(catalog: pd.DataFrame, config: Any) -> np.ndarray:
    if isinstance(config, Mapping):
        max_sigma = float(config.get("value", config.get("max", 0.1)))
        columns = [str(col) for col in config.get("columns", [])]
    else:
        max_sigma = float(config)
        columns = []
    if not columns:
        columns = sorted(
            col for col in catalog.columns if col.startswith("photoz_sigma_")
        )
    if not columns:
        raise ValueError(
            "selection.photoz_estimate_max_sigma requires photoz_sigma_* columns"
        )

    mask = np.ones(len(catalog), dtype=bool)
    for col in columns:
        if col not in catalog:
            raise ValueError(f"Photo-z sigma cut requested missing column: {col}")
        mask &= catalog[col].to_numpy(float) < max_sigma
    return mask


def _photoz_diff_norm_mask(catalog: pd.DataFrame, config: Any) -> np.ndarray:
    if isinstance(config, Mapping):
        max_norm = float(config.get("value", config.get("max", 0.1)))
        diff_col = str(config.get("column", config.get("diff_col", "z_phot_diff")))
    else:
        max_norm = float(config)
        diff_col = "z_phot_diff"
    if diff_col not in catalog:
        raise ValueError(f"Photo-z difference cut requested missing column: {diff_col}")
    return catalog[diff_col].to_numpy(float) < max_norm * (
        1.0 + catalog["z_phot"].to_numpy(float)
    )


def _max_column_mask(
    catalog: pd.DataFrame,
    config: Any,
    *,
    default_column: str,
    option_name: str,
) -> np.ndarray:
    if isinstance(config, Mapping):
        column = str(config.get("column", default_column))
        value = float(config.get("value", config.get("max")))
    else:
        column = default_column
        value = float(config)
    if column not in catalog:
        raise ValueError(f"selection.{option_name} requested missing column: {column}")
    return catalog[column].to_numpy(float) < value


def _magnitude_limit_mask(catalog: pd.DataFrame, config: Any) -> np.ndarray:
    configs = config if isinstance(config, list) else [config]
    mask = np.ones(len(catalog), dtype=bool)
    for item in configs:
        if not item or item is False:
            continue
        if not isinstance(item, Mapping):
            raise ValueError("selection.magnitude_limits entries must be mappings")
        if not bool(item.get("enabled", True)):
            continue
        mag = _magnitude_values(catalog, item)
        if "max" in item:
            mask &= mag < float(item["max"])
        if "min" in item:
            mask &= mag > float(item["min"])
    return mask


def _magnitude_values(catalog: pd.DataFrame, config: Mapping[str, Any]) -> np.ndarray:
    if "mag_col" in config:
        mag_col = str(config["mag_col"])
        if mag_col not in catalog:
            raise ValueError(f"Magnitude limit requested missing column: {mag_col}")
        return catalog[mag_col].to_numpy(float)

    if "flux_col" in config:
        flux_col = str(config["flux_col"])
    else:
        band = str(config.get("band", ""))
        if not band:
            raise ValueError(
                "Magnitude limit needs either 'band', 'mag_col', or 'flux_col'"
            )
        if f"mag_{band}" in catalog:
            return catalog[f"mag_{band}"].to_numpy(float)
        if f"cmodel_flux_{band}" in catalog:
            flux_col = f"cmodel_flux_{band}"
        else:
            flux_col = f"flux_{band}"

    if flux_col not in catalog:
        raise ValueError(f"Magnitude limit requested missing flux column: {flux_col}")
    zero_point = float(config.get("zero_point", MAGSYS_ZEROPOINT))
    flux = catalog[flux_col].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return zero_point - 2.5 * np.log10(flux)
