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
    footprint = _sample_footprint(selected, config.get("footprint"))

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

    foreground = clean_sample(foreground, _cleaning_for_sample(cleaning, "foreground"))
    background = clean_sample(background, _cleaning_for_sample(cleaning, "background"))
    if len(foreground) == 0 or len(background) == 0:
        raise ValueError("Foreground and background samples must both be non-empty")
    samples = {"foreground": foreground, "background": background}
    if footprint is not None:
        samples["footprint"] = footprint
    return samples


def write_sample_outputs(
    samples: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples["foreground"].to_parquet(output_dir / "foreground.parquet", index=False)
    samples["background"].to_parquet(output_dir / "background.parquet", index=False)
    if "footprint" in samples:
        samples["footprint"].to_parquet(output_dir / "footprint.parquet", index=False)


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
    if selection.get("shared_query"):
        query_mask = (
            catalog.eval(str(selection["shared_query"])).astype(bool).to_numpy()
        )
        mask &= query_mask
    pixel_depth_cuts = _enabled_option(selection, "pixel_depth_cuts")
    if pixel_depth_cuts is not None:
        mask &= _pixel_depth_cut_mask(catalog, pixel_depth_cuts, mask)
    pixel_occupancy_min = _enabled_option(selection, "pixel_occupancy_min")
    if pixel_occupancy_min is not None:
        mask &= _pixel_occupancy_mask(catalog, pixel_occupancy_min, mask)
    snr_min = _enabled_option(selection, "snr_min")
    if snr_min is not None:
        mask &= _snr_min_mask(catalog, snr_min)
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
    if "photoz_max_sigma_norm" in selection:
        max_sigma = float(selection["photoz_max_sigma_norm"])
        mask &= catalog["z_phot_err"].to_numpy(float) < max_sigma * (
            1.0 + catalog["z_phot"].to_numpy(float)
        )
    return mask


def _redshift_slice(catalog: pd.DataFrame, bounds: list[float]) -> pd.DataFrame:
    if len(bounds) != 2:
        raise ValueError("Redshift bounds must be [min, max]")
    lo, hi = float(bounds[0]), float(bounds[1])
    z_phot = catalog["z_phot"].to_numpy(float)
    return catalog.loc[(z_phot > lo) & (z_phot < hi)].reset_index(drop=True)


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


def _snr_min_mask(catalog: pd.DataFrame, config: Any) -> np.ndarray:
    if not isinstance(config, Mapping):
        raise ValueError("selection.snr_min must be a mapping")
    flux_template = str(config.get("flux_template", "flux_{band}"))
    fluxerr_template = str(config.get("fluxerr_template", "fluxerr_{band}"))
    bands = config.get("bands", config.get("values"))
    if not isinstance(bands, Mapping):
        raise ValueError("selection.snr_min.bands must map band names to thresholds")

    mask = np.ones(len(catalog), dtype=bool)
    for band, threshold in bands.items():
        band = str(band)
        flux_col = flux_template.format(band=band)
        fluxerr_col = fluxerr_template.format(band=band)
        missing = [col for col in (flux_col, fluxerr_col) if col not in catalog]
        if missing:
            raise ValueError(f"SNR cut requested missing columns: {missing}")
        flux = catalog[flux_col].to_numpy(float)
        fluxerr = catalog[fluxerr_col].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = flux / fluxerr
        mask &= snr > float(threshold)
    return mask


def _pixel_occupancy_mask(
    catalog: pd.DataFrame,
    config: Any,
    current_mask: np.ndarray,
) -> np.ndarray:
    if isinstance(config, Mapping):
        pixel_col = str(config.get("pixel_col", "pixel"))
        minimum = int(config.get("value", config.get("min", config.get("minimum", 1))))
    else:
        pixel_col = "pixel"
        minimum = int(config)
    if minimum < 1:
        raise ValueError("selection.pixel_occupancy_min must be positive")
    if pixel_col not in catalog:
        raise ValueError(f"Pixel occupancy cut requested missing column: {pixel_col}")

    pixels = catalog[pixel_col]
    counts = pixels.loc[current_mask].value_counts()
    accepted = set(counts[counts >= minimum].index.tolist())
    return pixels.isin(accepted).to_numpy()


def _pixel_depth_cut_mask(
    catalog: pd.DataFrame,
    config: Any,
    current_mask: np.ndarray,
) -> np.ndarray:
    if not isinstance(config, Mapping):
        raise ValueError("selection.pixel_depth_cuts must be a mapping")

    pixel_col = str(config.get("pixel_col", "pixel"))
    if pixel_col not in catalog:
        raise ValueError(f"Pixel depth cuts requested missing column: {pixel_col}")
    fluxerr_template = str(config.get("fluxerr_template", "fluxerr_{band}"))
    sigma = float(config.get("sigma", config.get("depth_sigma", 5.0)))
    if sigma <= 0:
        raise ValueError("selection.pixel_depth_cuts.sigma must be positive")

    depth_cache: dict[str, np.ndarray] = {}

    def depth_for(band: str, mask_for_depths: np.ndarray) -> np.ndarray:
        if band not in depth_cache:
            depth_cache[band] = _pixel_depth_values(
                catalog,
                pixel_col=pixel_col,
                fluxerr_col=fluxerr_template.format(band=band),
                sigma=sigma,
                rows_mask=mask_for_depths,
            )
        return depth_cache[band]

    mask = current_mask.copy()
    window = config.get("window", config.get("valid_range"))
    if window:
        if not isinstance(window, Mapping):
            raise ValueError("selection.pixel_depth_cuts.window must be a mapping")
        bands = [str(band) for band in window.get("bands", [])]
        if not bands:
            raise ValueError("selection.pixel_depth_cuts.window.bands is required")
        lo = window.get("min")
        hi = window.get("max")
        for band in bands:
            depth = depth_for(band, current_mask)
            if lo is not None:
                mask &= depth > float(lo)
            if hi is not None:
                mask &= depth < float(hi)

    occupancy = config.get("min_occupancy", config.get("pixel_occupancy_min"))
    if occupancy is not None:
        mask &= _pixel_occupancy_mask(
            catalog,
            {"pixel_col": pixel_col, "value": occupancy},
            mask,
        )

    min_depth = config.get("min_depth")
    if min_depth:
        if not isinstance(min_depth, Mapping):
            raise ValueError("selection.pixel_depth_cuts.min_depth must be a mapping")
        for band, threshold in min_depth.items():
            mask &= depth_for(str(band), current_mask) > float(threshold)

    quantile = config.get("quantile_min", config.get("drop_shallow_quantile"))
    if quantile:
        if not isinstance(quantile, Mapping):
            raise ValueError(
                "selection.pixel_depth_cuts.quantile_min must be a mapping"
            )
        bands = [str(band) for band in quantile.get("bands", [])]
        if not bands:
            raise ValueError(
                "selection.pixel_depth_cuts.quantile_min.bands is required"
            )
        q = float(quantile.get("value", quantile.get("quantile", 0.05)))
        if not 0.0 <= q < 1.0:
            raise ValueError(
                "selection.pixel_depth_cuts.quantile_min must be in [0, 1)"
            )
        unique = bool(quantile.get("unique", True))
        for band in bands:
            depth = depth_for(band, current_mask)
            values = depth[mask & np.isfinite(depth)]
            if unique:
                values = np.unique(values)
            if len(values) == 0:
                return np.zeros(len(catalog), dtype=bool)
            threshold = float(np.nanquantile(values, q))
            mask &= depth > threshold

    return mask


def _pixel_depth_values(
    catalog: pd.DataFrame,
    *,
    pixel_col: str,
    fluxerr_col: str,
    sigma: float,
    rows_mask: np.ndarray,
) -> np.ndarray:
    if fluxerr_col not in catalog:
        raise ValueError(f"Pixel depth cut requested missing column: {fluxerr_col}")

    fluxerr = catalog[fluxerr_col].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = MAGSYS_ZEROPOINT - 2.5 * np.log10(sigma * fluxerr)
    finite = rows_mask & np.isfinite(depth) & (fluxerr > 0)
    data = pd.DataFrame(
        {
            "pixel": catalog[pixel_col],
            "depth": depth,
        }
    )
    grouped = data.loc[finite].groupby("pixel")["depth"].median()
    return catalog[pixel_col].map(grouped).to_numpy(float)


def _sample_footprint(
    catalog: pd.DataFrame,
    config: Any,
) -> pd.DataFrame | None:
    if not config:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("Sample footprint config must be a mapping")
    if not bool(config.get("enabled", True)):
        return None
    columns = [str(col) for col in config.get("columns", [])]
    if not columns:
        columns = ["ra", "dec", "field", "pixel", "jackknife_region"]
    columns = [col for col in columns if col in catalog]
    if "pixel" not in columns:
        raise ValueError("Sample footprint output requires a 'pixel' column")
    return catalog[columns].copy().reset_index(drop=True)


def _cleaning_for_sample(
    cleaning: Mapping[str, Any],
    sample: str,
) -> Mapping[str, Any]:
    if sample not in {"foreground", "background"}:
        raise ValueError(f"Unknown sample for cleaning config: {sample}")
    if sample not in cleaning:
        return cleaning

    sample_config = cleaning.get(sample)
    if sample_config is None or sample_config is False:
        return {"enabled": False}
    if not isinstance(sample_config, Mapping):
        raise ValueError(f"cleaning.{sample} must be a mapping or false")

    base = {
        key: value
        for key, value in cleaning.items()
        if key not in {"foreground", "background"}
    }
    merged = dict(base)
    merged.update(sample_config)
    return merged
