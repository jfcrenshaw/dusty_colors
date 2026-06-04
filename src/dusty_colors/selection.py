"""Sample selection for canonical catalogs."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cleaning import apply_minimal_cleaning, clean_sample
from .footprint import assign_jackknife_regions

MAGSYS_ZEROPOINT = 31.4


class SampleSelector:
    """Select foreground/background samples from one canonical catalog.

    The order mirrors the analysis logic in the sample YAML: start from valid
    catalog rows, apply cuts shared by both samples, then make foreground and
    background samples with their own optional cleaning.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        config: Mapping[str, Any],
        *,
        bands: list[str] | None = None,
        photometry: str | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.selection = dict(config.get("selection", {}))
        self.cleaning = dict(config.get("cleaning", {}))
        self.bands = bands
        self.photometry = photometry

    def select(self) -> dict[str, pd.DataFrame]:
        selected = self._valid_catalog_rows()
        selected = self._apply_shared_selection(selected)
        selected = self._assign_jackknife(selected)
        footprint = self._footprint(selected)

        foreground = self._sample(selected, "foreground")
        background = self._sample(selected, "background")
        if len(foreground) == 0 or len(background) == 0:
            raise ValueError("Foreground and background samples must both be non-empty")

        samples = {"foreground": foreground, "background": background}
        if footprint is not None:
            samples["footprint"] = footprint
        return samples

    def _valid_catalog_rows(self) -> pd.DataFrame:
        return apply_minimal_cleaning(
            self.catalog,
            bands=self.bands,
            photometry=self.photometry,
        )

    def _apply_shared_selection(self, catalog: pd.DataFrame) -> pd.DataFrame:
        mask = SharedSelectionCuts(catalog, self.selection).mask()
        return catalog.loc[mask].reset_index(drop=True)

    def _assign_jackknife(self, catalog: pd.DataFrame) -> pd.DataFrame:
        jackknife = _enabled_option(self.config, "jackknife")
        if jackknife is None:
            return catalog
        return _assign_sample_jackknife(catalog, jackknife)

    def _footprint(self, catalog: pd.DataFrame) -> pd.DataFrame | None:
        return _sample_footprint(catalog, self.config.get("footprint"))

    def _sample(self, catalog: pd.DataFrame, sample: str) -> pd.DataFrame:
        sample_catalog = _redshift_slice(catalog, self.selection[f"{sample}_z"])
        query = self.selection.get(f"{sample}_query")
        if query:
            sample_catalog = sample_catalog.query(str(query)).reset_index(drop=True)
        return clean_sample(
            sample_catalog,
            _cleaning_for_sample(self.cleaning, sample),
        )


def select_samples(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    bands: list[str] | None = None,
    photometry: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Return foreground/background samples from an in-memory catalog."""
    return SampleSelector(
        catalog,
        config,
        bands=bands,
        photometry=photometry,
    ).select()


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


class SharedSelectionCuts:
    """Build the mask for cuts shared by foreground and background samples."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        selection: Mapping[str, Any],
    ) -> None:
        self.catalog = catalog
        self.selection = selection
        self.current_mask = np.ones(len(catalog), dtype=bool)

    def mask(self) -> np.ndarray:
        self._apply_shared_query()
        self._apply_pixel_cuts()
        self._apply_photometry_cuts()
        self._apply_photoz_cuts()
        return self.current_mask

    def _apply_shared_query(self) -> None:
        if not self.selection.get("shared_query"):
            return
        query_mask = (
            self.catalog.eval(str(self.selection["shared_query"]))
            .astype(bool)
            .to_numpy()
        )
        self.current_mask &= query_mask

    def _apply_pixel_cuts(self) -> None:
        pixel_depth_cuts = _enabled_option(self.selection, "pixel_depth_cuts")
        if pixel_depth_cuts is not None:
            self.current_mask &= _pixel_depth_cut_mask(
                self.catalog,
                pixel_depth_cuts,
                self.current_mask,
            )

        pixel_occupancy_min = _enabled_option(
            self.selection,
            "pixel_occupancy_min",
        )
        if pixel_occupancy_min is not None:
            self.current_mask &= _pixel_occupancy_mask(
                self.catalog,
                pixel_occupancy_min,
                self.current_mask,
            )

    def _apply_photometry_cuts(self) -> None:
        snr_min = _enabled_option(self.selection, "snr_min")
        if snr_min is not None:
            self.current_mask &= _snr_min_mask(self.catalog, snr_min)

        blendedness_max = _enabled_option(self.selection, "blendedness_max")
        if blendedness_max is not None:
            self.current_mask &= _max_column_mask(
                self.catalog,
                blendedness_max,
                default_column="blendedness_i",
                option_name="blendedness_max",
            )

        magnitude_limits = _enabled_option(self.selection, "magnitude_limits")
        if magnitude_limits is not None:
            self.current_mask &= _magnitude_limit_mask(
                self.catalog,
                magnitude_limits,
            )

    def _apply_photoz_cuts(self) -> None:
        if "photoz_max_sigma" in self.selection:
            self.current_mask &= self.catalog["z_phot_err"].to_numpy(float) < float(
                self.selection["photoz_max_sigma"]
            )

        photoz_estimate_sigma = _enabled_option(
            self.selection,
            "photoz_estimate_max_sigma",
        )
        if photoz_estimate_sigma is not None:
            self.current_mask &= _photoz_estimate_sigma_mask(
                self.catalog,
                photoz_estimate_sigma,
            )

        photoz_diff_norm = _enabled_option(self.selection, "photoz_max_diff_norm")
        if photoz_diff_norm is not None:
            self.current_mask &= _photoz_diff_norm_mask(
                self.catalog,
                photoz_diff_norm,
            )

        if "photoz_max_sigma_norm" in self.selection:
            max_sigma = float(self.selection["photoz_max_sigma_norm"])
            self.current_mask &= self.catalog["z_phot_err"].to_numpy(
                float
            ) < max_sigma * (1.0 + self.catalog["z_phot"].to_numpy(float))


def _redshift_slice(catalog: pd.DataFrame, bounds: list[float]) -> pd.DataFrame:
    if len(bounds) != 2:
        raise ValueError("Redshift bounds must be [min, max]")
    lo, hi = float(bounds[0]), float(bounds[1])
    z_phot = catalog["z_phot"].to_numpy(float)
    return catalog.loc[(z_phot > lo) & (z_phot < hi)].reset_index(drop=True)


def _enabled_option(
    config: Mapping[str, Any],
    name: str,
) -> Any:
    value = config.get(name)
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
    bands = config.get("bands")
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


class PixelDepthCuts:
    """Apply depth cuts that are measured per HEALPix footprint pixel."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        config: Mapping[str, Any],
        current_mask: np.ndarray,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.current_mask = current_mask
        self.pixel_col = str(config.get("pixel_col", "pixel"))
        if self.pixel_col not in catalog:
            raise ValueError(
                f"Pixel depth cuts requested missing column: {self.pixel_col}"
            )
        self.fluxerr_template = str(config.get("fluxerr_template", "fluxerr_{band}"))
        self.depth_sigma = float(config.get("depth_sigma", 10.0))
        if self.depth_sigma <= 0:
            raise ValueError("selection.pixel_depth_cuts.depth_sigma must be positive")
        aggregate = str(config.get("aggregate", "median"))
        if aggregate != "median":
            raise ValueError("selection.pixel_depth_cuts.aggregate must be 'median'")
        self._depth_cache: dict[str, np.ndarray] = {}

    def mask(self) -> np.ndarray:
        mask = self.current_mask.copy()
        mask = self._apply_valid_range(mask)
        mask = self._apply_min_occupancy(mask)
        mask = self._apply_completeness(mask)
        mask = self._drop_shallowest_pixels(mask)
        return mask

    def depth_for(self, band: str) -> np.ndarray:
        if band not in self._depth_cache:
            self._depth_cache[band] = _pixel_depth_values(
                self.catalog,
                pixel_col=self.pixel_col,
                fluxerr_col=self.fluxerr_template.format(band=band),
                depth_sigma=self.depth_sigma,
                rows_mask=self.current_mask,
            )
        return self._depth_cache[band]

    def _apply_valid_range(self, mask: np.ndarray) -> np.ndarray:
        valid_range = self.config.get("valid_range")
        if not valid_range:
            return mask
        if not isinstance(valid_range, Mapping):
            raise ValueError("selection.pixel_depth_cuts.valid_range must be a mapping")
        bands = [str(band) for band in valid_range.get("bands", [])]
        if not bands:
            raise ValueError("selection.pixel_depth_cuts.valid_range.bands is required")
        lo = valid_range.get("min")
        hi = valid_range.get("max")
        for band in bands:
            depth = self.depth_for(band)
            if lo is not None:
                mask &= depth > float(lo)
            if hi is not None:
                mask &= depth < float(hi)
        return mask

    def _apply_min_occupancy(self, mask: np.ndarray) -> np.ndarray:
        occupancy = self.config.get("min_occupancy")
        if occupancy is None:
            return mask
        return mask & _pixel_occupancy_mask(
            self.catalog,
            {"pixel_col": self.pixel_col, "value": occupancy},
            mask,
        )

    def _apply_completeness(self, mask: np.ndarray) -> np.ndarray:
        complete_to = self.config.get("complete_to")
        if not complete_to:
            return mask
        complete_configs = (
            complete_to if isinstance(complete_to, list) else [complete_to]
        )
        for item in complete_configs:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "pixel_depth_cuts.complete_to entries must be mappings"
                )
            band = str(item.get("band", ""))
            if not band:
                raise ValueError("pixel_depth_cuts.complete_to.band is required")
            if "magnitude" not in item:
                raise ValueError("pixel_depth_cuts.complete_to.magnitude is required")
            mask &= self.depth_for(band) > float(item["magnitude"])
        return mask

    def _drop_shallowest_pixels(self, mask: np.ndarray) -> np.ndarray:
        drop_shallowest = self.config.get("drop_shallowest")
        if not drop_shallowest:
            return mask
        if not isinstance(drop_shallowest, Mapping):
            raise ValueError(
                "selection.pixel_depth_cuts.drop_shallowest must be a mapping"
            )
        bands = [str(band) for band in drop_shallowest.get("bands", [])]
        if not bands:
            raise ValueError(
                "selection.pixel_depth_cuts.drop_shallowest.bands is required"
            )
        q = float(drop_shallowest.get("fraction", 0.05))
        if not 0.0 <= q < 1.0:
            raise ValueError(
                "selection.pixel_depth_cuts.drop_shallowest.fraction must be in [0, 1)"
            )
        unique = bool(drop_shallowest.get("unique_pixels", True))
        for band in bands:
            depth = self.depth_for(band)
            values = depth[mask & np.isfinite(depth)]
            if unique:
                values = np.unique(values)
            if len(values) == 0:
                return np.zeros(len(self.catalog), dtype=bool)
            threshold = float(np.nanquantile(values, q))
            mask &= depth > threshold
        return mask


def _pixel_depth_cut_mask(
    catalog: pd.DataFrame,
    config: Any,
    current_mask: np.ndarray,
) -> np.ndarray:
    if not isinstance(config, Mapping):
        raise ValueError("selection.pixel_depth_cuts must be a mapping")
    return PixelDepthCuts(catalog, config, current_mask).mask()


def _pixel_depth_values(
    catalog: pd.DataFrame,
    *,
    pixel_col: str,
    fluxerr_col: str,
    depth_sigma: float,
    rows_mask: np.ndarray,
) -> np.ndarray:
    if fluxerr_col not in catalog:
        raise ValueError(f"Pixel depth cut requested missing column: {fluxerr_col}")

    fluxerr = catalog[fluxerr_col].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = MAGSYS_ZEROPOINT - 2.5 * np.log10(depth_sigma * fluxerr)
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


def _assign_sample_jackknife(catalog: pd.DataFrame, config: Any) -> pd.DataFrame:
    if not isinstance(config, Mapping):
        raise ValueError("Sample jackknife config must be a mapping")
    regions_per_field = config.get("regions_per_field")
    if regions_per_field is None:
        return catalog
    return assign_jackknife_regions(
        catalog,
        regions_per_field=int(regions_per_field),
        field_col=str(config.get("field_col", "field")),
        output_col=str(config.get("output_col", "jackknife_region")),
    )


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
