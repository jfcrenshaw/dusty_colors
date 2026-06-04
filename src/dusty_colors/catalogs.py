"""Catalog adapters for the canonical dusty-colors schema."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .enrichments import apply_enrichments
from .footprint import (
    assign_fields,
    assign_healpix_pixels,
    footprint_table,
)
from .sources import assemble_sources

REQUIRED_COLUMNS = {
    "object_id",
    "ra",
    "dec",
    "field",
    "z_phot",
    "z_phot_err",
    "is_galaxy",
    "mask_ok",
    "quality_ok",
}
MAGSYS_ZEROPOINT = 31.4


def prepare_catalog(config: Mapping[str, Any], output_dir: str | Path) -> None:
    """Prepare and write a canonical catalog plus footprint table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = assemble_sources(config)
    catalog = _catalog_adapter(config).adapt(source)
    catalog = apply_extinction_correction(catalog, config)
    catalog = apply_enrichments(catalog, config)
    catalog = _apply_footprint_config(catalog, config)
    validate_catalog(catalog, bands=config.get("bands", []))

    catalog.to_parquet(output_dir / "catalog.parquet", index=False)
    footprint_table(catalog).to_parquet(output_dir / "footprint.parquet", index=False)


def validate_catalog(catalog: pd.DataFrame, *, bands: list[str]) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(catalog.columns))
    if missing:
        raise ValueError(f"Canonical catalog missing required columns: {missing}")
    for band in bands:
        has_flux = {f"flux_{band}", f"fluxerr_{band}"}.issubset(catalog.columns)
        has_mag = {f"mag_{band}", f"magerr_{band}"}.issubset(catalog.columns)
        if not has_flux and not has_mag:
            raise ValueError(f"Catalog lacks flux or magnitude photometry for {band}")


def validate_canonical_schema(
    catalog: pd.DataFrame,
    *,
    bands: list[str] | None = None,
    photometry: str | None = None,
) -> None:
    """Validate canonical columns, optionally requiring one photometry family."""
    missing = sorted(REQUIRED_COLUMNS - set(catalog.columns))
    if missing:
        raise ValueError(f"Canonical catalog missing required columns: {missing}")
    for band in bands or []:
        if photometry == "flux":
            required = {f"flux_{band}", f"fluxerr_{band}"}
        elif photometry == "mag":
            required = {f"mag_{band}", f"magerr_{band}"}
        else:
            required = set()
        if required:
            band_missing = sorted(required - set(catalog.columns))
            if band_missing:
                raise ValueError(f"Catalog missing {band} photometry: {band_missing}")
    if bands and photometry is None:
        validate_catalog(catalog, bands=bands)


class CatalogAdapter(ABC):
    """Base class for one raw catalog format.

    Subclasses turn survey-specific columns into the canonical dusty-colors
    schema. The shared helpers keep the adapter code focused on the astronomy:
    positions, redshifts, object flags, and photometry.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.column_map = dict(config.get("columns", {}))

    @abstractmethod
    def adapt(self, source: pd.DataFrame) -> pd.DataFrame:
        """Return a canonical catalog from an assembled source table."""

    def mapped_or_first(
        self,
        source: pd.DataFrame,
        logical_name: str,
        defaults: list[str],
    ) -> pd.Series:
        names = [self.mapped_name(logical_name, defaults[0]), *defaults]
        return self.first_existing(source, names)

    def optional_column(
        self,
        source: pd.DataFrame,
        logical_name: str,
        defaults: list[str],
        default: Any,
    ) -> Any:
        try:
            return self.mapped_or_first(source, logical_name, defaults)
        except ValueError:
            return default

    def mapped_name(self, logical_name: str, default: str) -> str:
        return str(self.column_map.get(logical_name, default))

    @staticmethod
    def first_existing(source: pd.DataFrame, names: list[str]) -> pd.Series:
        for name in names:
            if name in source:
                return source[name]
        raise ValueError(f"None of these columns exist: {names}")

    @staticmethod
    def copy_if_exists(
        source: pd.DataFrame,
        target: pd.DataFrame,
        source_col: str,
        target_col: str,
    ) -> None:
        if source_col in source:
            target[target_col] = source[source_col]

    @staticmethod
    def copy_first_existing(
        source: pd.DataFrame,
        target: pd.DataFrame,
        source_cols: list[str],
        target_col: str,
    ) -> None:
        for col in source_cols:
            if col in source:
                target[target_col] = source[col]
                return

    @staticmethod
    def ab_magnitude_to_nanjy(magnitude: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore", invalid="ignore"):
            return 10 ** ((MAGSYS_ZEROPOINT - magnitude) / 2.5)

    @staticmethod
    def magerr_to_nanjy_error(
        flux: np.ndarray,
        magerr: np.ndarray,
    ) -> np.ndarray:
        with np.errstate(invalid="ignore"):
            return np.abs(flux) * np.log(10.0) / 2.5 * magerr

    def with_photoz_columns(self, source: pd.DataFrame) -> pd.DataFrame:
        photoz = self.config.get("photoz", {})
        if not photoz:
            return source
        if not isinstance(photoz, Mapping):
            raise ValueError("Catalog 'photoz' must be a mapping")

        combine = photoz.get("combine", photoz)
        if not isinstance(combine, Mapping) or "estimates" not in combine:
            return source

        estimates = list(combine["estimates"])
        if len(estimates) == 0:
            raise ValueError("photoz.combine.estimates must not be empty")

        out = source.copy()
        values = []
        sigmas = []
        labels = []
        for estimate in estimates:
            if not isinstance(estimate, Mapping):
                raise ValueError("Each photoz estimate must be a mapping")
            z_col = str(estimate["z"])
            if z_col not in out:
                raise ValueError(f"Photo-z estimate column is missing: {z_col}")
            values.append(out[z_col].to_numpy(float))
            sigmas.append(self.photoz_sigma(out, estimate))
            labels.append(str(estimate.get("label", self.photoz_label(z_col))))

        z_values = np.column_stack(values)
        z_sigmas = np.column_stack(sigmas)
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(z_sigmas > 0, 1.0 / z_sigmas**2, 0.0)
            weight_sum = np.sum(weights, axis=1)
            z_phot = np.sum(z_values * weights, axis=1) / weight_sum
            z_err = np.sqrt(1.0 / weight_sum)
        bad = (weight_sum <= 0) | ~np.isfinite(z_phot) | ~np.isfinite(z_err)
        z_phot[bad] = np.nan
        z_err[bad] = np.nan

        out[str(combine.get("z_col", "z_phot"))] = z_phot
        out[str(combine.get("err_col", "z_phot_err"))] = z_err
        for label, value, sigma in zip(labels, values, sigmas):
            out[f"photoz_{label}"] = value
            out[f"photoz_sigma_{label}"] = sigma
        diff_col = combine.get("diff_col", "z_phot_diff")
        if diff_col:
            out[str(diff_col)] = np.nanmax(z_values, axis=1) - np.nanmin(
                z_values, axis=1
            )
        return out

    @staticmethod
    def photoz_sigma(
        source: pd.DataFrame,
        estimate: Mapping[str, Any],
    ) -> np.ndarray:
        if "err" in estimate:
            err_col = str(estimate["err"])
            if err_col not in source:
                raise ValueError(f"Photo-z error column is missing: {err_col}")
            return source[err_col].to_numpy(float)
        low_col = str(estimate["err_low"])
        high_col = str(estimate["err_high"])
        missing = [col for col in (low_col, high_col) if col not in source]
        if missing:
            raise ValueError(f"Photo-z interval columns are missing: {missing}")
        return 0.5 * (
            source[high_col].to_numpy(float) - source[low_col].to_numpy(float)
        )

    @staticmethod
    def photoz_label(z_col: str) -> str:
        label = z_col
        for suffix in ("_z_mode", "_z_median", "_z_mean", "_z"):
            if label.endswith(suffix):
                label = label[: -len(suffix)]
                break
        label = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()
        return label or "estimate"


class DP1CatalogAdapter(CatalogAdapter):
    """Adapt assembled Rubin DP1 tables into the canonical schema."""

    auxiliary_flux_types = ("cModel",)

    def adapt(self, source: pd.DataFrame) -> pd.DataFrame:
        source = self.with_photoz_columns(source)
        bands = _catalog_bands(self.config)
        flux_type = str(self.config.get("flux_type", "gaap1p0"))
        photometry = self.config.get("photometry")

        out = pd.DataFrame()
        out["object_id"] = self.mapped_or_first(
            source, "object_id", ["object_id", "objectID", "objectId"]
        )
        out["ra"] = self.mapped_or_first(source, "ra", ["ra", "coord_ra", "RA"])
        out["dec"] = self.mapped_or_first(source, "dec", ["dec", "coord_dec", "DEC"])
        out["field"] = self.optional_column(source, "field", ["field"], "unknown")
        out["z_phot"] = self.mapped_or_first(source, "z_phot", ["z_phot"])
        out["z_phot_err"] = self.optional_column(
            source,
            "z_phot_err",
            ["z_phot_err"],
            np.nan,
        )
        if "is_galaxy" in self.column_map or "is_galaxy" in source:
            out["is_galaxy"] = self.mapped_or_first(
                source, "is_galaxy", ["is_galaxy"]
            ).astype(bool)
        elif "refExtendedness" in source:
            threshold = float(self.config.get("extendedness_min", 0.5))
            out["is_galaxy"] = source["refExtendedness"] > threshold
        else:
            out["is_galaxy"] = True
        out["mask_ok"] = self.optional_column(
            source,
            "mask_ok",
            ["mask_ok"],
            True,
        )
        out["quality_ok"] = self.optional_column(
            source,
            "quality_ok",
            ["quality_ok"],
            True,
        )
        if "redshift" in source:
            out["spec_z"] = source["redshift"]
        if "confidence" in source:
            out["spec_z_confidence"] = source["confidence"]
        if "ebv" in source:
            out["ebv"] = source["ebv"]
        if "refExtendedness" in source:
            out["ref_extendedness"] = source["refExtendedness"]
        for col in source.columns:
            if col.endswith("_blendedness"):
                band = col.removesuffix("_blendedness")
                out[f"blendedness_{band}"] = source[col]
            if col == "z_phot_diff" or col.startswith("photoz_"):
                out[col] = source[col]
        if "stellar_mass_log" in source:
            out["stellar_mass_log"] = source["stellar_mass_log"]
        if "stellar_mass" in source:
            out["stellar_mass_log"] = _to_log_mass(source["stellar_mass"])

        for band in bands:
            if photometry in (None, "flux"):
                self.copy_if_exists(
                    source, out, f"{band}_{flux_type}Flux", f"flux_{band}"
                )
                self.copy_if_exists(
                    source, out, f"{band}_{flux_type}FluxErr", f"fluxerr_{band}"
                )
            if photometry in (None, "mag"):
                self.copy_if_exists(
                    source, out, f"{band}_{flux_type}Mag", f"mag_{band}"
                )
                self.copy_if_exists(
                    source, out, f"{band}_{flux_type}MagErr", f"magerr_{band}"
                )
            for aux_flux_type in self.config.get(
                "auxiliary_flux_types",
                self.auxiliary_flux_types,
            ):
                label = self.photometry_label(str(aux_flux_type))
                self.copy_if_exists(
                    source,
                    out,
                    f"{band}_{aux_flux_type}Flux",
                    f"{label}_flux_{band}",
                )
                self.copy_if_exists(
                    source,
                    out,
                    f"{band}_{aux_flux_type}FluxErr",
                    f"{label}_fluxerr_{band}",
                )
            self.copy_if_exists(source, out, f"{band}5_pixel", f"depth5_{band}")

        return out

    @staticmethod
    def photometry_label(flux_type: str) -> str:
        label = re.sub(r"[^0-9a-zA-Z]+", "_", flux_type).strip("_").lower()
        return label or "aux"


class ClaudsSExtractorCatalogAdapter(CatalogAdapter):
    """Adapt Picouet CLAUDS-HSC SourceExtractor catalogs."""

    required_columns = (
        "ID",
        "RA",
        "DEC",
        "field",
        "ZPHOT",
        "Z_BEST68_LOW",
        "Z_BEST68_HIGH",
        "OBJ_TYPE",
        "CLEAN",
        "EB_V",
        "Z_SPEC",
        "OFFSET_MAG_2s",
    )

    def adapt(self, source: pd.DataFrame) -> pd.DataFrame:
        self.require_columns(source, self.required_columns)
        mag_kind = str(self.config.get("mag_kind", "APER_2s"))
        if mag_kind != "APER_2s":
            raise ValueError("CLAUDS SourceExtractor adapter only supports APER_2s")
        bands = _catalog_bands(self.config)
        if not bands:
            raise ValueError("CLAUDS SourceExtractor config must define bands")

        out = pd.DataFrame()
        out["object_id"] = source["ID"]
        out["ra"] = source["RA"]
        out["dec"] = source["DEC"]
        out["field"] = source["field"]
        out["z_phot"] = source["ZPHOT"]
        out["z_phot_err"] = 0.5 * (
            source["Z_BEST68_HIGH"].to_numpy(float)
            - source["Z_BEST68_LOW"].to_numpy(float)
        )
        out["is_galaxy"] = source["OBJ_TYPE"] == 0
        out["mask_ok"] = source["CLEAN"].astype(bool)
        out["quality_ok"] = True
        out["spec_z"] = source["Z_SPEC"]
        out["ebv"] = source["EB_V"]
        out["stellar_mass_log"] = self.stellar_mass_log(source)

        offset = source["OFFSET_MAG_2s"].to_numpy(float)
        for band in bands:
            values, errors = self.picouet_photometry(source, band)
            magnitudes = values.copy()
            finite = np.isfinite(magnitudes)
            magnitudes[finite] = magnitudes[finite] + offset[finite]
            flux = self.ab_magnitude_to_nanjy(magnitudes)

            out[f"mag_{band}"] = magnitudes
            out[f"magerr_{band}"] = errors
            out[f"flux_{band}"] = flux
            out[f"fluxerr_{band}"] = self.magerr_to_nanjy_error(flux, errors)

        return out

    @staticmethod
    def require_columns(source: pd.DataFrame, columns: tuple[str, ...]) -> None:
        missing = sorted(set(columns) - set(source.columns))
        if missing:
            raise ValueError(f"CLAUDS SourceExtractor columns are missing: {missing}")

    @staticmethod
    def stellar_mass_log(source: pd.DataFrame) -> np.ndarray:
        if "MASS_MED" not in source and "MASS_MED_6B" not in source:
            raise ValueError(
                "CLAUDS SourceExtractor columns are missing: "
                "['MASS_MED' or 'MASS_MED_6B']"
            )
        values = np.full(len(source), np.nan, dtype=float)
        if "MASS_MED" in source:
            values = source["MASS_MED"].to_numpy(float)
        if "MASS_MED_6B" in source:
            fallback = source["MASS_MED_6B"].to_numpy(float)
            use = ~np.isfinite(values) & np.isfinite(fallback)
            values = values.copy()
            values[use] = fallback[use]
        return values

    def picouet_photometry(
        self,
        source: pd.DataFrame,
        band: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        required = []
        for source_band in self.source_bands_for(band):
            required.extend(
                [f"MAG_APER_2s_{source_band}", f"MAGERR_APER_2s_{source_band}"]
            )
        self.require_columns(source, tuple(required))

        values = np.full(len(source), np.nan, dtype=float)
        errors = np.full(len(source), np.nan, dtype=float)
        for source_band in self.source_bands_for(band):
            mag = source[f"MAG_APER_2s_{source_band}"].to_numpy(float)
            err = source[f"MAGERR_APER_2s_{source_band}"].to_numpy(float)
            use = ~np.isfinite(values) & np.isfinite(mag) & np.isfinite(err)
            values[use] = mag[use]
            errors[use] = err[use]
        return values, errors

    @staticmethod
    def source_bands_for(band: str) -> list[str]:
        if band == "u":
            return ["u", "uS"]
        return [band]


ADAPTER_CLASSES: dict[str, type[CatalogAdapter]] = {
    "dp1": DP1CatalogAdapter,
    "clauds_sextractor": ClaudsSExtractorCatalogAdapter,
}


def _catalog_adapter(config: Mapping[str, Any]) -> CatalogAdapter:
    adapter_name = str(config["adapter"])
    try:
        adapter_cls = ADAPTER_CLASSES[adapter_name]
    except KeyError as exc:
        raise ValueError(f"Unknown catalog adapter: {adapter_name}") from exc
    return adapter_cls(config)


def apply_extinction_correction(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply Galactic-extinction corrections to canonical photometry."""
    extinction = config.get("extinction", {})
    if not extinction or not bool(extinction.get("enabled", True)):
        return catalog
    if not isinstance(extinction, Mapping):
        raise ValueError("Catalog 'extinction' must be a mapping")

    coefficients = dict(extinction.get("coefficients", {}))
    if not coefficients:
        raise ValueError("Extinction correction requires coefficients by band")
    ebv_col = str(extinction.get("ebv_column", "ebv"))
    if ebv_col not in catalog:
        raise ValueError(f"Extinction EBV column is missing: {ebv_col}")

    out = catalog.copy()
    bands = list(extinction.get("bands", _catalog_bands(config)))
    ebv = out[ebv_col].to_numpy(float)
    for band in bands:
        if band not in coefficients:
            raise ValueError(f"Extinction coefficient missing for band: {band}")
        a_band = float(coefficients[band])
        flux_factor = 10 ** (0.4 * a_band * ebv)
        flux_col = f"flux_{band}"
        fluxerr_col = f"fluxerr_{band}"
        if flux_col in out:
            out[flux_col] = out[flux_col].to_numpy(float) * flux_factor
        if fluxerr_col in out:
            out[fluxerr_col] = out[fluxerr_col].to_numpy(float) * flux_factor
        mag_col = f"mag_{band}"
        if mag_col in out:
            out[mag_col] = out[mag_col].to_numpy(float) - a_band * ebv
    return out


def _apply_footprint_config(
    catalog: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    footprint = dict(config.get("footprint", {}))
    fields = footprint.get("fields")
    if fields:
        catalog = assign_fields(
            catalog,
            fields,
            radius_deg=float(footprint.get("field_radius_deg", 2.0)),
        )
    if "nside" in footprint:
        catalog = assign_healpix_pixels(catalog, nside=int(footprint["nside"]))
    return catalog


def _catalog_bands(config: Mapping[str, Any]) -> list[str]:
    bands: list[str] = []
    for key in ("bands", "extra_bands", "optional_bands"):
        bands.extend(str(band) for band in config.get(key, []))

    enrichments = config.get("enrichments", {})
    if isinstance(enrichments, Mapping):
        kcorrect = enrichments.get("kcorrect", {})
        if isinstance(kcorrect, Mapping):
            bands.extend(str(band) for band in kcorrect.get("response_bands", []))

    extinction = config.get("extinction", {})
    if isinstance(extinction, Mapping):
        bands.extend(str(band) for band in extinction.get("bands", []))

    unique = []
    seen = set()
    for band in bands:
        if band not in seen:
            unique.append(band)
            seen.add(band)
    return unique


def _to_log_mass(value: pd.Series) -> pd.Series:
    mass = value.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass > 0, np.log10(mass), np.nan)
