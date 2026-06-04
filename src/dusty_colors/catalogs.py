"""Catalog adapters for the canonical dusty-colors schema."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .enrichments import apply_enrichments
from .footprint import (
    assign_fields,
    assign_healpix_pixels,
    assign_jackknife_regions,
    footprint_table,
)
from .sources import assemble_sources

MAGSYS_ZEROPOINT = 31.4
DEFAULT_AUXILIARY_FLUX_TYPES = ("cModel", "free_cModel")

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


def prepare_catalog(config: Mapping[str, Any], output_dir: str | Path) -> None:
    """Prepare and write a canonical catalog plus footprint table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_name = str(config["adapter"])
    try:
        adapter = ADAPTERS[adapter_name]
    except KeyError as exc:
        raise ValueError(f"Unknown catalog adapter: {adapter_name}") from exc

    source = assemble_sources(config)
    catalog = adapter(source, config)
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


def adapt_dp1_catalog(source: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Adapt an assembled DP1 source table to the canonical schema."""
    source = _with_photoz_columns(source, config)
    bands = _catalog_bands(config)
    flux_type = str(config.get("flux_type", "gaap1p0"))
    photometry = config.get("photometry")
    column_map = dict(config.get("columns", {}))

    out = pd.DataFrame()
    out["object_id"] = _mapped_or_first(
        source, column_map, "object_id", ["object_id", "objectID", "objectId"]
    )
    out["ra"] = _mapped_or_first(source, column_map, "ra", ["ra", "coord_ra", "RA"])
    out["dec"] = _mapped_or_first(
        source, column_map, "dec", ["dec", "coord_dec", "DEC"]
    )
    out["field"] = _optional_column(source, column_map, "field", ["field"], "unknown")
    out["z_phot"] = _mapped_or_first(source, column_map, "z_phot", ["z_phot"])
    out["z_phot_err"] = _optional_column(
        source,
        column_map,
        "z_phot_err",
        ["z_phot_err"],
        np.nan,
    )
    if "is_galaxy" in column_map or "is_galaxy" in source:
        out["is_galaxy"] = _mapped_or_first(
            source, column_map, "is_galaxy", ["is_galaxy"]
        ).astype(bool)
    elif "refExtendedness" in source:
        threshold = float(config.get("extendedness_min", 0.5))
        out["is_galaxy"] = source["refExtendedness"] > threshold
    else:
        out["is_galaxy"] = True
    out["mask_ok"] = _optional_column(
        source,
        column_map,
        "mask_ok",
        ["mask_ok"],
        True,
    )
    out["quality_ok"] = _optional_column(
        source,
        column_map,
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
            _copy_if_exists(source, out, f"{band}_{flux_type}Flux", f"flux_{band}")
            _copy_if_exists(
                source, out, f"{band}_{flux_type}FluxErr", f"fluxerr_{band}"
            )
        if photometry in (None, "mag"):
            _copy_if_exists(source, out, f"{band}_{flux_type}Mag", f"mag_{band}")
            _copy_if_exists(source, out, f"{band}_{flux_type}MagErr", f"magerr_{band}")
        for aux_flux_type in config.get(
            "auxiliary_flux_types",
            DEFAULT_AUXILIARY_FLUX_TYPES,
        ):
            label = _photometry_label(str(aux_flux_type))
            _copy_if_exists(
                source,
                out,
                f"{band}_{aux_flux_type}Flux",
                f"{label}_flux_{band}",
            )
            _copy_if_exists(
                source,
                out,
                f"{band}_{aux_flux_type}FluxErr",
                f"{label}_fluxerr_{band}",
            )
        _copy_if_exists(source, out, f"{band}5_pixel", f"depth5_{band}")

    return out


def dp1_adapter(source: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    return adapt_dp1_catalog(source, config)


def adapt_clauds_sextractor_catalog(
    source: pd.DataFrame, config: Mapping[str, Any]
) -> pd.DataFrame:
    """Adapt an in-memory CLAUDS SourceExtractor table to canonical columns."""
    bands = list(config.get("bands", ["g", "r", "i", "z"]))
    mag_kind = str(config.get("mag_kind", "APER_2s"))
    column_map = dict(config.get("columns", {}))

    out = pd.DataFrame()
    out["object_id"] = _mapped_or_first(source, column_map, "object_id", ["ID"])
    out["ra"] = _mapped_or_first(source, column_map, "ra", ["RA"])
    out["dec"] = _mapped_or_first(source, column_map, "dec", ["DEC"])
    out["field"] = config.get("field", "CLAUDS")
    out["z_phot"] = _mapped_or_first(source, column_map, "z_phot", ["ZPHOT"])
    out["z_phot_err"] = 0.5 * (
        source[_mapped_name(column_map, "zpdf_u68", "ZPDF_U68")]
        - source[_mapped_name(column_map, "zpdf_l68", "ZPDF_L68")]
    )
    out["is_galaxy"] = (
        _mapped_or_first(source, column_map, "obj_type", ["OBJ_TYPE"]) == 0
    )
    out["mask_ok"] = _mapped_or_first(source, column_map, "mask", ["MASK"]) == 0
    out["quality_ok"] = True
    if "Z_SPEC" in source:
        out["spec_z"] = source["Z_SPEC"]
    if "MASS_MED" in source:
        out["stellar_mass_log"] = source["MASS_MED"]

    for band in bands:
        prefix = str(config.get("band_prefix", "HSC"))
        source_band = str(config.get("band_map", {}).get(band, band))
        stem = f"{prefix}_{source_band}_MAG_{mag_kind}"
        err_stem = f"{prefix}_{source_band}_MAGERR_{mag_kind}"
        _copy_first_existing(
            source,
            out,
            [column_map.get(f"mag_{band}", stem), stem, source_band, band],
            f"mag_{band}",
        )
        _copy_first_existing(
            source,
            out,
            [column_map.get(f"magerr_{band}", err_stem), err_stem],
            f"magerr_{band}",
        )

    offset = f"OFFSET_MAG_{mag_kind.split('_')[-1]}"
    if bool(config.get("apply_aperture_offset", False)) and offset in source:
        for band in bands:
            col = f"mag_{band}"
            if col in out:
                out[col] = out[col] + source[offset]

    return out


def clauds_sextractor_adapter(
    source: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    return adapt_clauds_sextractor_catalog(source, config)


ADAPTERS: dict[str, Callable[[pd.DataFrame, Mapping[str, Any]], pd.DataFrame]] = {
    "dp1": dp1_adapter,
    "clauds_sextractor": clauds_sextractor_adapter,
}


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
    catalog = _apply_footprint_depth_cuts(catalog, footprint)
    jackknife = dict(config.get("jackknife", {}))
    if jackknife and "regions_per_field" in jackknife:
        catalog = assign_jackknife_regions(
            catalog,
            regions_per_field=int(jackknife["regions_per_field"]),
        )
    return catalog


def _apply_footprint_depth_cuts(
    catalog: pd.DataFrame,
    footprint_config: Mapping[str, Any],
) -> pd.DataFrame:
    depth_config = footprint_config.get(
        "depth_cuts",
        footprint_config.get("depth_cut"),
    )
    if not depth_config:
        return catalog
    if not isinstance(depth_config, Mapping):
        raise ValueError("footprint.depth_cuts must be a mapping")
    if not bool(depth_config.get("enabled", True)):
        return catalog
    if "pixel" not in catalog:
        raise ValueError("footprint.depth_cuts requires HEALPix 'pixel' column")

    bands = [str(band) for band in depth_config.get("bands", [])]
    if not bands:
        bands = _infer_depth_bands(
            catalog, str(depth_config.get("fluxerr_template", ""))
        )
    if not bands:
        raise ValueError("footprint.depth_cuts needs a non-empty 'bands' list")

    pixel_col = str(depth_config.get("pixel_col", "pixel"))
    if pixel_col not in catalog:
        raise ValueError(f"footprint.depth_cuts missing pixel column: {pixel_col}")

    fluxerr_template = str(
        depth_config.get("fluxerr_template", "cmodel_fluxerr_{band}")
    )
    min_depth = depth_config.get("min_depth", {})
    if min_depth is None:
        min_depth = {}
    if not isinstance(min_depth, Mapping):
        raise ValueError("footprint.depth_cuts.min_depth must be a mapping")
    shallow_fraction = float(depth_config.get("drop_shallow_fraction", 0.0))
    if not 0.0 <= shallow_fraction < 1.0:
        raise ValueError("footprint.depth_cuts.drop_shallow_fraction must be in [0, 1)")

    accepted = set(catalog[pixel_col].dropna().astype(int).unique().tolist())
    depths_by_band = {
        band: _pixel_depths(catalog, pixel_col, fluxerr_template.format(band=band))
        for band in bands
    }

    for band, threshold in min_depth.items():
        band = str(band)
        if band not in depths_by_band:
            raise ValueError(f"min_depth requested band outside depth bands: {band}")
        depths = depths_by_band[band]
        keep = depths["depth10"].to_numpy(float) >= float(threshold)
        accepted &= set(depths.loc[keep, pixel_col].astype(int).tolist())

    if shallow_fraction > 0:
        for band in bands:
            depths = depths_by_band[band]
            in_current = depths[pixel_col].astype(int).isin(accepted)
            current_depths = depths.loc[in_current, "depth10"].to_numpy(float)
            current_depths = current_depths[np.isfinite(current_depths)]
            if len(current_depths) == 0:
                accepted = set()
                break
            threshold = float(np.nanquantile(current_depths, shallow_fraction))
            keep = in_current & (depths["depth10"].to_numpy(float) >= threshold)
            accepted &= set(depths.loc[keep, pixel_col].astype(int).tolist())

    if not accepted:
        raise ValueError("footprint.depth_cuts removed every footprint pixel")

    keep_rows = catalog[pixel_col].astype(int).isin(accepted).to_numpy()
    return catalog.loc[keep_rows].reset_index(drop=True)


def _pixel_depths(
    catalog: pd.DataFrame,
    pixel_col: str,
    fluxerr_col: str,
) -> pd.DataFrame:
    if fluxerr_col not in catalog:
        raise ValueError(
            f"Depth cut requested missing flux-error column: {fluxerr_col}"
        )
    fluxerr = catalog[fluxerr_col].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth10 = MAGSYS_ZEROPOINT - 2.5 * np.log10(10.0 * fluxerr)
    depth10[~np.isfinite(depth10) | (fluxerr <= 0)] = np.nan
    data = pd.DataFrame(
        {
            pixel_col: catalog[pixel_col].to_numpy(int),
            "depth10": depth10,
        }
    )
    out = data.groupby(pixel_col, as_index=False)["depth10"].median()
    out = out[np.isfinite(out["depth10"].to_numpy(float))].reset_index(drop=True)
    return out


def _infer_depth_bands(catalog: pd.DataFrame, template: str) -> list[str]:
    if not template:
        template = "cmodel_fluxerr_{band}"
    if "{band}" not in template:
        return []
    pattern = re.escape(template).replace(r"\{band\}", r"(?P<band>[^_]+)")
    regex = re.compile(f"^{pattern}$")
    bands = []
    for col in catalog.columns:
        match = regex.match(col)
        if match:
            bands.append(match.group("band"))
    return sorted(bands)


def _with_photoz_columns(
    source: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    photoz = config.get("photoz", {})
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
        sigmas.append(_photoz_sigma(out, estimate))
        labels.append(str(estimate.get("label", _photoz_estimate_label(z_col))))

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
        out[str(diff_col)] = np.nanmax(z_values, axis=1) - np.nanmin(z_values, axis=1)
    return out


def _photoz_sigma(
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
    return 0.5 * (source[high_col].to_numpy(float) - source[low_col].to_numpy(float))


def _photoz_estimate_label(z_col: str) -> str:
    label = z_col
    for suffix in ("_z_mode", "_z_median", "_z_mean", "_z"):
        if label.endswith(suffix):
            label = label[: -len(suffix)]
            break
    label = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()
    return label or "estimate"


def _photometry_label(flux_type: str) -> str:
    label = re.sub(r"[^0-9a-zA-Z]+", "_", flux_type).strip("_").lower()
    return label or "aux"


def _catalog_bands(config: Mapping[str, Any]) -> list[str]:
    bands: list[str] = []
    for key in ("bands", "extra_bands", "optional_bands"):
        bands.extend(str(band) for band in config.get(key, []))

    enrichments = config.get("enrichments", {})
    if isinstance(enrichments, Mapping):
        kcorrect = enrichments.get("kcorrect", {})
        if isinstance(kcorrect, Mapping):
            bands.extend(str(band) for band in kcorrect.get("response_bands", []))
            bands.extend(str(band) for band in kcorrect.get("bands", []))

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


def _first_existing(source: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in source:
            return source[name]
    raise ValueError(f"None of these columns exist: {names}")


def _mapped_name(column_map: Mapping[str, str], logical_name: str, default: str) -> str:
    return str(column_map.get(logical_name, default))


def _mapped_or_first(
    source: pd.DataFrame,
    column_map: Mapping[str, str],
    logical_name: str,
    defaults: list[str],
) -> pd.Series:
    names = [_mapped_name(column_map, logical_name, defaults[0]), *defaults]
    return _first_existing(source, names)


def _optional_column(
    source: pd.DataFrame,
    column_map: Mapping[str, str],
    logical_name: str,
    defaults: list[str],
    default: Any,
) -> Any:
    try:
        return _mapped_or_first(source, column_map, logical_name, defaults)
    except ValueError:
        return default


def _copy_if_exists(
    source: pd.DataFrame, target: pd.DataFrame, source_col: str, target_col: str
) -> None:
    if source_col in source:
        target[target_col] = source[source_col]


def _copy_first_existing(
    source: pd.DataFrame,
    target: pd.DataFrame,
    source_cols: list[str],
    target_col: str,
) -> None:
    for col in source_cols:
        if col in source:
            target[target_col] = source[col]
            return


def _to_log_mass(value: pd.Series) -> pd.Series:
    mass = value.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass > 0, np.log10(mass), np.nan)
