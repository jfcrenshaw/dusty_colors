"""Optional catalog-stage physical-property enrichments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import re

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
MAGSYS_ZEROPOINT = 31.4


def apply_enrichments(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Apply enabled catalog enrichments from YAML."""
    out = catalog.copy()
    enrichments = config.get("enrichments", {})
    if not enrichments:
        return out
    if not isinstance(enrichments, Mapping):
        raise ValueError("Catalog 'enrichments' must be a mapping")

    for name, enrichment_config in enrichments.items():
        if enrichment_config is None:
            continue
        if not isinstance(enrichment_config, Mapping):
            raise ValueError(f"Enrichment '{name}' must be a mapping")
        if not bool(enrichment_config.get("enabled", True)):
            continue
        if name == "kcorrect":
            out = apply_kcorrect(out, enrichment_config)
        elif name == "halo_mass":
            out = apply_halo_mass(out, enrichment_config)
        else:
            raise ValueError(f"Unknown catalog enrichment: {name}")
    return out


def apply_kcorrect(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Fit kcorrect templates and append rest-frame quantities."""
    return KcorrectEnrichment(config).apply(catalog)


class KcorrectEnrichment:
    """Fit kcorrect templates and add rest-frame quantities to a catalog."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    def apply(self, catalog: pd.DataFrame) -> pd.DataFrame:
        model = self.build_model()
        response_bands = self.response_bands(model)
        redshift_col = str(self.config.get("redshift_col", "z_phot"))
        if redshift_col not in catalog:
            raise ValueError(f"kcorrect redshift column is missing: {redshift_col}")

        out = catalog.copy()
        redshift = out[redshift_col].to_numpy(float)
        maggies = self.catalog_maggies(out, response_bands)
        errors = self.catalog_maggie_errors(out, response_bands)
        errors = self.apply_error_floor(errors, maggies, response_bands)

        absmag_bands = list(self.config.get("absmag_bands", response_bands))
        stellar_mass_col = str(self.config.get("stellar_mass_col", "stellar_mass_log"))
        linear_mass_col = self.config.get("linear_stellar_mass_col")
        self.add_empty_outputs(out, absmag_bands, stellar_mass_col, linear_mass_col)

        good = self.valid_rows(redshift, maggies, errors)
        if not np.any(good):
            return out

        self.fit_and_write_outputs(
            out,
            model,
            redshift=redshift,
            maggies=maggies,
            errors=errors,
            good=good,
            response_bands=response_bands,
            absmag_bands=absmag_bands,
            stellar_mass_col=stellar_mass_col,
            linear_mass_col=linear_mass_col,
        )
        return out

    def build_model(self) -> Any:
        Kcorrect = self.kcorrect_class()
        model = self.config.get("model")
        if model is not None:
            return Kcorrect(filename=str(self.resolve_path(model)))

        responses = self.config.get("responses")
        if not responses:
            raise ValueError(
                "kcorrect enrichment requires either 'model' or 'responses'"
            )

        kwargs: dict[str, Any] = {
            "responses": self.resolve_response_names(responses),
        }
        if "responses_out" in self.config:
            kwargs["responses_out"] = self.resolve_response_names(
                self.config["responses_out"]
            )
        if "responses_map" in self.config:
            kwargs["responses_map"] = self.resolve_response_names(
                self.config["responses_map"]
            )
        if "redshift_range" in self.config:
            kwargs["redshift_range"] = list(self.config["redshift_range"])
        if "nredshift" in self.config:
            kwargs["nredshift"] = int(self.config["nredshift"])
        if "abcorrect" in self.config:
            kwargs["abcorrect"] = bool(self.config["abcorrect"])
        if "interpolate_templates" in self.config:
            kwargs["interpolate_templates"] = bool(self.config["interpolate_templates"])
        return Kcorrect(**kwargs)

    def response_bands(self, model: Any) -> list[str]:
        bands = list(
            self.config.get("response_bands") or self.infer_response_bands(model)
        )
        if not bands:
            raise ValueError(
                "kcorrect enrichment needs response_bands or inferrable response names"
            )
        return bands

    def catalog_maggies(
        self,
        catalog: pd.DataFrame,
        bands: Sequence[str],
    ) -> np.ndarray:
        values = []
        for band in bands:
            col = f"flux_{band}"
            if col not in catalog:
                raise ValueError(f"kcorrect requires canonical flux column: {col}")
            values.append(self.nanjy_to_maggies(catalog[col].to_numpy(float)))
        return np.column_stack(values)

    def catalog_maggie_errors(
        self,
        catalog: pd.DataFrame,
        bands: Sequence[str],
    ) -> np.ndarray:
        values = []
        for band in bands:
            col = f"fluxerr_{band}"
            if col not in catalog:
                raise ValueError(
                    f"kcorrect requires canonical flux error column: {col}"
                )
            values.append(self.nanjy_to_maggies(catalog[col].to_numpy(float)))
        return np.column_stack(values)

    def apply_error_floor(
        self,
        errors: np.ndarray,
        maggies: np.ndarray,
        bands: Sequence[str],
    ) -> np.ndarray:
        out = errors.copy()
        floors = dict(self.config.get("error_floor", {}))
        for i, band in enumerate(bands):
            if band in floors:
                out[:, i] = np.maximum(
                    out[:, i], np.abs(maggies[:, i]) * float(floors[band])
                )
        return out

    def valid_rows(
        self,
        redshift: np.ndarray,
        maggies: np.ndarray,
        errors: np.ndarray,
    ) -> np.ndarray:
        max_redshift = self.config.get("max_redshift")
        min_redshift = float(self.config.get("min_redshift", 0.0))
        # kcorrect asks Astropy for distance moduli; z <= 0 has no finite value.
        good = np.isfinite(redshift) & (redshift > 0.0) & (redshift >= min_redshift)
        if max_redshift is not None:
            good &= redshift <= float(max_redshift)
        good &= np.isfinite(maggies).all(axis=1)
        good &= np.isfinite(errors).all(axis=1)
        good &= (maggies > 0).all(axis=1)
        good &= (errors > 0).all(axis=1)
        return good

    @staticmethod
    def add_empty_outputs(
        catalog: pd.DataFrame,
        absmag_bands: Sequence[str],
        stellar_mass_col: str,
        linear_mass_col: Any,
    ) -> None:
        for band in absmag_bands:
            catalog[f"absmag_{band}"] = np.nan
        catalog[stellar_mass_col] = np.nan
        if linear_mass_col:
            catalog[str(linear_mass_col)] = np.nan

    def fit_and_write_outputs(
        self,
        catalog: pd.DataFrame,
        model: Any,
        *,
        redshift: np.ndarray,
        maggies: np.ndarray,
        errors: np.ndarray,
        good: np.ndarray,
        response_bands: Sequence[str],
        absmag_bands: Sequence[str],
        stellar_mass_col: str,
        linear_mass_col: Any,
    ) -> None:
        ivar = 1.0 / errors[good] ** 2
        coeffs = model.fit_coeffs(
            redshift=redshift[good],
            maggies=maggies[good],
            ivar=ivar,
        )
        absmag = model.absmag(
            redshift=redshift[good],
            maggies=maggies[good],
            ivar=ivar,
            coeffs=coeffs,
        )
        derived = model.derived(redshift=redshift[good], coeffs=coeffs)
        stellar_mass = np.asarray(derived["mremain"], dtype=float)

        response_lookup = {band: i for i, band in enumerate(response_bands)}
        for band in absmag_bands:
            if band not in response_lookup:
                raise ValueError(f"absmag band '{band}' not in response_bands")
            catalog.loc[good, f"absmag_{band}"] = absmag[:, response_lookup[band]]
        with np.errstate(divide="ignore", invalid="ignore"):
            catalog.loc[good, stellar_mass_col] = np.where(
                stellar_mass > 0,
                np.log10(stellar_mass),
                np.nan,
            )
        if linear_mass_col:
            catalog.loc[good, str(linear_mass_col)] = stellar_mass

    @staticmethod
    def kcorrect_class() -> Any:
        try:
            from kcorrect.kcorrect import Kcorrect
        except ImportError as exc:
            raise ImportError(
                "kcorrect enrichment requires the 'kcorrect' package"
            ) from exc
        return Kcorrect

    @staticmethod
    def resolve_response_names(responses: Any) -> list[str]:
        if not isinstance(responses, (list, tuple)):
            raise ValueError("kcorrect responses must be a list")
        return [
            KcorrectEnrichment.resolve_response_name(response) for response in responses
        ]

    @staticmethod
    def resolve_response_name(response: Any) -> str:
        path = Path(response)
        source_path = path
        if path.suffix == ".dat":
            path = path.with_suffix("")
        if path.parent == Path("."):
            return str(path)

        source_exists = source_path.exists() or (ROOT / source_path).exists()
        if not path.is_absolute() and source_exists:
            path = ROOT / path
        elif not path.is_absolute() and path.exists():
            path = path.resolve()
        elif path.is_absolute():
            path = path.resolve()
        return str(path)

    @staticmethod
    def infer_response_bands(model: Any) -> list[str]:
        known = {"u", "g", "r", "i", "z", "y"}
        bands = []
        for response in getattr(model, "responses", []):
            tokens = re.split(r"[_./\\-]+", str(response).lower())
            band = None
            for token in reversed(tokens):
                token = token.removesuffix("0")
                if token in known:
                    band = token
                    break
            if band is None:
                return []
            bands.append(band)
        return bands

    @staticmethod
    def resolve_path(path: Any) -> Path:
        if path is None:
            raise ValueError("kcorrect enrichment requires 'model'")
        out = Path(path)
        if not out.is_absolute() and not out.exists():
            out = ROOT / out
        return out

    @staticmethod
    def nanjy_to_maggies(value: np.ndarray) -> np.ndarray:
        return np.asarray(value, dtype=float) * 10 ** (-0.4 * MAGSYS_ZEROPOINT)


def apply_halo_mass(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Append Moster et al. (2013) halo masses and R200."""
    return HaloMassEnrichment(config).apply(catalog)


class HaloMassEnrichment:
    """Add Moster et al. (2013) halo masses and R200 to a catalog."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    def apply(self, catalog: pd.DataFrame) -> pd.DataFrame:
        stellar_mass_col = str(self.config.get("stellar_mass_col", "stellar_mass_log"))
        redshift_col = str(self.config.get("redshift_col", "z_phot"))
        missing = [
            col for col in (stellar_mass_col, redshift_col) if col not in catalog
        ]
        if missing:
            raise ValueError(f"halo_mass enrichment missing columns: {missing}")

        stellar_mass = self.stellar_mass_linear(catalog[stellar_mass_col])
        redshift = catalog[redshift_col].to_numpy(float)
        good = self.valid_rows(stellar_mass, redshift)

        out = catalog.copy()
        halo_col = str(self.config.get("halo_mass_col", "halo_mass_log"))
        r200_col = str(self.config.get("r200_col", "r200_mpc"))
        out[halo_col] = np.nan
        out[r200_col] = np.nan
        if not np.any(good):
            return out

        self.write_halo_outputs(
            out,
            stellar_mass=stellar_mass,
            redshift=redshift,
            good=good,
            halo_col=halo_col,
            r200_col=r200_col,
        )
        return out

    def stellar_mass_linear(self, stellar_mass: pd.Series) -> np.ndarray:
        values = stellar_mass.to_numpy(float)
        if bool(self.config.get("stellar_mass_is_log", True)):
            with np.errstate(over="ignore", invalid="ignore"):
                return 10**values
        return values

    def valid_rows(
        self,
        stellar_mass: np.ndarray,
        redshift: np.ndarray,
    ) -> np.ndarray:
        good = (
            np.isfinite(stellar_mass)
            & (stellar_mass > 0)
            & np.isfinite(redshift)
            & (redshift >= float(self.config.get("min_redshift", 0.0)))
        )
        max_redshift = self.config.get("max_redshift")
        if max_redshift is not None:
            good &= redshift <= float(max_redshift)
        return good

    def write_halo_outputs(
        self,
        catalog: pd.DataFrame,
        *,
        stellar_mass: np.ndarray,
        redshift: np.ndarray,
        good: np.ndarray,
        halo_col: str,
        r200_col: str,
    ) -> None:
        good_indices = np.where(good)[0]
        halos = np.array(
            [
                estimate_halo_mass(
                    stellar_mass[index],
                    redshift[index],
                    log_mass_min=float(self.config.get("log_mass_min", 9.0)),
                    log_mass_max=float(self.config.get("log_mass_max", 14.0)),
                )
                for index in good_indices
            ]
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            catalog.loc[good, halo_col] = np.where(halos > 0, np.log10(halos), np.nan)
        catalog.loc[good, r200_col] = [
            virial_radius_r200_mpc(halo, redshift[index])
            for index, halo in zip(good_indices, halos)
        ]


def estimate_stellar_mass_from_halo_mass(
    halo_mass: float,
    redshift: float,
) -> float:
    """Moster et al. (2013) stellar-mass to halo-mass relation."""
    zfactor = redshift / (1.0 + redshift)
    m1 = 10 ** (11.590 + 1.195 * zfactor)
    n = 0.0351 - 0.0247 * zfactor
    beta = 1.376 - 0.826 * zfactor
    gamma = 0.608 + 0.329 * zfactor
    ratio = halo_mass / m1
    return 2.0 * halo_mass * n * (ratio**-beta + ratio**gamma) ** -1


def estimate_halo_mass(
    stellar_mass: float,
    redshift: float,
    *,
    log_mass_min: float = 9.0,
    log_mass_max: float = 14.0,
) -> float:
    """Invert the stellar-mass to halo-mass relation."""
    if not np.isfinite(stellar_mass) or stellar_mass <= 0 or not np.isfinite(redshift):
        return np.nan
    log_stellar_mass = np.log10(stellar_mass)

    def objective(log_halo_mass: float) -> float:
        halo_mass = 10**log_halo_mass
        model = estimate_stellar_mass_from_halo_mass(halo_mass, redshift)
        if not np.isfinite(model) or model <= 0:
            return np.nan
        return np.log10(model) - log_stellar_mass

    try:
        return 10 ** brentq(objective, log_mass_min, log_mass_max)
    except ValueError:
        return np.nan


def virial_radius_r200_mpc(halo_mass: float, redshift: float) -> float:
    """Return R200 in physical Mpc for a halo mass in solar masses."""
    if not np.isfinite(halo_mass) or halo_mass <= 0 or not np.isfinite(redshift):
        return np.nan
    critical_density = cosmo.critical_density(redshift).to_value(u.M_sun / u.Mpc**3)
    return (3.0 * halo_mass / (800.0 * np.pi * critical_density)) ** (1.0 / 3.0)
