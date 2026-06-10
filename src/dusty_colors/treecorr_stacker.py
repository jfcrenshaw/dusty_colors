"""TreeCorr-based color stacking on prepared canonical samples."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from typing import Any, Literal

import healpy as hp
import numpy as np
import pandas as pd
import treecorr
import yaml
from astropy.cosmology import Planck18 as cosmo

from .observables import build_observable, observable_column_names

ColorMode = Literal["fcolors", "mcolors"]
StackDirection = Literal["forward", "flipped"]


@dataclass(frozen=True)
class _RadialBin:
    lo: float
    hi: float
    center: float


@dataclass
class _BinResult:
    raw: float
    raw_err: float
    weight: float
    npairs: float
    corrs: tuple[Any, ...]


@dataclass
class _Profile:
    raw: np.ndarray
    raw_err: np.ndarray
    color: np.ndarray
    color_err: np.ndarray
    weight: np.ndarray
    npairs: np.ndarray
    ref_color: float
    ref_color_err: float
    corrs: list[tuple[Any, ...]]


@dataclass
class _JackknifeStats:
    mean: np.ndarray
    err: np.ndarray
    covariance: np.ndarray
    samples: np.ndarray


@dataclass
class _EstimatorResult:
    signal: np.ndarray
    signal_err: np.ndarray
    covariance: np.ndarray
    analytic_err: np.ndarray
    delta: np.ndarray
    delta_err: np.ndarray
    raw_delta: np.ndarray
    raw_delta_err: np.ndarray
    ref: float
    ref_err: float
    raw_ref: float
    raw_ref_err: float
    random_delta: np.ndarray | None
    random_delta_err: np.ndarray | None
    random_ref: float | None
    random_ref_err: float | None
    jackknife: _JackknifeStats | None


@dataclass
class TreeCorrStacker:
    """Measure color excess profiles with the TreeCorr estimator.

    The stacker consumes prepared canonical foreground/background tables. It does
    not prepare catalogs, select samples, or manage pipeline-level manifests.
    Expected canonical columns are ``ra``, ``dec``, ``z_phot`` on foreground
    samples, ``ra`` and ``dec`` on background samples, and either
    ``flux_<band>``/``fluxerr_<band>`` or ``mag_<band>``/``magerr_<band>`` for
    requested colors.

    The workflow is: validate samples, prepare jackknife/random catalogs, measure
    forward/flipped profiles, subtract random and reference-annulus signals, then
    write arrays for plotting and downstream checks.
    """

    foreground: pd.DataFrame
    background: pd.DataFrame
    out_dir: Path
    footprint: pd.DataFrame | None = None
    config: Mapping[str, Any] = field(default_factory=dict)

    colors: tuple[str, ...] = ("g-r", "r-i", "i-z", "g-i")
    modes: tuple[ColorMode, ...] = ("fcolors", "mcolors")
    r_bin_edges: list[float] | np.ndarray = field(
        default_factory=lambda: np.geomspace(5.0, 1000.0, 6)
    )
    reference_annulus: tuple[float, float] = (2000.0, 4000.0)
    snr_max: float = 100.0

    bin_slop: float = 0.0
    num_threads: int | None = None
    jackknife: bool = True
    patch_col: str = "jackknife_region"
    cross_patch_weight: str = "match"
    random_correction: bool = True
    random_multiplier: float = 5.0
    random_seed: int = 42
    random_nside: int = 1024
    flipped_correction: bool = True
    prefer_observable_columns: bool = False
    diagnostic_plots: bool = True
    diagnostic_photoz_bins: int | list[float] | np.ndarray = 40
    diagnostic_color_bins: int | list[float] | np.ndarray = 40

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.foreground = self.foreground.reset_index(drop=True).copy()
        self.background = self.background.reset_index(drop=True).copy()
        self.footprint = (
            None
            if self.footprint is None
            else self.footprint.reset_index(drop=True).copy()
        )

        self.colors = tuple(self.colors)
        self.modes = tuple(self.modes)
        self.r_bin_edges = np.asarray(self.r_bin_edges, dtype=float)
        self.reference_annulus = (
            float(self.reference_annulus[0]),
            float(self.reference_annulus[1]),
        )

        self._fg_pos_cat: Any | None = None
        self._bg_pos_cat: Any | None = None
        self._random_fg_pos_cat: Any | None = None
        self._random_bg_pos_cat: Any | None = None
        self._random_foreground: pd.DataFrame | None = None
        self._random_background: pd.DataFrame | None = None
        self._random_foreground_da: np.ndarray | None = None
        self._treecorr_patch_col = "_treecorr_patch"
        self._diagnostic_cache: dict[ColorMode, dict[str, np.ndarray]] = {}

    @classmethod
    def from_sample_dir(
        cls,
        sample_dir: str | Path,
        out_dir: str | Path,
        stack_config: Mapping[str, Any],
        *,
        footprint_path: str | Path | None = None,
    ) -> TreeCorrStacker:
        """Build a stacker from a prepared sample directory."""
        sample_dir = Path(sample_dir)
        footprint = None
        if footprint_path is not None and Path(footprint_path).exists():
            footprint = pd.read_parquet(footprint_path)

        config = dict(stack_config)
        if "reference_annulus" in config:
            config["reference_annulus"] = tuple(config["reference_annulus"])
        stacker_kwargs = cls._init_kwargs(config)
        return cls(
            foreground=pd.read_parquet(sample_dir / "foreground.parquet"),
            background=pd.read_parquet(sample_dir / "background.parquet"),
            footprint=footprint,
            out_dir=Path(out_dir),
            config=config,
            **stacker_kwargs,
        )

    @classmethod
    def _init_kwargs(cls, config: Mapping[str, Any]) -> dict[str, Any]:
        blocked = {"foreground", "background", "footprint", "out_dir", "config"}
        allowed = {
            field.name
            for field in dataclass_fields(cls)
            if field.init and field.name not in blocked
        }
        return {key: value for key, value in config.items() if key in allowed}

    def run(self, *, force: bool = False) -> None:
        """Run all requested stack modes."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

        if not force and all(path.exists() for path in self._expected_files()):
            print(f"TreeCorr stacking already done: {self.out_dir}")
            return

        self._load_samples()
        bins = self._radial_bins()
        ref_bin = self._reference_bin()

        print(f"Running TreeCorr stacking: {self.out_dir}")
        for mode in self.modes:
            self._run_mode(mode, bins, ref_bin)
        print("   TreeCorr stacking complete")

    def _load_samples(self) -> None:
        self._validate_samples()
        self._foreground_da = cosmo.angular_diameter_distance(
            self.foreground["z_phot"].to_numpy(float)
        ).to_value("kpc")
        self._drop_bad_positions()
        self._setup_jackknife()
        self._setup_randoms()

    def _validate_samples(self) -> None:
        fg_required = {"ra", "dec", "z_phot"}
        bg_required = {"ra", "dec"}
        self._require_columns(self.foreground, fg_required, "foreground")
        self._require_columns(self.background, bg_required, "background")
        for color in self.colors:
            for mode in self.modes:
                self._validate_observable_columns(color, mode)

    def _validate_observable_columns(self, color: str, mode: ColorMode) -> None:
        catalogs = [(f"background {color}", self.background)]
        if self.flipped_correction:
            catalogs.append((f"foreground {color}", self.foreground))

        for label, catalog in catalogs:
            try:
                build_observable(
                    catalog.iloc[:0],
                    color,
                    mode,
                    snr_max=self.snr_max,
                )
            except KeyError as exc:
                raise ValueError(
                    f"{label} needs either flux or magnitude columns for {mode}"
                ) from exc

    def _require_columns(
        self, catalog: pd.DataFrame, columns: set[str], label: str
    ) -> None:
        missing = sorted(columns - set(catalog.columns))
        if missing:
            raise ValueError(f"{label} is missing required columns: {missing}")

    def _drop_bad_positions(self) -> None:
        fg_good = (
            np.isfinite(self.foreground["ra"].to_numpy(float))
            & np.isfinite(self.foreground["dec"].to_numpy(float))
            & np.isfinite(self._foreground_da)
            & (self._foreground_da > 0)
        )
        bg_good = np.isfinite(self.background["ra"].to_numpy(float)) & np.isfinite(
            self.background["dec"].to_numpy(float)
        )

        self.foreground = self.foreground.loc[fg_good].reset_index(drop=True)
        self._foreground_da = self._foreground_da[fg_good]
        self.background = self.background.loc[bg_good].reset_index(drop=True)

        if len(self.foreground) == 0 or len(self.background) == 0:
            raise ValueError("Foreground and background samples must both be non-empty")

    def _setup_jackknife(self) -> None:
        self._use_jackknife = False
        self._npatch = 1
        if not self.jackknife:
            self.foreground[self._treecorr_patch_col] = 0
            self.background[self._treecorr_patch_col] = 0
            return

        if (
            self.patch_col not in self.foreground
            or self.patch_col not in self.background
        ):
            raise ValueError(f"Jackknife patch column '{self.patch_col}' not found")

        fg_patch = self.foreground[self.patch_col].to_numpy(int)
        bg_patch = self.background[self.patch_col].to_numpy(int)
        patches = np.intersect1d(np.unique(fg_patch), np.unique(bg_patch))
        if len(patches) < 2:
            raise ValueError(
                "TreeCorr jackknife covariance needs at least two shared patches"
            )

        fg_keep = np.isin(fg_patch, patches)
        bg_keep = np.isin(bg_patch, patches)
        if not np.all(fg_keep) or not np.all(bg_keep):
            print(
                "   TreeCorr jackknife using "
                f"{len(patches)} shared patches; dropped "
                f"{int(np.sum(~fg_keep))} foreground and "
                f"{int(np.sum(~bg_keep))} background objects outside them"
            )
            self.foreground = self.foreground.loc[fg_keep].reset_index(drop=True)
            self._foreground_da = self._foreground_da[fg_keep]
            self.background = self.background.loc[bg_keep].reset_index(drop=True)

        patch_map = {patch: i for i, patch in enumerate(patches)}
        self.foreground[self._treecorr_patch_col] = [
            patch_map[patch] for patch in self.foreground[self.patch_col].to_numpy(int)
        ]
        self.background[self._treecorr_patch_col] = [
            patch_map[patch] for patch in self.background[self.patch_col].to_numpy(int)
        ]

        self._npatch = len(patches)
        self._use_jackknife = True

    def _setup_randoms(self) -> None:
        if not self.random_correction:
            return
        if self.footprint is None:
            raise ValueError("Random correction requires a prepared footprint table")
        if "pixel" not in self.footprint:
            raise ValueError("Random correction requires footprint column 'pixel'")
        if self.random_multiplier <= 0:
            raise ValueError("random_multiplier must be positive")

        footprint = self.footprint
        if getattr(self, "_use_jackknife", False):
            if self.patch_col not in footprint:
                raise ValueError(
                    f"Random correction requires footprint column '{self.patch_col}'"
                )
            patch_map = dict(
                zip(
                    self.foreground[self.patch_col].to_numpy(int),
                    self.foreground[self._treecorr_patch_col].to_numpy(int),
                )
            )
            patch_map.update(
                zip(
                    self.background[self.patch_col].to_numpy(int),
                    self.background[self._treecorr_patch_col].to_numpy(int),
                )
            )
            footprint = footprint[footprint[self.patch_col].isin(patch_map)]
            footprint = footprint.assign(
                **{
                    self._treecorr_patch_col: [
                        patch_map[patch]
                        for patch in footprint[self.patch_col].to_numpy(int)
                    ]
                }
            )
        else:
            footprint = footprint.assign(**{self._treecorr_patch_col: 0})

        pixels_by_patch = {
            patch: np.unique(group["pixel"].dropna().to_numpy(int))
            for patch, group in footprint.groupby(self._treecorr_patch_col)
        }
        pixels_by_patch = {
            patch: pixels for patch, pixels in pixels_by_patch.items() if len(pixels)
        }
        missing = set(self.foreground[self._treecorr_patch_col].unique()) - set(
            pixels_by_patch
        )
        missing |= set(self.background[self._treecorr_patch_col].unique()) - set(
            pixels_by_patch
        )
        if missing:
            raise ValueError(f"Missing random-footprint pixels for patches: {missing}")

        rng = np.random.default_rng(self.random_seed)
        self._random_foreground, self._random_foreground_da = self._random_catalog_like(
            self.foreground,
            self._foreground_da,
            pixels_by_patch,
            rng,
        )
        self._random_background, _ = self._random_catalog_like(
            self.background,
            None,
            pixels_by_patch,
            rng,
        )
        print(
            "   TreeCorr random correction using "
            f"{len(self._random_foreground)} foreground and "
            f"{len(self._random_background)} background random positions"
        )

    def _random_catalog_like(
        self,
        catalog: pd.DataFrame,
        radial_distance: np.ndarray | None,
        pixels_by_patch: dict[int, np.ndarray],
        rng: np.random.Generator,
    ) -> tuple[pd.DataFrame, np.ndarray | None]:
        rows = []
        distances = []
        for patch in sorted(catalog[self._treecorr_patch_col].unique()):
            in_patch = catalog[self._treecorr_patch_col].to_numpy(int) == patch
            indices = np.where(in_patch)[0]
            n_random = max(1, int(np.ceil(self.random_multiplier * len(indices))))
            templates = rng.choice(indices, size=n_random, replace=True)
            ra, dec = self._sample_patch_positions(
                pixels_by_patch[int(patch)],
                n_random,
                rng,
            )
            rows.append(
                pd.DataFrame(
                    {
                        "ra": ra,
                        "dec": dec,
                        self._treecorr_patch_col: int(patch),
                    }
                )
            )
            if radial_distance is not None:
                distances.append(radial_distance[templates])

        random_catalog = pd.concat(rows, ignore_index=True)
        random_distance = np.concatenate(distances) if distances else None
        return random_catalog, random_distance

    def _sample_patch_positions(
        self,
        pixels: np.ndarray,
        n_random: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        pixels = np.asarray(pixels, dtype=int)
        pixel_set = set(pixels.tolist())
        lon, lat = hp.pix2ang(self.random_nside, pixels, nest=True, lonlat=True)
        pixel_radius = np.rad2deg(np.sqrt(hp.nside2pixarea(self.random_nside)))
        ra_min = np.min(lon) - pixel_radius
        ra_max = np.max(lon) + pixel_radius
        dec_min = np.max([np.min(lat) - pixel_radius, -90.0])
        dec_max = np.min([np.max(lat) + pixel_radius, 90.0])

        ras: list[np.ndarray] = []
        decs: list[np.ndarray] = []
        n_have = 0
        while n_have < n_random:
            batch = max(1024, 4 * (n_random - n_have))
            ra = rng.uniform(ra_min, ra_max, size=batch)
            sin_dec = rng.uniform(
                np.sin(np.deg2rad(dec_min)),
                np.sin(np.deg2rad(dec_max)),
                size=batch,
            )
            dec = np.rad2deg(np.arcsin(sin_dec))
            pix = hp.ang2pix(self.random_nside, ra, dec, nest=True, lonlat=True)
            keep = np.fromiter((p in pixel_set for p in pix), dtype=bool, count=batch)
            if np.any(keep):
                ras.append(ra[keep])
                decs.append(dec[keep])
                n_have += int(np.sum(keep))

        return np.concatenate(ras)[:n_random], np.concatenate(decs)[:n_random]

    def _run_mode(
        self, mode: ColorMode, bins: list[_RadialBin], ref_bin: _RadialBin
    ) -> None:
        results = {}

        print(f"   stacking {mode} with TreeCorr...", end="", flush=True)
        for color in self.colors:
            print(f" {color}", end="", flush=True)
            background = self._observable(self.background, color, mode)
            foreground = (
                self._observable(self.foreground, color, mode)
                if self.flipped_correction
                else None
            )

            forward = self._profile(background, bins, ref_bin, mode, "forward")
            flipped = (
                self._profile(foreground, bins, ref_bin, mode, "flipped")
                if foreground is not None
                else None
            )
            random_forward = None
            random_flipped = None
            if self.random_correction:
                random_forward = self._profile(
                    background, bins, ref_bin, mode, "forward", random=True
                )
                if foreground is not None:
                    random_flipped = self._profile(
                        foreground, bins, ref_bin, mode, "flipped", random=True
                    )

            results.update(
                self._result(
                    color,
                    bins,
                    mode,
                    forward,
                    flipped,
                    random_forward,
                    random_flipped,
                )
            )
        print(".")

        results.update(self._diagnostic_arrays(mode, bins))
        np.savez_compressed(self._stack_file(mode), **results)

    def _observable(
        self, catalog: pd.DataFrame, color: str, mode: ColorMode
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        value_col, err_col = observable_column_names(color, mode)
        if self.prefer_observable_columns and {value_col, err_col}.issubset(catalog):
            value = catalog[value_col].to_numpy(float)
            err = catalog[err_col].to_numpy(float)
        else:
            observable = build_observable(catalog, color, mode, snr_max=self.snr_max)
            value = observable["value"].to_numpy(float)
            err = observable["error"].to_numpy(float)
        good = np.isfinite(value) & np.isfinite(err) & (err > 0)
        if mode == "fcolors":
            good &= value > 0
        return value, err, good

    def _profile(
        self,
        observable: tuple[np.ndarray, np.ndarray, np.ndarray],
        bins: list[_RadialBin],
        ref_bin: _RadialBin,
        mode: ColorMode,
        direction: StackDirection,
        random: bool = False,
    ) -> _Profile:
        value, err, good = observable
        bin_results = [
            self._stack_bin(value, err, good, b, direction, random=random)
            for b in bins + [ref_bin]
        ]
        rows = np.array(
            [
                (result.raw, result.raw_err, result.weight, result.npairs)
                for result in bin_results
            ]
        )

        raw = rows[:-1, 0]
        raw_err = rows[:-1, 1]
        color, color_err = self._to_color(raw, raw_err, mode)
        ref_color, ref_color_err = self._to_color(rows[-1:, 0], rows[-1:, 1], mode)

        return _Profile(
            raw=raw,
            raw_err=raw_err,
            color=color,
            color_err=color_err,
            weight=rows[:-1, 2],
            npairs=rows[:-1, 3],
            ref_color=float(ref_color[0]),
            ref_color_err=float(ref_color_err[0]),
            corrs=[result.corrs for result in bin_results],
        )

    def _stack_bin(
        self,
        value: np.ndarray,
        err: np.ndarray,
        good: np.ndarray,
        radial_bin: _RadialBin,
        direction: StackDirection,
        random: bool = False,
    ) -> _BinResult:
        mean, count_weight, npairs, _ = self._pair_mean(
            value, good, radial_bin, direction, random=random
        )
        mean2, _, _, _ = self._pair_mean(
            value**2, good, radial_bin, direction, random=random
        )
        if count_weight == 0 or not np.isfinite(mean) or not np.isfinite(mean2):
            return _BinResult(np.nan, np.nan, 0.0, npairs, ())

        variance_floor = max(mean2 - mean**2, 0.0)
        weight = np.zeros_like(err)
        weight[good] = 1.0 / (err[good] ** 2 + variance_floor)

        avg, weight_sum, npairs, corrs = self._pair_mean(
            value, good, radial_bin, direction, weight, random=random
        )
        avg_err = np.sqrt(1.0 / weight_sum) if weight_sum > 0 else np.nan
        return _BinResult(avg, avg_err, weight_sum, npairs, corrs)

    def _pair_mean(
        self,
        value: np.ndarray,
        good: np.ndarray,
        radial_bin: _RadialBin,
        direction: StackDirection,
        weight: np.ndarray | None = None,
        random: bool = False,
    ) -> tuple[float, float, float, tuple[Any, ...]]:
        if not np.any(good):
            return np.nan, 0.0, 0.0, ()

        if direction == "forward":
            source = self._background_catalog(
                good, k=value[good], w=None if weight is None else weight[good]
            )
            corr = self._nk(radial_bin)
            corr.process(
                (
                    self._random_foreground_position_catalog()
                    if random
                    else self._foreground_position_catalog()
                ),
                source,
                metric="Rlens",
                num_threads=self.num_threads,
            )
            return (
                self._safe_mean(corr.xi[0], corr.weight[0]),
                corr.weight[0],
                corr.npairs[0],
                (corr,),
            )

        pair_weight = np.ones(np.sum(good)) if weight is None else weight[good]
        denominator = self._nn(radial_bin)
        denominator.process(
            self._foreground_catalog(good, w=pair_weight),
            (
                self._random_background_position_catalog()
                if random
                else self._background_position_catalog()
            ),
            metric="Rlens",
            num_threads=self.num_threads,
        )
        if denominator.weight[0] == 0:
            return np.nan, 0.0, denominator.npairs[0], ()

        numerator = self._nn(radial_bin)
        numerator.process(
            self._foreground_catalog(good, w=pair_weight * value[good]),
            (
                self._random_background_position_catalog()
                if random
                else self._background_position_catalog()
            ),
            metric="Rlens",
            num_threads=self.num_threads,
        )
        return (
            numerator.weight[0] / denominator.weight[0],
            denominator.weight[0],
            denominator.npairs[0],
            (numerator, denominator),
        )

    def _result(
        self,
        color: str,
        bins: list[_RadialBin],
        mode: ColorMode,
        forward: _Profile,
        flipped: _Profile | None,
        random_forward: _Profile | None = None,
        random_flipped: _Profile | None = None,
    ) -> dict[str, np.ndarray]:
        estimate = self._color_excess_estimate(
            mode,
            forward,
            flipped,
            len(bins),
            random_forward=random_forward,
            random_flipped=random_flipped,
        )
        return self._result_arrays(
            color,
            bins,
            forward,
            flipped,
            estimate,
            random_forward=random_forward,
            random_flipped=random_flipped,
        )

    def _color_excess_estimate(
        self,
        mode: ColorMode,
        forward: _Profile,
        flipped: _Profile | None,
        n_bins: int,
        random_forward: _Profile | None = None,
        random_flipped: _Profile | None = None,
    ) -> _EstimatorResult:
        use_flipped = self.flipped_correction
        if use_flipped and flipped is None:
            raise ValueError("Flipped correction requires a flipped profile")

        has_random = random_forward is not None or random_flipped is not None
        if has_random and random_forward is None:
            raise ValueError("Random correction requires a random forward profile")
        if use_flipped and has_random and random_flipped is None:
            raise ValueError(
                "Random flipped correction requires a random flipped profile"
            )

        raw_delta = forward.color.copy()
        raw_delta_err = forward.color_err.copy()
        raw_ref = forward.ref_color
        raw_ref_err = forward.ref_color_err
        if use_flipped:
            raw_delta = raw_delta - flipped.color
            raw_delta_err = np.hypot(raw_delta_err, flipped.color_err)
            raw_ref = raw_ref - flipped.ref_color
            raw_ref_err = np.hypot(raw_ref_err, flipped.ref_color_err)

        random_delta = np.zeros_like(raw_delta)
        random_delta_err = np.zeros_like(raw_delta_err)
        random_ref = 0.0
        random_ref_err = 0.0
        if has_random:
            random_delta = random_forward.color.copy()
            random_delta_err = random_forward.color_err.copy()
            random_ref = random_forward.ref_color
            random_ref_err = random_forward.ref_color_err
            if use_flipped:
                random_delta = random_delta - random_flipped.color
                random_delta_err = np.hypot(random_delta_err, random_flipped.color_err)
                random_ref = random_ref - random_flipped.ref_color
                random_ref_err = np.hypot(
                    random_ref_err,
                    random_flipped.ref_color_err,
                )

        delta = raw_delta - random_delta
        delta_err = np.hypot(raw_delta_err, random_delta_err)
        ref = raw_ref - random_ref
        ref_err = np.hypot(raw_ref_err, random_ref_err)
        estimator = delta - ref
        analytic_err = np.hypot(delta_err, ref_err)
        jackknife = self._jackknife_stats(
            mode,
            forward,
            flipped,
            n_bins,
            random_forward=random_forward,
            random_flipped=random_flipped,
        )
        covariance = (
            jackknife.covariance if jackknife is not None else np.diag(analytic_err**2)
        )
        estimator_err = jackknife.err if jackknife is not None else analytic_err

        return _EstimatorResult(
            signal=estimator,
            signal_err=estimator_err,
            covariance=covariance,
            analytic_err=analytic_err,
            delta=delta,
            delta_err=delta_err,
            raw_delta=raw_delta,
            raw_delta_err=raw_delta_err,
            ref=float(ref),
            ref_err=float(ref_err),
            raw_ref=float(raw_ref),
            raw_ref_err=float(raw_ref_err),
            random_delta=random_delta if has_random else None,
            random_delta_err=random_delta_err if has_random else None,
            random_ref=float(random_ref) if has_random else None,
            random_ref_err=float(random_ref_err) if has_random else None,
            jackknife=jackknife,
        )

    def _result_arrays(
        self,
        color: str,
        bins: list[_RadialBin],
        forward: _Profile,
        flipped: _Profile | None,
        estimate: _EstimatorResult,
        random_forward: _Profile | None = None,
        random_flipped: _Profile | None = None,
    ) -> dict[str, np.ndarray]:
        use_flipped = self.flipped_correction
        result = {
            f"{color}_bin_centers": self._bin_centers(bins),
            f"{color}_avg": estimate.signal,
            f"{color}_err": estimate.signal_err,
            f"{color}_cov": estimate.covariance,
            f"{color}_analytic_err": estimate.analytic_err,
            f"{color}_delta_avg": estimate.delta,
            f"{color}_delta_err": estimate.delta_err,
            f"{color}_raw_delta_avg": estimate.raw_delta,
            f"{color}_raw_delta_err": estimate.raw_delta_err,
            f"{color}_ref_avg": np.array(estimate.ref),
            f"{color}_ref_err": np.array(estimate.ref_err),
            f"{color}_raw_ref_avg": np.array(estimate.raw_ref),
            f"{color}_raw_ref_err": np.array(estimate.raw_ref_err),
            f"{color}_uncorrected_avg": estimate.raw_delta - estimate.raw_ref,
            f"{color}_forward_avg": forward.color,
            f"{color}_forward_err": forward.color_err,
            f"{color}_forward_raw_avg": forward.raw,
            f"{color}_forward_raw_err": forward.raw_err,
            f"{color}_forward_npairs": forward.npairs,
        }
        if use_flipped:
            result.update(
                {
                    f"{color}_flipped_avg": flipped.color,
                    f"{color}_flipped_err": flipped.color_err,
                    f"{color}_flipped_raw_avg": flipped.raw,
                    f"{color}_flipped_raw_err": flipped.raw_err,
                    f"{color}_flipped_npairs": flipped.npairs,
                }
            )
        if estimate.random_delta is not None:
            result.update(
                {
                    f"{color}_random_delta_avg": estimate.random_delta,
                    f"{color}_random_delta_err": estimate.random_delta_err,
                    f"{color}_random_ref_avg": np.array(estimate.random_ref),
                    f"{color}_random_ref_err": np.array(estimate.random_ref_err),
                    f"{color}_random_forward_avg": random_forward.color,
                    f"{color}_random_forward_err": random_forward.color_err,
                    f"{color}_random_forward_raw_avg": random_forward.raw,
                    f"{color}_random_forward_raw_err": random_forward.raw_err,
                    f"{color}_random_forward_npairs": random_forward.npairs,
                }
            )
            if use_flipped:
                result.update(
                    {
                        f"{color}_random_flipped_avg": random_flipped.color,
                        f"{color}_random_flipped_err": random_flipped.color_err,
                        f"{color}_random_flipped_raw_avg": random_flipped.raw,
                        f"{color}_random_flipped_raw_err": random_flipped.raw_err,
                        f"{color}_random_flipped_npairs": random_flipped.npairs,
                    }
                )
        if estimate.jackknife is not None:
            result.update(
                {
                    f"{color}_jackknife_avg": estimate.jackknife.mean,
                    f"{color}_jackknife_err": estimate.jackknife.err,
                    f"{color}_jackknife_samples": estimate.jackknife.samples,
                    f"{color}_jackknife_patch": np.arange(
                        estimate.jackknife.samples.shape[0]
                    ),
                }
            )
        return result

    def _jackknife_stats(
        self,
        mode: ColorMode,
        forward: _Profile,
        flipped: _Profile | None,
        n_bins: int,
        random_forward: _Profile | None = None,
        random_flipped: _Profile | None = None,
    ) -> _JackknifeStats | None:
        if not getattr(self, "_use_jackknife", False):
            return None

        use_flipped = self.flipped_correction
        if use_flipped and flipped is None:
            raise ValueError("Flipped correction requires a flipped profile")

        has_random = random_forward is not None or random_flipped is not None
        if has_random and random_forward is None:
            raise ValueError("Random correction requires a random forward profile")
        if use_flipped and has_random and random_flipped is None:
            raise ValueError(
                "Random flipped correction requires a random flipped profile"
            )

        all_profile_corrs = list(forward.corrs)
        if use_flipped:
            all_profile_corrs += list(flipped.corrs)
        if has_random:
            all_profile_corrs += list(random_forward.corrs)
            if use_flipped:
                all_profile_corrs += list(random_flipped.corrs)
        if any(len(corrs) == 0 for corrs in all_profile_corrs):
            return None

        corrs = []
        for i in range(len(forward.corrs)):
            corrs.extend(forward.corrs[i])
            if use_flipped:
                corrs.extend(flipped.corrs[i])
            if has_random:
                corrs.extend(random_forward.corrs[i])
                if use_flipped:
                    corrs.extend(random_flipped.corrs[i])

        def read_forward(corrs: list[Any], i: int) -> tuple[float, int]:
            return corrs[i].xi[0], i + 1

        def read_flipped(corrs: list[Any], i: int) -> tuple[float, int]:
            numerator = corrs[i]
            denominator = corrs[i + 1]
            i += 2
            if denominator.weight[0] == 0:
                return np.nan, i
            return numerator.weight[0] / denominator.weight[0], i

        def estimator(corrs: list[Any]) -> np.ndarray:
            raw_forward = []
            raw_flipped = []
            raw_random_forward = []
            raw_random_flipped = []
            i = 0
            for _ in range(n_bins + 1):
                value, i = read_forward(corrs, i)
                raw_forward.append(value)

                if use_flipped:
                    value, i = read_flipped(corrs, i)
                    raw_flipped.append(value)

                if has_random:
                    value, i = read_forward(corrs, i)
                    raw_random_forward.append(value)

                    if use_flipped:
                        value, i = read_flipped(corrs, i)
                        raw_random_flipped.append(value)

            forward_color = self._raw_to_color(np.array(raw_forward), mode)
            delta = forward_color.copy()
            if use_flipped:
                flipped_color = self._raw_to_color(np.array(raw_flipped), mode)
                delta -= flipped_color
            if has_random:
                random_forward_color = self._raw_to_color(
                    np.array(raw_random_forward), mode
                )
                delta -= random_forward_color
                if use_flipped:
                    random_flipped_color = self._raw_to_color(
                        np.array(raw_random_flipped), mode
                    )
                    delta += random_flipped_color
            ref = delta[-1]
            delta = delta[:-1]
            return delta - ref

        samples, _ = treecorr.build_multi_cov_design_matrix(
            corrs,
            method="jackknife",
            func=estimator,
            cross_patch_weight=self.cross_patch_weight,
        )
        mean = np.mean(samples, axis=0)
        centered = samples - mean
        covariance = (1.0 - 1.0 / samples.shape[0]) * centered.conj().T.dot(centered)
        covariance = np.real_if_close(covariance)
        err = np.sqrt(np.clip(np.diag(covariance), 0, None))
        return _JackknifeStats(
            mean=mean,
            err=err,
            covariance=covariance,
            samples=samples,
        )

    def _diagnostic_arrays(
        self,
        mode: ColorMode,
        bins: list[_RadialBin],
    ) -> dict[str, np.ndarray]:
        if not self.diagnostic_plots:
            return {}
        if mode not in self._diagnostic_cache:
            self._diagnostic_cache[mode] = self._compute_diagnostic_arrays(mode, bins)
        return self._diagnostic_cache[mode]

    def _compute_diagnostic_arrays(
        self,
        mode: ColorMode,
        bins: list[_RadialBin],
    ) -> dict[str, np.ndarray]:
        if len(self.foreground) == 0 or len(self.background) == 0:
            return {}

        from scipy.spatial import cKDTree

        radial_edges = np.asarray(self.r_bin_edges, dtype=float)
        fg_vectors = self._unit_vectors(self.foreground)
        bg_vectors = self._unit_vectors(self.background)
        background_tree = cKDTree(bg_vectors)

        background_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if "z_phot" in self.background:
            z_phot = self.background["z_phot"].to_numpy(float)
            background_values["photoz"] = (z_phot, np.isfinite(z_phot))

        for color in self.colors:
            value, _, good = self._observable(self.background, color, mode)
            color_value = self._raw_to_color(value, mode)
            background_values[f"color:{color}"] = (
                color_value,
                np.isfinite(color_value) & good,
            )

        hist_edges: dict[str, np.ndarray] = {}
        if "photoz" in background_values:
            hist_edges["photoz"] = self._diagnostic_bin_edges(
                background_values["photoz"][0],
                background_values["photoz"][1],
                self.diagnostic_photoz_bins,
                "diagnostic_photoz_bins",
            )
        for color in self.colors:
            key = f"color:{color}"
            hist_edges[key] = self._diagnostic_bin_edges(
                background_values[key][0],
                background_values[key][1],
                self.diagnostic_color_bins,
                "diagnostic_color_bins",
            )

        hist_counts = {
            key: np.zeros((len(bins), len(edges) - 1), dtype=float)
            for key, edges in hist_edges.items()
        }

        max_r = radial_edges[-1]
        for fg_vector, da in zip(fg_vectors, self._foreground_da):
            if not np.isfinite(da) or da <= 0:
                continue
            max_theta = max_r / da
            max_chord = 2.0 * np.sin(0.5 * max_theta)
            neighbors = background_tree.query_ball_point(fg_vector, max_chord)
            if not neighbors:
                continue

            bg_index = np.asarray(neighbors, dtype=int)
            chords = np.linalg.norm(bg_vectors[bg_index] - fg_vector, axis=1)
            theta = 2.0 * np.arcsin(np.clip(0.5 * chords, 0.0, 1.0))
            radius = theta * da
            radial_bin = np.searchsorted(radial_edges, radius, side="right") - 1
            in_range = (radial_bin >= 0) & (radial_bin < len(bins))
            if not np.any(in_range):
                continue

            bg_index = bg_index[in_range]
            radial_bin = radial_bin[in_range]
            for key, edges in hist_edges.items():
                values, good = background_values[key]
                good_pair = good[bg_index]
                if not np.any(good_pair):
                    continue
                self._add_pair_histograms(
                    hist_counts[key],
                    radial_bin[good_pair],
                    values[bg_index][good_pair],
                    edges,
                )

        result = {
            "diagnostic_radial_bin_edges": radial_edges,
            "diagnostic_radial_bin_centers": self._bin_centers(bins),
        }
        if "photoz" in hist_counts:
            result.update(
                {
                    "diagnostic_photoz_bin_edges": hist_edges["photoz"],
                    "diagnostic_photoz_counts": hist_counts["photoz"],
                }
            )
        for color in self.colors:
            key = f"color:{color}"
            result.update(
                {
                    f"{color}_diagnostic_color_bin_edges": hist_edges[key],
                    f"{color}_diagnostic_color_counts": hist_counts[key],
                }
            )
        return result

    def _diagnostic_bin_edges(
        self,
        values: np.ndarray,
        good: np.ndarray,
        bins: int | list[float] | np.ndarray,
        label: str,
    ) -> np.ndarray:
        if isinstance(bins, int):
            if bins < 1:
                raise ValueError(f"{label} must be positive")
            finite = values[good & np.isfinite(values)]
            if len(finite) == 0:
                return np.linspace(0.0, 1.0, bins + 1)
            lo = float(np.nanmin(finite))
            hi = float(np.nanmax(finite))
            if not np.isfinite(lo) or not np.isfinite(hi):
                return np.linspace(0.0, 1.0, bins + 1)
            if lo == hi:
                pad = 0.5 if lo == 0 else 0.05 * abs(lo)
                lo -= pad
                hi += pad
            return np.linspace(lo, hi, bins + 1)

        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError(f"{label} must contain at least two edges")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError(f"{label} edges must be finite and increasing")
        return edges

    def _add_pair_histograms(
        self,
        counts: np.ndarray,
        radial_bin: np.ndarray,
        values: np.ndarray,
        edges: np.ndarray,
    ) -> None:
        value_bin = np.searchsorted(edges, values, side="right") - 1
        value_bin[values == edges[-1]] = len(edges) - 2
        good = (value_bin >= 0) & (value_bin < counts.shape[1])
        if np.any(good):
            np.add.at(counts, (radial_bin[good], value_bin[good]), 1.0)

    def _unit_vectors(self, catalog: pd.DataFrame) -> np.ndarray:
        ra = np.deg2rad(catalog["ra"].to_numpy(float))
        dec = np.deg2rad(catalog["dec"].to_numpy(float))
        cos_dec = np.cos(dec)
        return np.column_stack(
            (
                cos_dec * np.cos(ra),
                cos_dec * np.sin(ra),
                np.sin(dec),
            )
        )

    def _foreground_position_catalog(self) -> Any:
        if self._fg_pos_cat is None:
            self._fg_pos_cat = self._foreground_catalog()
        return self._fg_pos_cat

    def _background_position_catalog(self) -> Any:
        if self._bg_pos_cat is None:
            self._bg_pos_cat = self._background_catalog()
        return self._bg_pos_cat

    def _random_foreground_position_catalog(self) -> Any:
        if self._random_foreground is None or self._random_foreground_da is None:
            raise RuntimeError("Random foreground catalog has not been generated")
        if self._random_fg_pos_cat is None:
            self._random_fg_pos_cat = self._catalog(
                self._random_foreground,
                self._random_foreground_da,
            )
        return self._random_fg_pos_cat

    def _random_background_position_catalog(self) -> Any:
        if self._random_background is None:
            raise RuntimeError("Random background catalog has not been generated")
        if self._random_bg_pos_cat is None:
            self._random_bg_pos_cat = self._catalog(
                self._random_background,
                np.ones(len(self._random_background)),
            )
        return self._random_bg_pos_cat

    def _foreground_catalog(
        self,
        mask: np.ndarray | None = None,
        *,
        w: np.ndarray | None = None,
    ) -> Any:
        catalog = self.foreground if mask is None else self.foreground.loc[mask]
        distance = self._foreground_da if mask is None else self._foreground_da[mask]
        return self._catalog(catalog, distance, w=w)

    def _background_catalog(
        self,
        mask: np.ndarray | None = None,
        *,
        k: np.ndarray | None = None,
        w: np.ndarray | None = None,
    ) -> Any:
        catalog = self.background if mask is None else self.background.loc[mask]
        return self._catalog(catalog, np.ones(len(catalog)), k=k, w=w)

    def _catalog(
        self,
        catalog: pd.DataFrame,
        radial_distance: np.ndarray,
        *,
        k: np.ndarray | None = None,
        w: np.ndarray | None = None,
    ) -> Any:
        radial_distance = np.asarray(radial_distance, dtype=float)
        k = None if k is None else np.asarray(k, dtype=float)
        w = None if w is None else np.asarray(w, dtype=float)
        if getattr(self, "_use_jackknife", False):
            return self._patch_catalogs(catalog, radial_distance, k=k, w=w)
        return treecorr.Catalog(
            **self._catalog_kwargs(catalog, radial_distance, k=k, w=w)
        )

    def _patch_catalogs(
        self,
        catalog: pd.DataFrame,
        radial_distance: np.ndarray,
        *,
        k: np.ndarray | None = None,
        w: np.ndarray | None = None,
    ) -> list[Any]:
        patch = catalog[self._treecorr_patch_col].to_numpy(int)
        catalogs = []
        for patch_id in np.unique(patch):
            use = patch == patch_id
            kwargs = self._catalog_kwargs(
                catalog.loc[use],
                radial_distance[use],
                k=None if k is None else k[use],
                w=None if w is None else w[use],
            )
            kwargs["patch"] = int(patch_id)
            kwargs["npatch"] = self._npatch
            catalogs.append(treecorr.Catalog(**kwargs))
        return catalogs

    def _catalog_kwargs(
        self,
        catalog: pd.DataFrame,
        radial_distance: np.ndarray,
        *,
        k: np.ndarray | None = None,
        w: np.ndarray | None = None,
    ) -> dict[str, Any]:
        kwargs = {
            "ra": catalog["ra"].to_numpy(float),
            "dec": catalog["dec"].to_numpy(float),
            "ra_units": "degrees",
            "dec_units": "degrees",
            "r": radial_distance,
        }
        if k is not None:
            kwargs["k"] = k
        if w is not None:
            kwargs["w"] = w
        return kwargs

    def _radial_bins(self) -> list[_RadialBin]:
        edges = np.asarray(self.r_bin_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2:
            raise ValueError("r_bin_edges must contain at least two edges")
        if np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
            raise ValueError("r_bin_edges must be positive and increasing")
        return [
            _RadialBin(float(lo), float(hi), float(np.sqrt(lo * hi)))
            for lo, hi in zip(edges[:-1], edges[1:])
        ]

    def _reference_bin(self) -> _RadialBin:
        lo, hi = self.reference_annulus
        if lo <= 0 or hi <= lo:
            raise ValueError("reference_annulus must have max > min > 0")
        return _RadialBin(lo, hi, float(np.sqrt(lo * hi)))

    def _nk(self, radial_bin: _RadialBin) -> Any:
        return treecorr.NKCorrelation(**self._corr_kwargs(radial_bin))

    def _nn(self, radial_bin: _RadialBin) -> Any:
        return treecorr.NNCorrelation(**self._corr_kwargs(radial_bin))

    def _corr_kwargs(self, radial_bin: _RadialBin) -> dict[str, Any]:
        return {
            "min_sep": radial_bin.lo,
            "max_sep": radial_bin.hi,
            "nbins": 1,
            "bin_type": "Log",
            "bin_slop": self.bin_slop,
        }

    def _stack_file(self, mode: ColorMode) -> Path:
        return self.out_dir / f"stack_{mode}.npz"

    def _expected_files(self) -> list[Path]:
        return [self._stack_file(mode) for mode in self.modes]

    def _save_config(self) -> None:
        with open(self.out_dir / "config_resolved.yaml", "w") as file:
            yaml.safe_dump(self._to_builtin(dict(self.config)), file, sort_keys=False)

    def _to_builtin(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: self._to_builtin(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_builtin(val) for val in value]
        return value

    def _to_color(
        self, avg: np.ndarray, err: np.ndarray, mode: ColorMode
    ) -> tuple[np.ndarray, np.ndarray]:
        if mode == "mcolors":
            return avg.copy(), err.copy()

        color = np.full_like(avg, np.nan, dtype=float)
        color_err = np.full_like(err, np.nan, dtype=float)
        good = np.isfinite(avg) & np.isfinite(err) & (avg > 0)
        color[good] = -2.5 * np.log10(avg[good])
        color_err[good] = 2.5 / np.log(10.0) * err[good] / avg[good]
        return color, color_err

    def _raw_to_color(self, raw: np.ndarray, mode: ColorMode) -> np.ndarray:
        if mode == "mcolors":
            return raw.copy()

        color = np.full_like(raw, np.nan, dtype=float)
        good = np.isfinite(raw) & (raw > 0)
        color[good] = -2.5 * np.log10(raw[good])
        return color

    def _safe_mean(self, value: float, weight: float) -> float:
        return np.nan if weight == 0 or not np.isfinite(value) else float(value)

    def _bin_centers(self, bins: list[_RadialBin]) -> np.ndarray:
        return np.array([radial_bin.center for radial_bin in bins])


def run_treecorr_stack(
    sample_dir: str | Path,
    output_dir: str | Path,
    stack_config: Mapping[str, Any],
    *,
    footprint_path: str | Path | None = None,
    force: bool = False,
) -> None:
    """Run TreeCorr stacking from prepared sample outputs."""
    stacker = TreeCorrStacker.from_sample_dir(
        sample_dir,
        output_dir,
        stack_config,
        footprint_path=footprint_path,
    )
    stacker.run(force=force)
