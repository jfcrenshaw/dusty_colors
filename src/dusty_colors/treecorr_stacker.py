"""TreeCorr-based color stacking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import healpy as hp
import numpy as np
import pandas as pd
import treecorr
import yaml
from astropy.cosmology import Planck18 as cosmo

from .selector import Default as DefaultSelector
from .selector import Selector

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
class TreeCorrStacker:
    """Stack colors around foreground galaxies using TreeCorr.

    The implemented estimator is the one described in ``paper.tex``:

    ``E(c) = (forward_stack - flipped_stack) - reference_annulus``.

    Forward stacks measure background-galaxy colors around foreground galaxies.
    Flipped stacks measure foreground-galaxy colors over the same foreground-lens
    pair geometry.  In both cases TreeCorr bins pairs by foreground-lens-plane
    impact parameter using ``Rlens`` and foreground ``D_A(z_phot)``.

    When ``random_correction`` is enabled, the estimator subtracts the same
    forward-minus-flipped stack measured with random positions drawn uniformly
    from the selected HEALPix footprint, patch by patch.
    """

    name: str = "treecorr"
    selector: Selector = field(default_factory=DefaultSelector)

    clean_foreground: bool = False
    clean_background: bool = True

    colors: tuple[str, ...] = ("g-r", "r-i", "i-z", "g-i")
    modes: tuple[ColorMode, ...] = ("fcolors", "mcolors")
    flux_type: str = "gaap1p0"
    snr_max: float = 100.0

    r_bin_edges: list[float] | np.ndarray = field(
        default_factory=lambda: np.geomspace(0.005, 1.0, 6)
    )
    r_aper_min: float = 2.0
    r_aper_max: float = 3.0

    bin_slop: float = 0.0
    num_threads: int | None = None
    jackknife: bool = True
    patch_col: str = "jackknife_region"
    cross_patch_weight: str = "match"
    random_correction: bool = True
    random_multiplier: float = 10.0
    random_seed: int = 42
    random_nside: int = 1024

    exclude_jk: int | None = None
    select_jk: int | None = None

    def __post_init__(self) -> None:
        root = Path(__file__).resolve().parents[2]
        self.in_dir = root / f"results/catalogs/{self.selector.name}"
        self.out_dir = root / f"results/stacks/{self.name}"

        self.file_config = self.out_dir / "config_stacker.yaml"
        self.file_stack_fcolors = self.out_dir / "stack_fcolors.npz"
        self.file_stack_mcolors = self.out_dir / "stack_mcolors.npz"

        self._fg_pos_cat: Any | None = None
        self._bg_pos_cat: Any | None = None
        self._random_fg_pos_cat: Any | None = None
        self._random_bg_pos_cat: Any | None = None
        self._footprint_catalog: pd.DataFrame | None = None
        self._treecorr_patch_col = "_treecorr_patch"

    def run(
        self,
        force_selector: bool = False,
        force_stacker: bool = False,
    ) -> None:
        """Run the TreeCorr stacker."""
        self._load_catalogs(force_selector)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._save_config()

        if not force_stacker and all(path.exists() for path in self._expected_files()):
            print(f"TreeCorr stacking already done for variant: {self.name}")
            return

        print("Running TreeCorr stacking for variant:", self.name)
        bins = self._radial_bins()
        ref_bin = self._reference_bin()
        for mode in self.modes:
            self._run_mode(mode, bins, ref_bin)
        print("   TreeCorr stacking complete")

    def _run_mode(
        self, mode: ColorMode, bins: list[_RadialBin], ref_bin: _RadialBin
    ) -> None:
        results = {}

        print(f"   stacking {mode} with TreeCorr...", end="", flush=True)
        for color in self.colors:
            print(f" {color}", end="", flush=True)
            foreground = self._observable(self._foreground, color, mode)
            background = self._observable(self._background, color, mode)

            forward = self._profile(background, bins, ref_bin, mode, "forward")
            flipped = self._profile(foreground, bins, ref_bin, mode, "flipped")
            random_forward = None
            random_flipped = None
            if self.random_correction:
                random_forward = self._profile(
                    background, bins, ref_bin, mode, "forward", random=True
                )
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

        np.savez_compressed(self._stack_file(mode), **results)

    def _load_catalogs(self, force_selector: bool) -> None:
        self.selector.run(force=force_selector)
        self._footprint_catalog = None
        self._foreground = self._read_catalog("foreground", self.clean_foreground)
        self._background = self._read_catalog("background", self.clean_background)

        if self.exclude_jk is not None and self.select_jk is not None:
            raise ValueError("Cannot both exclude and select a jackknife region")
        if self.exclude_jk is not None or self.select_jk is not None:
            region = self.exclude_jk if self.exclude_jk is not None else self.select_jk
            query = (
                f"{self.patch_col} != {region}"
                if self.exclude_jk is not None
                else f"{self.patch_col} == {region}"
            )
            self._foreground = self._foreground.query(query).reset_index(drop=True)
            self._background = self._background.query(query).reset_index(drop=True)

        self._foreground_da = cosmo.angular_diameter_distance(
            self._foreground["z_phot"].to_numpy(float)
        ).value
        self._drop_bad_positions()
        self._setup_jackknife()
        self._setup_randoms()
        self._fg_pos_cat = None
        self._bg_pos_cat = None
        self._random_fg_pos_cat = None
        self._random_bg_pos_cat = None

    def _read_catalog(self, sample: str, cleaned: bool) -> pd.DataFrame:
        if sample == "all":
            return pd.read_parquet(self.in_dir / "galaxy_catalog.parquet").reset_index(
                drop=True
            )
        suffix = "_cleaned" if cleaned else ""
        path = self.in_dir / f"galaxy_catalog_{sample}{suffix}.parquet"
        return pd.read_parquet(path).reset_index(drop=True)

    def _drop_bad_positions(self) -> None:
        fg_good = (
            np.isfinite(self._foreground["coord_ra"].to_numpy(float))
            & np.isfinite(self._foreground["coord_dec"].to_numpy(float))
            & np.isfinite(self._foreground_da)
            & (self._foreground_da > 0)
        )
        bg_good = np.isfinite(
            self._background["coord_ra"].to_numpy(float)
        ) & np.isfinite(self._background["coord_dec"].to_numpy(float))

        self._foreground = self._foreground.loc[fg_good].reset_index(drop=True)
        self._foreground_da = self._foreground_da[fg_good]
        self._background = self._background.loc[bg_good].reset_index(drop=True)

        if len(self._foreground) == 0 or len(self._background) == 0:
            raise ValueError(
                "Foreground and background catalogs must both be non-empty"
            )

    def _read_footprint_catalog(self, label: str, require_patch: bool) -> pd.DataFrame:
        if self._footprint_catalog is None:
            self._footprint_catalog = self._read_catalog("all", cleaned=False)
        if "pixel" not in self._footprint_catalog:
            raise ValueError(f"{label} needs a 'pixel' column in the selected catalog")
        if require_patch and self.patch_col not in self._footprint_catalog:
            raise ValueError(
                f"{label} needs patch column '{self.patch_col}' in the selected catalog"
            )
        return self._footprint_catalog

    def _setup_jackknife(self) -> None:
        self._use_jackknife = False
        self._npatch = 1
        if not self.jackknife:
            self._foreground[self._treecorr_patch_col] = 0
            self._background[self._treecorr_patch_col] = 0
            return

        if (
            self.patch_col not in self._foreground
            or self.patch_col not in self._background
        ):
            raise ValueError(f"Jackknife patch column '{self.patch_col}' not found")

        fg_patch = self._foreground[self.patch_col].to_numpy(int)
        bg_patch = self._background[self.patch_col].to_numpy(int)
        patches = np.intersect1d(np.unique(fg_patch), np.unique(bg_patch))
        if len(patches) < 2:
            raise ValueError(
                "TreeCorr jackknife covariance needs at least two patches shared "
                "by the foreground and background catalogs"
            )

        fg_keep = np.isin(fg_patch, patches)
        bg_keep = np.isin(bg_patch, patches)
        n_fg_drop = int(np.sum(~fg_keep))
        n_bg_drop = int(np.sum(~bg_keep))
        if n_fg_drop or n_bg_drop:
            print(
                "   TreeCorr jackknife using "
                f"{len(patches)} shared patches; dropped "
                f"{n_fg_drop} foreground and {n_bg_drop} background objects "
                "outside the shared patch set"
            )
            self._foreground = self._foreground.loc[fg_keep].reset_index(drop=True)
            self._foreground_da = self._foreground_da[fg_keep]
            self._background = self._background.loc[bg_keep].reset_index(drop=True)

        patch_map = {patch: i for i, patch in enumerate(patches)}
        self._foreground[self._treecorr_patch_col] = [
            patch_map[patch] for patch in self._foreground[self.patch_col].to_numpy(int)
        ]
        self._background[self._treecorr_patch_col] = [
            patch_map[patch] for patch in self._background[self.patch_col].to_numpy(int)
        ]

        self._npatch = len(patches)
        self._use_jackknife = True

    def _setup_randoms(self) -> None:
        self._random_foreground = None
        self._random_background = None
        self._random_foreground_da = None
        if not self.random_correction:
            return
        if self.random_multiplier <= 0:
            raise ValueError("random_multiplier must be positive")

        footprint = self._read_footprint_catalog(
            "Random correction", require_patch=True
        )

        if getattr(self, "_use_jackknife", False):
            patch_map = dict(
                zip(
                    self._foreground[self.patch_col].to_numpy(int),
                    self._foreground[self._treecorr_patch_col].to_numpy(int),
                )
            )
            patch_map.update(
                zip(
                    self._background[self.patch_col].to_numpy(int),
                    self._background[self._treecorr_patch_col].to_numpy(int),
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
            patch: group["pixel"].dropna().to_numpy(int)
            for patch, group in footprint.groupby(self._treecorr_patch_col)
        }
        pixels_by_patch = {
            patch: np.unique(pixels)
            for patch, pixels in pixels_by_patch.items()
            if len(pixels) > 0
        }
        missing = set(self._foreground[self._treecorr_patch_col].unique()) - set(
            pixels_by_patch
        )
        missing |= set(self._background[self._treecorr_patch_col].unique()) - set(
            pixels_by_patch
        )
        if missing:
            raise ValueError(f"Missing random-footprint pixels for patches: {missing}")

        rng = np.random.default_rng(self.random_seed)
        self._random_foreground, self._random_foreground_da = self._random_catalog_like(
            self._foreground,
            self._foreground_da,
            pixels_by_patch,
            rng,
        )
        self._random_background, _ = self._random_catalog_like(
            self._background,
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
                        "coord_ra": ra,
                        "coord_dec": dec,
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

        return (
            np.concatenate(ras)[:n_random],
            np.concatenate(decs)[:n_random],
        )

    def _observable(
        self, catalog: pd.DataFrame, color: str, mode: ColorMode
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if mode == "mcolors":
            value = catalog[color].to_numpy(float)
            err = catalog[f"{color}_Err"].to_numpy(float)
            floor = np.sqrt(2.0) * 2.5 / np.log(10.0) / self.snr_max
        else:
            band1, band2 = color.split("-")
            flux1 = catalog[f"{band1}_{self.flux_type}Flux"].to_numpy(float)
            flux2 = catalog[f"{band2}_{self.flux_type}Flux"].to_numpy(float)
            err1 = catalog[f"{band1}_{self.flux_type}FluxErr"].to_numpy(float)
            err2 = catalog[f"{band2}_{self.flux_type}FluxErr"].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                value = flux1 / flux2
                err = np.abs(value) * np.sqrt((err1 / flux1) ** 2 + (err2 / flux2) ** 2)
            floor = np.abs(value) * np.sqrt(2.0) / self.snr_max

        err = np.clip(err, floor, None)
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
        flipped: _Profile,
        random_forward: _Profile | None = None,
        random_flipped: _Profile | None = None,
    ) -> dict[str, np.ndarray]:
        has_random = random_forward is not None or random_flipped is not None
        if has_random and (random_forward is None or random_flipped is None):
            raise ValueError("Random correction requires both random profiles")

        raw_delta = forward.color - flipped.color
        raw_delta_err = np.hypot(forward.color_err, flipped.color_err)
        raw_ref = forward.ref_color - flipped.ref_color
        raw_ref_err = np.hypot(forward.ref_color_err, flipped.ref_color_err)

        random_delta = np.zeros_like(raw_delta)
        random_delta_err = np.zeros_like(raw_delta_err)
        random_ref = 0.0
        random_ref_err = 0.0
        if has_random:
            random_delta = random_forward.color - random_flipped.color
            random_delta_err = np.hypot(
                random_forward.color_err, random_flipped.color_err
            )
            random_ref = random_forward.ref_color - random_flipped.ref_color
            random_ref_err = np.hypot(
                random_forward.ref_color_err,
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
            len(bins),
            random_forward=random_forward,
            random_flipped=random_flipped,
        )
        covariance = (
            jackknife.covariance if jackknife is not None else np.diag(analytic_err**2)
        )
        estimator_err = jackknife.err if jackknife is not None else analytic_err

        result = {
            f"{color}_bin_centers": self._bin_centers(bins),
            f"{color}_avg": estimator,
            f"{color}_err": estimator_err,
            f"{color}_cov": covariance,
            f"{color}_analytic_err": analytic_err,
            f"{color}_delta_avg": delta,
            f"{color}_delta_err": delta_err,
            f"{color}_raw_delta_avg": raw_delta,
            f"{color}_raw_delta_err": raw_delta_err,
            f"{color}_ref_avg": np.array(ref),
            f"{color}_ref_err": np.array(ref_err),
            f"{color}_raw_ref_avg": np.array(raw_ref),
            f"{color}_raw_ref_err": np.array(raw_ref_err),
            f"{color}_uncorrected_avg": raw_delta - raw_ref,
            f"{color}_forward_avg": forward.color,
            f"{color}_forward_err": forward.color_err,
            f"{color}_flipped_avg": flipped.color,
            f"{color}_flipped_err": flipped.color_err,
            f"{color}_forward_raw_avg": forward.raw,
            f"{color}_forward_raw_err": forward.raw_err,
            f"{color}_flipped_raw_avg": flipped.raw,
            f"{color}_flipped_raw_err": flipped.raw_err,
            f"{color}_forward_npairs": forward.npairs,
            f"{color}_flipped_npairs": flipped.npairs,
        }
        if has_random:
            result.update(
                {
                    f"{color}_random_delta_avg": random_delta,
                    f"{color}_random_delta_err": random_delta_err,
                    f"{color}_random_ref_avg": np.array(random_ref),
                    f"{color}_random_ref_err": np.array(random_ref_err),
                    f"{color}_random_forward_avg": random_forward.color,
                    f"{color}_random_forward_err": random_forward.color_err,
                    f"{color}_random_flipped_avg": random_flipped.color,
                    f"{color}_random_flipped_err": random_flipped.color_err,
                    f"{color}_random_forward_raw_avg": random_forward.raw,
                    f"{color}_random_forward_raw_err": random_forward.raw_err,
                    f"{color}_random_flipped_raw_avg": random_flipped.raw,
                    f"{color}_random_flipped_raw_err": random_flipped.raw_err,
                    f"{color}_random_forward_npairs": random_forward.npairs,
                    f"{color}_random_flipped_npairs": random_flipped.npairs,
                }
            )
        if jackknife is not None:
            result.update(
                {
                    f"{color}_jackknife_avg": jackknife.mean,
                    f"{color}_jackknife_err": jackknife.err,
                    f"{color}_jackknife_samples": jackknife.samples,
                    f"{color}_jackknife_patch": np.arange(jackknife.samples.shape[0]),
                }
            )
        return result

    def _jackknife_stats(
        self,
        mode: ColorMode,
        forward: _Profile,
        flipped: _Profile,
        n_bins: int,
        random_forward: _Profile | None = None,
        random_flipped: _Profile | None = None,
    ) -> _JackknifeStats | None:
        if not getattr(self, "_use_jackknife", False):
            return None
        has_random = random_forward is not None or random_flipped is not None
        if has_random and (random_forward is None or random_flipped is None):
            raise ValueError("Random correction requires both random profiles")

        all_profile_corrs = forward.corrs + flipped.corrs
        if has_random:
            all_profile_corrs += random_forward.corrs + random_flipped.corrs
        if any(len(corrs) == 0 for corrs in all_profile_corrs):
            return None

        corrs = []
        iterator = zip(forward.corrs, flipped.corrs)
        if has_random:
            iterator = zip(
                forward.corrs,
                flipped.corrs,
                random_forward.corrs,
                random_flipped.corrs,
            )
        for profile_corrs in iterator:
            forward_corrs = profile_corrs[0]
            flipped_corrs = profile_corrs[1]
            corrs.extend(forward_corrs)
            corrs.extend(flipped_corrs)
            if has_random:
                corrs.extend(profile_corrs[2])
                corrs.extend(profile_corrs[3])

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

                value, i = read_flipped(corrs, i)
                raw_flipped.append(value)

                if has_random:
                    value, i = read_forward(corrs, i)
                    raw_random_forward.append(value)

                    value, i = read_flipped(corrs, i)
                    raw_random_flipped.append(value)

            forward_color = self._raw_to_color(np.array(raw_forward), mode)
            flipped_color = self._raw_to_color(np.array(raw_flipped), mode)
            delta = forward_color - flipped_color
            if has_random:
                random_forward_color = self._raw_to_color(
                    np.array(raw_random_forward), mode
                )
                random_flipped_color = self._raw_to_color(
                    np.array(raw_random_flipped), mode
                )
                delta -= random_forward_color - random_flipped_color
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
        catalog = self._foreground if mask is None else self._foreground.loc[mask]
        distance = self._foreground_da if mask is None else self._foreground_da[mask]
        return self._catalog(catalog, distance, w=w)

    def _background_catalog(
        self,
        mask: np.ndarray | None = None,
        *,
        k: np.ndarray | None = None,
        w: np.ndarray | None = None,
    ) -> Any:
        catalog = self._background if mask is None else self._background.loc[mask]
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
            "ra": catalog["coord_ra"].to_numpy(float),
            "dec": catalog["coord_dec"].to_numpy(float),
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
        if self.r_aper_min <= 0 or self.r_aper_max <= self.r_aper_min:
            raise ValueError("Reference annulus must have r_aper_max > r_aper_min > 0")
        return _RadialBin(
            self.r_aper_min,
            self.r_aper_max,
            float(np.sqrt(self.r_aper_min * self.r_aper_max)),
        )

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
        config = asdict(self)
        config["selector"] = asdict(self.selector)
        with open(self.file_config, "w") as file:
            yaml.safe_dump(self._to_builtin(config), file, sort_keys=False)

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


TreeCorrDefault = TreeCorrStacker
