"""TreeCorr-based color stacking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

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
        default_factory=lambda: np.geomspace(5.0e-3, 1.5, 7)
    )
    r_aper_min: float = 2.0
    r_aper_max: float = 4.0

    bin_slop: float = 0.0
    num_threads: int | None = None
    jackknife: bool = True
    patch_col: str = "jackknife_region"
    cross_patch_weight: str = "match"

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

            results.update(self._result(color, bins, mode, forward, flipped))
        print(".")

        np.savez_compressed(self._stack_file(mode), **results)

    def _load_catalogs(self, force_selector: bool) -> None:
        self.selector.run(force=force_selector)
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
        self._fg_pos_cat = None
        self._bg_pos_cat = None

    def _read_catalog(self, sample: str, cleaned: bool) -> pd.DataFrame:
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

    def _setup_jackknife(self) -> None:
        self._use_jackknife = False
        self._npatch = 1
        if not self.jackknife:
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
    ) -> _Profile:
        value, err, good = observable
        bin_results = [
            self._stack_bin(value, err, good, b, direction) for b in bins + [ref_bin]
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
    ) -> _BinResult:
        mean, count_weight, npairs, _ = self._pair_mean(
            value, good, radial_bin, direction
        )
        mean2, _, _, _ = self._pair_mean(value**2, good, radial_bin, direction)
        if count_weight == 0 or not np.isfinite(mean) or not np.isfinite(mean2):
            return _BinResult(np.nan, np.nan, 0.0, npairs, ())

        variance_floor = max(mean2 - mean**2, 0.0)
        weight = np.zeros_like(err)
        weight[good] = 1.0 / (err[good] ** 2 + variance_floor)

        avg, weight_sum, npairs, corrs = self._pair_mean(
            value, good, radial_bin, direction, weight
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
    ) -> tuple[float, float, float, tuple[Any, ...]]:
        if not np.any(good):
            return np.nan, 0.0, 0.0, ()

        if direction == "forward":
            source = self._background_catalog(
                good, k=value[good], w=None if weight is None else weight[good]
            )
            corr = self._nk(radial_bin)
            corr.process(
                self._foreground_position_catalog(),
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
            self._background_position_catalog(),
            metric="Rlens",
            num_threads=self.num_threads,
        )
        if denominator.weight[0] == 0:
            return np.nan, 0.0, denominator.npairs[0], ()

        numerator = self._nn(radial_bin)
        numerator.process(
            self._foreground_catalog(good, w=pair_weight * value[good]),
            self._background_position_catalog(),
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
    ) -> dict[str, np.ndarray]:
        delta = forward.color - flipped.color
        delta_err = np.hypot(forward.color_err, flipped.color_err)
        ref = forward.ref_color - flipped.ref_color
        ref_err = np.hypot(forward.ref_color_err, flipped.ref_color_err)
        estimator = delta - ref
        analytic_err = np.hypot(delta_err, ref_err)
        jackknife = self._jackknife_stats(mode, forward, flipped, len(bins))
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
            f"{color}_ref_avg": np.array(ref),
            f"{color}_ref_err": np.array(ref_err),
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
    ) -> _JackknifeStats | None:
        if not getattr(self, "_use_jackknife", False):
            return None
        if any(len(corrs) == 0 for corrs in forward.corrs + flipped.corrs):
            return None

        corrs = []
        for forward_corrs, flipped_corrs in zip(forward.corrs, flipped.corrs):
            corrs.extend(forward_corrs)
            corrs.extend(flipped_corrs)

        def estimator(corrs: list[Any]) -> np.ndarray:
            raw_forward = []
            raw_flipped = []
            i = 0
            for _ in range(n_bins + 1):
                raw_forward.append(corrs[i].xi[0])
                i += 1

                numerator = corrs[i]
                denominator = corrs[i + 1]
                i += 2
                if denominator.weight[0] == 0:
                    raw_flipped.append(np.nan)
                else:
                    raw_flipped.append(numerator.weight[0] / denominator.weight[0])

            forward_color = self._raw_to_color(np.array(raw_forward), mode)
            flipped_color = self._raw_to_color(np.array(raw_flipped), mode)
            delta = forward_color[:-1] - flipped_color[:-1]
            ref = forward_color[-1] - flipped_color[-1]
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
