"""Class to perform stacking."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.cosmology import Planck18 as cosmo
from scipy.spatial import cKDTree

from .selector import Default as DefaultSelector
from .selector import Selector
from .utils import plot_stack

import matplotlib.pyplot as plt


@dataclass
class Stacker:
    """Class to define variants for stacking."""

    name: str = "default"

    selector: Selector = field(default_factory=DefaultSelector)

    # Use cleaned catalogs?
    clean_foreground: bool = False
    clean_background: bool = True

    # Stacking options
    free_fluxes: bool = False  # Whether to use "free" flux variants
    snr_max: float = 100  # Maximum SNR (sets error floor)
    weighted: bool = False  # Whether to use weighted averages in bins
    r_aper_min: float = 2.5  # Inner aperture radius (Mpc) for background normalization
    r_aper_max: float = 3.5  # Outer aperture radius (Mpc) for background normalization

    # Defining bins
    bin_by_angle: bool = False  # If false, bin by physical impact parameter
    r_bins: list | np.ndarray = field(default_factory=lambda: np.geomspace(2e-2, 1, 5))

    # For null tests
    randomize_positions: bool = False  # Whether to randomize positions
    random_seed: int = 42  # Random seed for position randomization

    # How to calculate errors
    bootstrap: bool = False  # If false, use analytic errors

    # Plotting settings
    r_norm: float = 0.8  # Radius (Mpc or arcmin) beyond which is used to normalize

    # Exclude (or limit to) a jackknife region
    exclude_jk: int | None = None
    select_jk: int | None = None

    def __post_init__(self) -> None:
        """Post-init processing."""
        # Directories and files
        current_file = Path(__file__).resolve()
        root = current_file.parents[2]
        self.in_dir = root / f"results/catalogs/{self.selector.name}"
        self.out_dir = root / f"results/stacks/{self.name}"

        self.file_config = self.out_dir / "config_stacker.yaml"

        self.file_pairs = self.out_dir / "pairs.npz"
        self.file_pairs_flipped = self.out_dir / "pairs_flipped.npz"

        self.file_stack_fluxes = self.out_dir / "stack_fluxes.npz"
        self.file_stack_fluxes_flipped = self.out_dir / "stack_fluxes_flipped.npz"
        self.file_stack_fluxes_fig = self.out_dir / "fig_flux_stack.pdf"

        self.file_stack_mags = self.out_dir / "stack_mags.npz"
        self.file_stack_mags_flipped = self.out_dir / "stack_mags_flipped.npz"
        self.file_stack_mags_fig = self.out_dir / "fig_mag_stack.pdf"

        self.file_stack_fcolors = self.out_dir / "stack_fcolors.npz"
        self.file_stack_fcolors_flipped = self.out_dir / "stack_fcolors_flipped.npz"
        self.file_stack_fcolors_fig = self.out_dir / "fig_fcolors_stack.pdf"

        self.file_stack_mcolors = self.out_dir / "stack_mcolors.npz"
        self.file_stack_mcolors_flipped = self.out_dir / "stack_mcolors_flipped.npz"
        self.file_stack_mcolors_fig = self.out_dir / "fig_mcolors_stack.pdf"

        self.file_stack_shear = self.out_dir / "stack_shear.npz"
        self.file_stack_shear_flipped = self.out_dir / "stack_shear_flipped.npz"
        self.file_stack_shear_fig = self.out_dir / "fig_shear_stack.pdf"

        self._flipped = False

    def _save_config(self) -> None:
        with open(self.file_config, "w") as file:
            config = asdict(self)
            config["selector"] = asdict(self.selector)
            yaml.dump(config, file, sort_keys=False)

    def _get_bins(self) -> tuple[np.ndarray, np.ndarray]:
        """Get bin edges."""
        # Create densely-packed bins that are centered on the specified r_bins
        centers = np.array(self.r_bins)
        edges = [centers[0] - (centers[1] - centers[0]) / 2]
        for i in range(len(centers)):
            edges.append(centers[i] + (centers[i] - edges[i]))
        edges = np.array(edges)

        return centers, edges

    def _find_pairs(self, force: bool = False) -> None:
        """Find foreground-background pairs."""
        # Skip time-intensive pair finding if already done
        if self.file_pairs.exists() and not force:
            print(f"   pairs already found, loading from {self.file_pairs}")
            data = np.load(self.file_pairs)
            self._pairs = data["pairs"]
            self._separation = data["separation"]
            print(f"   {len(self._pairs)} pairs loaded")
            return

        # Otherwise, move forward with finding new pairs
        print("   finding pairs...", end=" ", flush=True)

        if self.randomize_positions:
            rng = np.random.default_rng(self.random_seed)

            # Foreground positions
            R = np.sqrt(rng.uniform(size=len(self._foreground)))
            theta = rng.uniform(0, 2 * np.pi, size=len(R))
            fg_ra, fg_dec = R * np.cos(theta), R * np.sin(theta)

            # Background positions
            R = np.sqrt(rng.uniform(size=len(self._background)))
            theta = rng.uniform(0, 2 * np.pi, size=len(R))
            bg_ra, bg_dec = R * np.cos(theta), R * np.sin(theta)
        else:
            fg_ra = self._foreground["coord_ra"]
            fg_dec = self._foreground["coord_dec"]
            bg_ra = self._background["coord_ra"]
            bg_dec = self._background["coord_dec"]

        # Flat-sky approximation
        fg_xy = np.column_stack((fg_ra * np.cos(np.deg2rad(fg_dec)), fg_dec))
        bg_xy = np.column_stack((bg_ra * np.cos(np.deg2rad(bg_dec)), bg_dec))

        # Determine max distance to consider
        bin_centers, bin_edges = self._get_bins()
        # Extend search to aperture radius so aperture averages can be computed
        r_max = self.r_aper_max if not self.bin_by_angle else bin_edges[-1]

        # Build spatial KDTree on background
        bg_tree = cKDTree(bg_xy)

        # Determine max angular separation for each foreground galaxy
        if not self.bin_by_angle:
            # Calculate angular diameter distances for all foreground galaxies
            dA = cosmo.angular_diameter_distance(self._foreground["z_phot"]).value

            # Compute max separation in degrees for each foreground galaxy
            max_sep_deg = np.rad2deg(r_max / dA)
        else:
            # If binning by angle, max sep. same for all foreground galaxies
            max_sep_deg = np.full(len(fg_xy), r_max / 60)  # arcmin to deg

        pairs = []
        separation = []

        # Loop over foreground galaxies and find all background matches
        for i_fg, (coord, sep_deg) in enumerate(zip(fg_xy, max_sep_deg)):
            # Matches within max separation
            matches = bg_tree.query_ball_point(coord, r=sep_deg)

            # Save for each pair found
            for i_bg in matches:
                # Save pair indices
                pairs.append((i_fg, i_bg))

                # Calculate angular separation
                dx = coord[0] - bg_xy[i_bg, 0]
                dy = coord[1] - bg_xy[i_bg, 1]
                theta = np.sqrt(dx**2 + dy**2)

                # Save separation in arcmin or Mpc
                if self.bin_by_angle:
                    separation.append(theta * 60)  # deg to arcmin
                else:
                    separation.append(dA[i_fg] * np.deg2rad(theta))  # Mpc

        # Save results
        self._pairs = np.array(pairs)
        self._separation = np.array(separation)
        np.savez_compressed(
            self.file_pairs, pairs=self._pairs, separation=self._separation
        )
        print(f"{len(self._pairs)} found")

    def _calc_binned_stats(
        self,
        sep: np.ndarray,
        x: np.ndarray,
        err: np.ndarray,
        bin_edges: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate binned statistics."""
        # Assign bins
        n_bins = len(bin_edges) - 1
        bin_indices = np.digitize(sep, bin_edges) - 1
        valid = (bin_indices >= 0) & (bin_indices < n_bins)
        bin_indices = bin_indices[valid]
        x = x[valid]
        err = err[valid]

        # Calculate weighted average in each bin
        avgs = []
        errs = []
        for i in range(n_bins):
            # Find values in bin
            in_bin = bin_indices == i

            # If bin is empty, append NaN
            if in_bin.sum() == 0:
                avgs.append(np.nan)
                errs.append(np.nan)
                continue

            # Calculate weights
            xi = x[in_bin]
            ei = err[in_bin]
            if self.weighted:
                wi = 1 / (ei**2 + xi.var())
            else:
                wi = np.ones_like(xi)

            # Weighted average and error
            if self.bootstrap:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(xi), size=(100, len(xi)), replace=True)
                samples = np.average(xi[idx], weights=wi[idx], axis=1)
                avgs.append(samples.mean())
                errs.append(samples.std())
            else:
                avgs.append(np.average(xi, weights=wi))
                errs.append(np.sum(wi**2 * (ei**2 + xi.var())) / wi.sum() ** 2)

        return np.array(avgs), np.array(errs)

    def _compute_aperture_avg(
        self,
        x: np.ndarray,
        err: np.ndarray,
        i_source: np.ndarray,
        valid: np.ndarray,
        n_source: int,
        sep: np.ndarray,
    ) -> np.ndarray:
        """Mean (or weighted mean) observable per source galaxy across valid pairs in the aperture annulus."""
        aper_mask = (sep >= self.r_aper_min) & (sep <= self.r_aper_max)
        combined = valid & aper_mask
        valid_i = i_source[combined]
        valid_x = x[combined]
        valid_err = err[combined]

        count = np.bincount(valid_i, minlength=n_source).astype(float)
        has_pairs = count > 0
        x_aper = np.full(n_source, np.nan)

        if not self.weighted:
            x_sum = np.bincount(valid_i, weights=valid_x, minlength=n_source)
            x_aper[has_pairs] = x_sum[has_pairs] / count[has_pairs]
        else:
            # Variance of x per source galaxy (population variance, matching _calc_binned_stats)
            x_sum = np.bincount(valid_i, weights=valid_x, minlength=n_source)
            x_sq_sum = np.bincount(valid_i, weights=valid_x**2, minlength=n_source)
            x_mean = np.zeros(n_source)
            x_sq_mean = np.zeros(n_source)
            x_mean[has_pairs] = x_sum[has_pairs] / count[has_pairs]
            x_sq_mean[has_pairs] = x_sq_sum[has_pairs] / count[has_pairs]
            # Clamp to zero to guard against floating-point cancellation
            x_var = np.maximum(x_sq_mean - x_mean**2, 0.0)

            # Per-pair weights: 1 / (ei^2 + var of x around this source)
            w = 1.0 / (valid_err**2 + x_var[valid_i])
            wx_sum = np.bincount(valid_i, weights=w * valid_x, minlength=n_source)
            w_sum = np.bincount(valid_i, weights=w, minlength=n_source)
            x_aper[has_pairs] = wx_sum[has_pairs] / w_sum[has_pairs]

        return x_aper

    def _footprint_mask(self, flipped: bool) -> np.ndarray:
        """Boolean mask over pairs keeping only those whose lens is fully inside the footprint."""
        r_cut = 20  # 20 - 3.33 * self.r_aper
        if not flipped:
            r_center = self._foreground["r_center"].values
            idx = self._pairs[:, 0]
        else:
            r_center = self._background["r_center"].values
            idx = self._pairs[:, 1]
        return r_center[idx] < r_cut

    def _stack_flux_or_mag(self, flux: bool, flipped: bool) -> dict:
        """Perform stacking of either fluxes or magnitudes."""
        # Container for results
        results = {}

        # Restrict to pairs whose lens galaxy has a complete aperture inside the footprint
        fp_mask = self._footprint_mask(flipped)
        pairs = self._pairs[fp_mask]
        separation = self._separation[fp_mask]

        # Loop over bands
        for band in "griz":
            print(f" {band}", end="", flush=True)

            # Set bins
            bin_centers, bin_edges = self._get_bins()
            results[f"{band}_bin_centers"] = bin_centers

            # Get columns
            col = f"{band}_{'free_' if self.free_fluxes else ''}cModel"
            if flux:
                col += "Flux"
            else:
                col += "Mag"
            if not flipped:
                x = self._background[col].iloc[pairs[:, 1]].values
                err = self._background[col + "Err"].iloc[pairs[:, 1]].values
                i_source = pairs[:, 0]
                n_source = len(self._foreground)
            else:
                x = self._foreground[col].iloc[pairs[:, 0]].values
                err = self._foreground[col + "Err"].iloc[pairs[:, 0]].values
                i_source = pairs[:, 1]
                n_source = len(self._background)

            # Identify valid pairs
            mask = np.isfinite(x) & np.isfinite(err) & (err > 0)

            # Apply SNR maximum (only to err; x is unchanged for aperture averaging)
            if flux:
                err = np.where(
                    mask, x * np.clip(err / x, 1 / self.snr_max, None), np.nan
                )
            else:
                err = np.where(
                    mask, np.clip(err, 2.5 / np.log(10) / self.snr_max, None), np.nan
                )

            # Normalize each pair's observable by the aperture mean of its source galaxy.
            # x_aper[i] = mean observable of all valid background galaxies within the
            # aperture annulus [r_aper_min, r_aper_max] of foreground galaxy i.
            if not self.bin_by_angle:
                x_aper = self._compute_aperture_avg(
                    x, err, i_source, mask, n_source, separation
                )
                x_aper_per_pair = x_aper[i_source]
                if flux:
                    x = x / x_aper_per_pair
                    err = err / np.abs(x_aper_per_pair)
                else:
                    x = x - x_aper_per_pair
                    # err is unchanged for a difference

            # Filter after normalization (NaN x_aper propagates to x/err here)
            final_mask = np.isfinite(x) & np.isfinite(err) & (err > 0)
            x = x[final_mask]
            err = err[final_mask]
            sep = separation[final_mask]

            # Calculate binned stats
            avgs, errs = self._calc_binned_stats(sep, x, err, bin_edges)
            results[f"{band}_avg"] = avgs
            results[f"{band}_err"] = errs

        return results

    def _stack_fluxes(self, force: bool = False, flipped: bool = False) -> None:
        """Stack fluxes."""
        # Skip time-intensive stacking if already done
        file = self.file_stack_fluxes_flipped if flipped else self.file_stack_fluxes
        if file.exists() and not force:
            print("   fluxes already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking fluxes...", end="", flush=True)
        results = self._stack_flux_or_mag(flux=True, flipped=flipped)

        # Save all results
        np.savez_compressed(file, **results)  # type: ignore
        print(".")

        self._new_flux_stack = True

    def _stack_mags(self, force: bool = False, flipped: bool = False) -> None:
        """Stack magnitudes."""
        # Skip time-intensive stacking if already done
        file = self.file_stack_mags_flipped if flipped else self.file_stack_mags
        if file.exists() and not force:
            print("   mags already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking mags...", end="", flush=True)
        results = self._stack_flux_or_mag(flux=False, flipped=flipped)

        # Save all results
        np.savez_compressed(file, **results)  # type: ignore
        print(".")

        self._new_mags_stack = True

    def _stack_colors(self, flux: bool, flipped: bool) -> dict:
        """Perform stacking of either flux ratios or color differences."""
        # Container for results
        results = {}

        # Restrict to pairs whose lens galaxy has a complete aperture inside the footprint
        fp_mask = self._footprint_mask(flipped)
        pairs = self._pairs[fp_mask]
        separation = self._separation[fp_mask]

        # Loop over bands
        for color in ["g-r", "r-i", "i-z", "g-i"]:
            print(f" {color}", end="", flush=True)

            # Set bins
            band1, band2 = color.split("-")
            bin_centers, bin_edges = self._get_bins()
            results[f"{color}_bin_centers"] = bin_centers

            # Source galaxy indices for aperture averaging
            if not flipped:
                i_source = pairs[:, 0]
                n_source = len(self._foreground)
            else:
                i_source = pairs[:, 1]
                n_source = len(self._background)

            if flux:
                # Retrieve columns
                if self.free_fluxes:
                    col1 = f"{band1}_free_cModel"
                    col2 = f"{band2}_free_cModel"
                else:
                    col1 = f"{band1}_gaap1p0"
                    col2 = f"{band2}_gaap1p0"
                col1 += "Flux"
                col2 += "Flux"

                if not flipped:
                    x1 = self._background[col1].iloc[pairs[:, 1]].values
                    err1 = self._background[col1 + "Err"].iloc[pairs[:, 1]].values
                    x2 = self._background[col2].iloc[pairs[:, 1]].values
                    err2 = self._background[col2 + "Err"].iloc[pairs[:, 1]].values
                else:
                    x1 = self._foreground[col1].iloc[pairs[:, 0]].values
                    err1 = self._foreground[col1 + "Err"].iloc[pairs[:, 0]].values
                    x2 = self._foreground[col2].iloc[pairs[:, 0]].values
                    err2 = self._foreground[col2 + "Err"].iloc[pairs[:, 0]].values

                # Apply SNR maximum
                err1 = x1 * np.clip(err1 / x1, 1 / self.snr_max, None)
                err2 = x2 * np.clip(err2 / x2, 1 / self.snr_max, None)

                # Calculate flux ratio and its error
                x = x1 / x2
                err = x * np.sqrt((err1 / x1) ** 2 + (err2 / x2) ** 2)

            else:
                # Retrieve pre-computed color columns
                if not flipped:
                    x = self._background[f"{color}"].iloc[pairs[:, 1]].values
                    err = self._background[f"{color}_Err"].iloc[pairs[:, 1]].values
                else:
                    x = self._foreground[f"{color}"].iloc[pairs[:, 0]].values
                    err = self._foreground[f"{color}_Err"].iloc[pairs[:, 0]].values

                # Apply SNR maximum
                err = np.clip(err, np.sqrt(2) * 2.5 / np.log(10) / self.snr_max, None)

            # Identify valid pairs
            mask = np.isfinite(x) & np.isfinite(err) & (err > 0)

            # Normalize each pair's color by its source galaxy's aperture mean color.
            # For flux: stack (f1/f2) / mean(f1/f2 in aperture).
            # For mags: stack (m1-m2) - mean(m1-m2 in aperture).
            if not self.bin_by_angle:
                x_aper = self._compute_aperture_avg(
                    x, err, i_source, mask, n_source, separation
                )
                x_aper_per_pair = x_aper[i_source]
                if flux:
                    x = x / x_aper_per_pair
                    err = err / np.abs(x_aper_per_pair)
                else:
                    x = x - x_aper_per_pair
                    # err is unchanged for a difference

            # Filter after normalization
            final_mask = np.isfinite(x) & np.isfinite(err) & (err > 0)
            x = x[final_mask]
            err = err[final_mask]
            sep = separation[final_mask]

            # Calculate binned stats
            avgs, errs = self._calc_binned_stats(sep, x, err, bin_edges)
            results[f"{color}_avg"] = avgs
            results[f"{color}_err"] = errs

        return results

    def _stack_fcolors(self, force: bool = False, flipped: bool = False) -> None:
        """Stack colors using flux ratios."""
        # Skip time-intensive stacking if already done
        file = self.file_stack_fcolors_flipped if flipped else self.file_stack_fcolors
        if file.exists() and not force:
            print("   flux-colors already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking flux-colors...", end="", flush=True)
        results = self._stack_colors(flux=True, flipped=flipped)

        # Save all results
        np.savez_compressed(file, **results)  # type: ignore
        print(".")

        self._new_fcolors_stack = True

    def _stack_mcolors(self, force: bool = False, flipped: bool = False) -> None:
        """Stack colors using magnitude differences."""
        # Skip time-intensive stacking if already done
        file = self.file_stack_mcolors_flipped if flipped else self.file_stack_mcolors
        if file.exists() and not force:
            print("   magnitude-colors already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking magnitude-colors...", end="", flush=True)
        results = self._stack_colors(flux=False, flipped=flipped)

        # Save all results
        np.savez_compressed(file, **results)  # type: ignore
        print(".")

        self._new_mcolors_stack = True

    def _stack_shear(self, force: bool = False, flipped: bool = False) -> None:
        """Stack shear."""
        raise NotImplementedError("Shear stacking not yet implemented.")

    def _create_figure(self, stack_type: str, force: bool = False) -> None:
        """Create figure for given stack type."""
        if stack_type == "fluxes":
            if (
                not force
                and not self._new_flux_stack
                and self.file_stack_fluxes_fig.exists()
            ):
                return
            fig_file = self.file_stack_fluxes_fig
        elif stack_type == "mags":
            if (
                not force
                and not self._new_mags_stack
                and self.file_stack_mags_fig.exists()
            ):
                return
            fig_file = self.file_stack_mags_fig
        elif stack_type == "fcolors":
            if (
                not force
                and not self._new_fcolors_stack
                and self.file_stack_fcolors_fig.exists()
            ):
                return
            fig_file = self.file_stack_fcolors_fig
        elif stack_type == "mcolors":
            if (
                not force
                and not self._new_mcolors_stack
                and self.file_stack_mcolors_fig.exists()
            ):
                return
            fig_file = self.file_stack_mcolors_fig
        elif stack_type == "shear":
            if (
                not force
                and not self._new_shear_stack
                and self.file_stack_shear_fig.exists()
            ):
                return
            fig_file = self.file_stack_shear_fig
        else:
            raise ValueError(f"Unknown stack type: {stack_type}")

        fig = plot_stack(self.out_dir, stack_type=stack_type, r_norm=self.r_norm)
        fig.savefig(fig_file, bbox_inches="tight")
        print(f"   saved figure to {fig_file}")
        plt.close(fig)

    def _run_stacking(
        self,
        force_stacker: bool = False,
        force_pairer: bool = False,
        flipped: bool = False,
    ) -> None:
        self._save_config()
        self._find_pairs(force=force_pairer)
        if len(self._pairs) == 0:
            print("   no pairs found, skipping stacking")
            return
        self._stack_fluxes(force=force_stacker, flipped=flipped)
        self._stack_mags(force=force_stacker, flipped=flipped)
        self._stack_fcolors(force=force_stacker, flipped=flipped)
        self._stack_mcolors(force=force_stacker, flipped=flipped)
        # self._stack_shear()

        print("   stacking complete")

    def run(
        self,
        force_selector: bool = False,
        force_pairer: bool = False,
        force_stacker: bool = False,
        force_plotter: bool = False,
    ) -> None:
        # Run selector
        self.selector.run(force=force_selector)

        # Set foreground and background samples
        if self.clean_foreground:
            self._foreground: pd.DataFrame = pd.read_parquet(
                self.in_dir / "galaxy_catalog_foreground_cleaned.parquet"
            )
        else:
            self._foreground: pd.DataFrame = pd.read_parquet(
                self.in_dir / "galaxy_catalog_foreground.parquet"
            )
        if self.clean_background:
            self._background: pd.DataFrame = pd.read_parquet(
                self.in_dir / "galaxy_catalog_background_cleaned.parquet"
            )
        else:
            self._background: pd.DataFrame = pd.read_parquet(
                self.in_dir / "galaxy_catalog_background.parquet"
            )
        if self.exclude_jk is not None and self.select_jk is not None:
            raise ValueError("Cannot both exclude and select a jackknife region")
        if self.exclude_jk is not None or self.select_jk is not None:
            region = self.exclude_jk if self.exclude_jk is not None else self.select_jk
            if region not in self._foreground["jackknife_region"].unique():
                raise ValueError(
                    f"Jackknife region {region} not found in foreground sample"
                )
            if region not in self._background["jackknife_region"].unique():
                raise ValueError(
                    f"Jackknife region {region} not found in background sample"
                )
            query = (
                f"jackknife_region != {region}"
                if self.exclude_jk is not None
                else f"jackknife_region == {region}"
            )
            self._foreground = self._foreground.query(query)
            self._background = self._background.query(query)

        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Flags for new stacks
        self._new_flux_stack = False
        self._new_mags_stack = False
        self._new_fcolors_stack = False
        self._new_mcolors_stack = False
        self._new_shear_stack = False

        # Check for expected output files and run unflipped
        if (
            force_stacker
            or not self.file_stack_fluxes.exists()
            or not self.file_stack_mags.exists()
            or not self.file_stack_fcolors.exists()
            or not self.file_stack_mcolors.exists()
            # or not self.file_stack_shear.exists()
        ):
            print("Running stacking for variant:", self.name)
            self._flipped = False
            self._run_stacking(force_stacker=force_stacker, force_pairer=force_pairer)
        else:
            print("Stacking already done for variant:", self.name)

        # Check for expected output files and run flipped
        if (
            force_stacker
            or not self.file_stack_fluxes_flipped.exists()
            or not self.file_stack_mags_flipped.exists()
            or not self.file_stack_fcolors_flipped.exists()
            or not self.file_stack_mcolors_flipped.exists()
            # or not self.file_stack_shear_flipped.exists()
        ):
            print("Running FLIPPED stacking for variant:", self.name)
            self._run_stacking(force_stacker=force_stacker, flipped=True)
        else:
            print("FLIPPED stacking already done for variant:", self.name)

        # Save figs
        if (
            force_plotter
            or any(
                (
                    self._new_flux_stack,
                    self._new_mags_stack,
                    self._new_fcolors_stack,
                    self._new_mcolors_stack,
                )
            )
            or not self.file_stack_fluxes_fig.exists()
            or not self.file_stack_mags_fig.exists()
            or not self.file_stack_fcolors_fig.exists()
            or not self.file_stack_mcolors_fig.exists()
            # or not self.file_stack_shear_fig.exists()
        ):
            print("Creating figures for variant:", self.name)
            self._create_figure("fluxes", force=force_plotter)
            self._create_figure("mags", force=force_plotter)
            self._create_figure("fcolors", force=force_plotter)
            self._create_figure("mcolors", force=force_plotter)
            # self._create_figure("shear", force=force_plotter)
        else:
            print("Figures already exist for variant:", self.name)

        print()


# Alias default variant
Default = Stacker


# Other variants
@dataclass
class UnCleaned(Default):
    name: str = "uncleaned"
    clean_background: bool = False


@dataclass
class SNRMax20(Default):
    name: str = "snr_max_20"
    snr_max: float = 20.0


@dataclass
class SNRMax500(Default):
    name: str = "snr_max_500"
    snr_max: float = 500.0


@dataclass
class SNRMaxInf(Default):
    name: str = "snr_max_inf"
    snr_max: float = np.inf


@dataclass
class RandPositions(Default):
    name: str = "rand_positions_42"
    randomize_positions: bool = True
