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


@dataclass
class Stacker:
    """Class to define variants for stacking."""

    name: str = "default"

    selector: Selector = field(default_factory=DefaultSelector)

    # Clean background?
    clean_background: bool = True

    # Stacking options
    free_fluxes: bool = False  # Whether to use "free" flux variants
    snr_max: float = 100  # Maximum SNR (sets error floor)

    # Defining bins
    bin_by_angle: bool = False  # If false, bin by physical impact parameter
    r_min: float = 0.0  # Minimum radius (Mpc or arcmin)
    r_max: float = 4  # Maximum radius (Mpc or arcmin)
    n_bins: int | dict = 10
    # field(
    #    default_factory=lambda: dict(u=40, g=40, r=20, i=10, z=10, y=10)
    # )

    # Toggle tomographic selection
    tomographic: bool = True  # If false, use all pairs with z_bg > z_fg + dz_min
    dz_min: float = 0.2

    # For null tests
    fg_stars: bool = False  # Whether to use stars as foreground objects
    randomize_positions: bool = False  # Whether to randomize positions

    # How to calculate errors
    bootstrap: bool = True  # If false, use analytic errors

    def __post_init__(self) -> None:
        """Post-init processing."""
        # Directories and files
        self.in_dir = Path(f"results/catalogs/{self.selector.name}")
        self.out_dir = Path(f"results/stacks/{self.name}")

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

        # Check for not-yet-implemented options
        if not self.tomographic:
            raise NotImplementedError("Non-tomographic selection not yet implemented.")
        if self.fg_stars:
            raise NotImplementedError("Using stars as foreground not yet implemented.")

        # Set foreground and background samples
        if self.fg_stars:
            self._foreground: pd.DataFrame = pd.read_parquet(
                self.in_dir / "star_catalog.parquet"
            )
        else:
            self._foreground: pd.DataFrame = pd.read_parquet(
                self.in_dir / "galaxy_catalog_foreground.parquet"
            )
        if self.clean_background:
            self._background: pd.DataFrame = pd.read_parquet(
                self.in_dir / "galaxy_catalog_background_cleaned.parquet"
            )
            self._background_cleaned = True
        else:
            self._background: pd.DataFrame = pd.read_parquet(
                self.in_dir / "galaxy_catalog_background.parquet"
            )
            self._background_cleaned = False

        self._flipped = False

    def save_config(self) -> None:
        with open(self.file_config, "w") as file:
            config = asdict(self)
            config["selector"] = asdict(self.selector)
            yaml.dump(config, file, sort_keys=False)

    def _get_bins(self, band: str) -> np.ndarray:
        """Get bin edges."""
        if isinstance(self.n_bins, dict):
            n_bins = self.n_bins[band]
        else:
            n_bins = self.n_bins
        if self.bin_by_angle:
            # Bins in arcmin
            return np.linspace(self.r_min, self.r_max, n_bins + 1)
        else:
            # Bins in Mpc
            r = np.sqrt(np.linspace(self.r_min**2, self.r_max**2, n_bins))
            r = np.insert(r, 1, r[1] / 2)  # Add extra bin at small scales
            return r

    def find_pairs(self, force: bool = False) -> None:
        """Find foreground-background pairs."""
        # Skip time-intensive pair finding if already done
        file = self.file_pairs_flipped if self._flipped else self.file_pairs
        if file.exists() and not force:
            print(f"   pairs already found, loading from {self.file_pairs}")
            data = np.load(self.file_pairs)
            self._pairs = data["pairs"]
            self._separation = data["separation"]
            print(f"   {len(self._pairs)} pairs loaded")
            return

        # Otherwise, move forward with finding new pairs
        print("   finding pairs...", end=" ", flush=True)

        if self.randomize_positions:
            rng = np.random.default_rng(42)
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

        # Build spatial KDTree on background
        bg_tree = cKDTree(bg_xy)

        # Determine max angular separation for each foreground galaxy
        if not self.bin_by_angle:
            # Calculate angular diameter distances for all foreground galaxies
            dA = cosmo.angular_diameter_distance(self._foreground["z_phot"]).value

            # Compute max separation in degrees for each foreground galaxy
            max_sep_deg = np.rad2deg(self.r_max / dA)
        else:
            # If binning by angle, max sep. same for all foreground galaxies
            max_sep_deg = np.full(len(fg_xy), self.r_max / 60)  # arcmin to deg

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
        np.savez_compressed(file, pairs=self._pairs, separation=self._separation)
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
            wi = 1 / (ei**2 + xi.var())

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

    def _stack_flux_or_mag(self, flux: bool) -> dict:
        """Perform stacking of either fluxes or magnitudes."""
        # Container for results
        results = {}

        # Loop over bands
        for band in "ugrizy":
            print(f" {band}", end="", flush=True)

            # Set bins
            bin_edges = self._get_bins(band)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            results[f"{band}_bin_centers"] = bin_centers

            # Get columns
            col = f"{band}_{'free_' if self.free_fluxes else ''}cModel"
            if flux:
                col += "Flux"
            else:
                col += "Mag"
            x = self._background[col].iloc[self._pairs[:, 1]].values
            err = self._background[col + "Err"].iloc[self._pairs[:, 1]].values

            # Filtering
            mask = np.isfinite(x) & np.isfinite(err) & (err > 0)
            x = x[mask]
            err = err[mask]
            sep = self._separation[mask]

            # Apply SNR maximum
            if flux:
                err = x * np.clip(err / x, 1 / self.snr_max, None)
            else:
                err = np.clip(err, 2.5 / np.log(10) * 1 / self.snr_max, None)

            # Calculate binned stats
            avgs, errs = self._calc_binned_stats(sep, x, err, bin_edges)
            results[f"{band}_avg"] = avgs
            results[f"{band}_err"] = errs

        return results

    def stack_fluxes(self, force: bool = False) -> None:
        """Stack fluxes."""
        # Skip time-intensive stacking if already done
        file = (
            self.file_stack_fluxes_flipped if self._flipped else self.file_stack_fluxes
        )
        if file.exists() and not force:
            print("   fluxes already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking fluxes...", end="", flush=True)
        results = self._stack_flux_or_mag(flux=True)

        # Save all results
        np.savez_compressed(file, **results)  # type: ignore
        print(".")

        self._new_flux_stack = True

    def stack_mags(self, force: bool = False) -> None:
        """Stack magnitudes."""
        # Skip time-intensive stacking if already done
        file = self.file_stack_mags_flipped if self._flipped else self.file_stack_mags
        if file.exists() and not force:
            print("   mags already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking mags...", end="", flush=True)
        results = self._stack_flux_or_mag(flux=False)

        # Save all results
        np.savez_compressed(file, **results)  # type: ignore
        print(".")

        self._new_mags_stack = True

    def _stack_colors(self, flux: bool) -> dict:
        """Perform stacking of either flux ratios or color differences."""
        # Container for results
        results = {}

        # Loop over bands
        bands = list("ugrizy")
        for i in range(len(bands) - 1):
            band1 = bands[i]
            band2 = bands[i + 1]
            print(f" {band1}-{band2}", end="", flush=True)

            # Set bins
            bin_edges = self._get_bins(band1)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            results[f"{band1}-{band2}_bin_centers"] = bin_centers

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

                x1 = self._background[col1].iloc[self._pairs[:, 1]].values
                err1 = self._background[col1 + "Err"].iloc[self._pairs[:, 1]].values
                x2 = self._background[col2].iloc[self._pairs[:, 1]].values
                err2 = self._background[col2 + "Err"].iloc[self._pairs[:, 1]].values

                # Apply SNR maximum
                err1 = x1 * np.clip(err1 / x1, 1 / self.snr_max, None)
                err2 = x2 * np.clip(err2 / x2, 1 / self.snr_max, None)

                # Calculate color and error
                x = x1 / x2
                err = x * np.sqrt((err1 / x1) ** 2 + (err2 / x2) ** 2)

            else:
                # Retrieve columns
                x = self._background[f"{band1}-{band2}"].iloc[self._pairs[:, 1]].values
                err = (
                    self._background[f"{band1}-{band2}_Err"]
                    .iloc[self._pairs[:, 1]]
                    .values
                )

                # Apply SNR maximum
                err = np.clip(
                    err, np.sqrt(2) * 2.5 / np.log(10) * 1 / self.snr_max, None
                )

            # Filtering
            mask = np.isfinite(x) & np.isfinite(err) & (err > 0)
            x = x[mask]
            err = err[mask]
            sep = self._separation[mask]

            # Calculate binned stats
            avgs, errs = self._calc_binned_stats(sep, x, err, bin_edges)
            results[f"{band1}-{band2}_avg"] = avgs
            results[f"{band1}-{band2}_err"] = errs

        return results

    def stack_fcolors(self, force: bool = False) -> None:
        """Stack colors using flux ratios."""
        # Don't stack flux colors if background is cleaned
        if self._background_cleaned:
            print("   skipping flux-color stacking (background data is cleaned)")
            return

        # Skip time-intensive stacking if already done
        file = (
            self.file_stack_fcolors_flipped
            if self._flipped
            else self.file_stack_fcolors
        )
        if file.exists() and not force:
            print("   flux-colors already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking flux-colors...", end="", flush=True)
        results = self._stack_colors(flux=True)

        # Save all results
        np.savez_compressed(file, **results)  # type: ignore
        print(".")

        self._new_fcolors_stack = True

    def stack_mcolors(self, force: bool = False) -> None:
        """Stack colors using magnitude differences."""
        # Skip time-intensive stacking if already done
        file = (
            self.file_stack_mcolors_flipped
            if self._flipped
            else self.file_stack_mcolors
        )
        if file.exists() and not force:
            print("   magnitude-colors already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking magnitude-colors...", end="", flush=True)
        results = self._stack_colors(flux=False)

        # Save all results
        np.savez_compressed(file, **results)  # type: ignore
        print(".")

        self._new_mcolors_stack = True

    def stack_shear(self, force: bool = False) -> None:
        """Stack shear."""
        raise NotImplementedError("Shear stacking not yet implemented.")

    def create_figure(self, stack_type: str, force: bool = False) -> None:
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
            ) or self._background_cleaned:
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

        fig = plot_stack(self.out_dir, stack_type=stack_type)
        fig.savefig(fig_file, bbox_inches="tight")
        print(f"   saved figure to {fig_file}")

    def _run_stacking(self, force_stacker: bool = False) -> None:
        self.save_config()
        self.find_pairs(force=force_stacker)
        if len(self._pairs) == 0:
            print("   no pairs found, skipping stacking")
            return
        self.stack_fluxes(force=force_stacker)
        self.stack_mags(force=force_stacker)
        self.stack_fcolors(force=force_stacker)
        self.stack_mcolors(force=force_stacker)
        # self.stack_shear()

        print("   stacking complete")

    def run(
        self,
        force_selector: bool = False,
        force_stacker: bool = False,
        force_plotter: bool = False,
    ) -> None:
        # Run selector
        self.selector.run(force=force_selector)

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
            or (not self.file_stack_fcolors.exists() and not self._background_cleaned)
            or not self.file_stack_mcolors.exists()
            # or not self.file_stack_shear.exists()
        ):
            print("Running stacking for variant:", self.name)
            self._flipped = False
            self._run_stacking(force_stacker=force_stacker)
        else:
            print("Stacking already done for variant:", self.name)

        # Check for expected output files and run flipped
        if (
            force_stacker
            or not self.file_stack_fluxes_flipped.exists()
            or not self.file_stack_mags_flipped.exists()
            or (
                not self.file_stack_fcolors_flipped.exists()
                and not self._background_cleaned
            )
            or not self.file_stack_mcolors_flipped.exists()
            # or not self.file_stack_shear_flipped.exists()
        ):
            print("Running FLIPPED stacking for variant:", self.name)
            self._flipped = True
            self._foreground, self._background = self._background, self._foreground
            self._run_stacking(force_stacker=force_stacker)
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
            or (
                not self.file_stack_fcolors_fig.exists()
                and not self._background_cleaned
            )
            or not self.file_stack_mcolors_fig.exists()
            # or not self.file_stack_shear_fig.exists()
        ):
            print("Creating figures for variant:", self.name)
            self.create_figure("fluxes", force=force_plotter)
            self.create_figure("mags", force=force_plotter)
            self.create_figure("fcolors", force=force_plotter)
            self.create_figure("mcolors", force=force_plotter)
            # self.create_figure("shear", force=force_plotter)
        else:
            print("Figures already exist for variant:", self.name)

        print()


# Alias default variant
Default = Stacker


# Other variants
@dataclass
class Nbins5(Default):
    name: str = "nbins_5"
    n_bins: int | dict = 5


@dataclass
class Nbins20(Default):
    name: str = "nbins_20"
    n_bins: int | dict = 20


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
class BinByImpact(Default):
    name: str = "bin_by_impact"
    bin_by_angle: bool = True
    r_max: float = 60.0  # arcmin


@dataclass
class FreeFluxes(Default):
    name: str = "free_fluxes"
    free_fluxes: bool = True


@dataclass
class NullGalaxiesFlip(Default):
    name: str = "null_galaxies_flip"
    fg_stars: bool = False
    flip: bool = True


@dataclass
class NullStars(Default):
    name: str = "null_stars"
    fg_stars: bool = True
    flip: bool = False


@dataclass
class NullStarsFlip(Default):
    name: str = "null_stars_flip"
    fg_stars: bool = True
    flip: bool = True


@dataclass
class RandPositions(Default):
    name: str = "rand_positions"
    randomize_positions: bool = True
