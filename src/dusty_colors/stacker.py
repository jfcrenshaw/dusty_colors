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


@dataclass
class Stacker:
    """Class to define variants for stacking."""

    name: str = "default"

    selector: Selector = field(default_factory=DefaultSelector)

    # Stacking options
    stack_mags: bool = False  # Whether to stack in mags instead of fluxes
    free_fluxes: bool = False  # Whether to use "free" flux variants
    snr_max: float = 100  # Maximum SNR (sets error floor)

    # Defining bins
    bin_by_angle: bool = False  # If false, bin by physical impact parameter
    r_min: float = 0.0  # Minimum radius (Mpc or arcmin)
    r_max: float = 4.0  # Maximum radius (Mpc or arcmin)
    n_bins: int = 10  # Number of bins

    # Toggle tomographic selection
    tomographic: bool = True  # If false, use all pairs with z_bg > z_fg + dz_min
    dz_min: float = 0.2

    # For null tests
    fg_stars: bool = False  # Whether to use stars as foreground objects
    flip: bool = False  # Whether to flip foreground/background samples

    def __post_init__(self) -> None:
        """Post-init processing."""
        # Directories and files
        self.in_dir = Path(f"results/catalogs/{self.selector.name}")
        self.out_dir = Path(f"results/stacks/{self.name}")
        self.file_pairs = self.out_dir / "pairs.npz"
        self.file_stack_fluxes = self.out_dir / "stack_fluxes.npz"
        self.file_stack_colors = self.out_dir / "stack_colors.npz"
        self.file_stack_shear = self.out_dir / "stack_shear.npz"

    def find_pairs(self, force: bool = False) -> None:
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

        # Flat-sky approximation
        ra = self._foreground["coord_ra"]
        dec = self._foreground["coord_dec"]
        fg_xy = np.column_stack((ra * np.cos(np.deg2rad(dec)), dec))

        ra = self._background["coord_ra"]
        dec = self._background["coord_dec"]
        bg_xy = np.column_stack((ra * np.cos(np.deg2rad(dec)), dec))

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
        np.savez_compressed(
            self.file_pairs, pairs=self._pairs, separation=self._separation
        )
        print(f"{len(self._pairs)} found")

    def _calc_binned_stats(self, sep, x, weights, bin_edges):
        """Calculate binned statistics."""
        # Assign bins
        bin_indices = np.digitize(sep, bin_edges) - 1
        valid = (bin_indices >= 0) & (bin_indices < self.n_bins)
        bin_indices = bin_indices[valid]
        x = x[valid]
        weights = weights[valid]

        # Calculate weighted average in each bin
        avgs = []
        errs = []
        for i in range(self.n_bins):
            # Find values in bin
            in_bin = bin_indices == i

            # If bin is empty, append NaN
            if in_bin.sum() == 0:
                avgs.append(np.nan)
                errs.append(np.nan)
                continue

            # Weighted average and error
            avgs.append(np.average(x[in_bin], weights=weights[in_bin]))
            var = (
                x[in_bin].var()
                * np.sum(weights[in_bin] ** 2)
                / weights[in_bin].sum() ** 2
            )
            var += 1 / weights[in_bin].sum()
            errs.append(np.sqrt(var))

        return np.array(avgs), np.array(errs)

    def stack_fluxes(self, force: bool = False) -> None:
        """Stack fluxes."""
        # Skip time-intensive stacking if already done
        if self.file_stack_fluxes.exists() and not force:
            print("   fluxes already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking fluxes...", end="", flush=True)

        # Set bins
        bin_edges = np.linspace(self.r_min, self.r_max, self.n_bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Container for results
        results = {"bin_centers": bin_centers}

        # Loop over bands
        for band in "ugrizy":
            print(f" {band}", end="", flush=True)

            # Get columns
            col = f"{band}_{'free_' if self.free_fluxes else ''}cModel"
            if self.stack_mags:
                col += "Mag"
            else:
                col += "Flux"
            x = self._background[col].iloc[self._pairs[:, 1]]
            err = self._background[col + "Err"].iloc[self._pairs[:, 1]]

            # Filtering
            mask = np.isfinite(x) & np.isfinite(err) & (err > 0)
            x = x[mask]
            err = err[mask]
            sep = self._separation[mask]

            # Apply SNR maximum
            if self.stack_mags:
                err = np.clip(err, 2.5 / np.log(10) * 1 / self.snr_max, None)
            else:
                err = x * np.clip(err / x, 1 / self.snr_max, None)

            # Calculate weights
            weights = 1 / err**2

            # Calculate binned stats
            avgs, errs = self._calc_binned_stats(sep, x, weights, bin_edges)
            results[f"{band}_avg"] = np.array(avgs)
            results[f"{band}_err"] = np.array(errs)

        # Save all results
        np.savez_compressed(self.file_stack_fluxes, **results)  # type: ignore
        print(".")

    def stack_colors(self, force: bool = False) -> None:
        """Stack colors."""
        # Skip time-intensive stacking if already done
        if self.file_stack_colors.exists() and not force:
            print("   colors already stacked")
            return

        # Otherwise, move forward with finding new pairs
        print("   stacking colors...", end="", flush=True)

        # Set bins
        bin_edges = np.linspace(self.r_min, self.r_max, self.n_bins + 1)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

        # Container for results
        results = {"bin_centers": bin_centers}

        # Loop over bands
        bands = list("ugrizy")
        for i in range(len(bands) - 1):
            band1 = bands[i]
            band2 = bands[i + 1]
            print(f" {band1}-{band2}", end="", flush=True)

            # Get columns
            col1 = f"{band1}_cModel"
            col2 = f"{band2}_cModel"
            if self.stack_mags:
                col1 += "Mag"
                col2 += "Mag"
            else:
                col1 += "Flux"
                col2 += "Flux"
            x1 = self._background[col1].iloc[self._pairs[:, 1]]
            err1 = self._background[col1 + "Err"].iloc[self._pairs[:, 1]]
            x2 = self._background[col2].iloc[self._pairs[:, 1]]
            err2 = self._background[col2 + "Err"].iloc[self._pairs[:, 1]]

            # Filtering
            mask = (
                np.isfinite(x1)
                & np.isfinite(err1)
                & (err1 > 0)
                & np.isfinite(x2)
                & np.isfinite(err2)
                & (err2 > 0)
            )
            x1 = x1[mask]
            err1 = err1[mask]
            x2 = x2[mask]
            err2 = err2[mask]
            sep = self._separation[mask]

            # Apply SNR maximum
            if self.stack_mags:
                err1 = np.clip(err1, 2.5 / np.log(10) * 1 / self.snr_max, None)
                err2 = np.clip(err2, 2.5 / np.log(10) * 1 / self.snr_max, None)
            else:
                err1 = x1 * np.clip(err1 / x1, 1 / self.snr_max, None)
                err2 = x2 * np.clip(err2 / x2, 1 / self.snr_max, None)

            # Calculate color and error
            if self.stack_mags:
                x = x1 - x2
                err = np.sqrt(err1**2 + err2**2)
            else:
                x = x1 / x2
                err = x * np.sqrt((err1 / x1) ** 2 + (err2 / x2) ** 2)

            # Calculate weights
            weights = 1 / err**2

            # Calculate binned stats
            avgs, errs = self._calc_binned_stats(sep, x, weights, bin_edges)
            results[f"{band1}-{band2}_avg"] = np.array(avgs)
            results[f"{band1}-{band2}_err"] = np.array(errs)

        # Save all results
        np.savez_compressed(self.file_stack_colors, **results)  # type: ignore
        print(".")

    def stack_shear(self, force: bool = False) -> None:
        """Stack shear."""
        raise NotImplementedError("Shear stacking not yet implemented.")

    def run(self, force_selector: bool = False, force_stacker: bool = False) -> None:
        """Run stacking."""
        # Run selector
        self.selector.run(force=force_selector)

        # Set foreground and background samples
        if self.fg_stars:
            self._foreground = pd.read_parquet(self.in_dir / "star_catalog.parquet")
        else:
            self._foreground = pd.read_parquet(
                self.in_dir / "galaxy_catalog_foreground.parquet"
            )
        self._background = pd.read_parquet(
            self.in_dir / "galaxy_catalog_background.parquet"
        )
        if self.flip:
            self._foreground, self._background = self._background, self._foreground

        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Check for expected output files and run
        if (
            force_stacker
            or not self.file_stack_fluxes.exists()
            # or not self.file_stack_colors.exists()
            # or not self.file_stack_shear.exists()
        ):
            print(f"Running stacking for variant: {self.name}")

            # Save the config
            with open(self.out_dir / "config_stacker.yaml", "w") as file:
                config = asdict(self)
                config["selector"] = asdict(self.selector)
                yaml.dump(config, file, sort_keys=False)

            # Run stacking
            self.find_pairs(force=force_stacker)
            self.stack_fluxes(force=force_stacker)
            # self.stack_colors(force=force_stacker)
            # self.stack_shear()
            ...

            # Create plots to visualize stacking
            ...

            print("   stacking complete")
        else:
            print(f"Stacking already done for variant: {self.name}")

        print()


# Alias default variant
Default = Stacker


# Other variants
@dataclass
class SNRMax20(Default):
    name: str = "snr_max_20"
    snr_max: float = 20.0


@dataclass
class SNRMax500(Default):
    name: str = "snr_max_500"
    snr_max: float = 500.0


@dataclass
class BinByAngle(Default):
    name: str = "bin_by_angle"
    bin_by_angle: bool = True


@dataclass
class StackMags(Default):
    name: str = "stack_mags"
    stack_mags: bool = True


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
