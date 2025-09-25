"""Class to perform selection."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class Selector:
    """Class to perform selection on the processed catalog."""

    name: str = "default"

    # Choose the catalog
    catalog: str = "dp1"

    # Cuts on the raw catalog
    min_snr_u: float = 0
    min_snr_g: float = 5
    min_snr_r: float = 5
    min_snr_i: float = 10
    min_snr_z: float = 5
    min_snr_y: float = 0
    bright_cut: float = 18.0  # Don't use objects brighter than this (mag)
    bright_radius: float = 0  # Radius around bright objects to mask (arcsec)
    blendedness_cut: float = 1.0  # Maximum blendedness

    # Foreground/background selection
    fg_zmin: float = 0.2
    fg_zmax: float = 0.5
    bg_zmin: float = 0.6
    bg_zmax: float = 1.5
    tomographic: bool = True  # If false, use all pairs with z_bg > z_fg + dz_min
    dz_min: float = 0.2

    # Red sequence selection
    bg_red_seq: bool = False  # If true, select bg galaxies on the red sequence
    red_seq_cut: str = "(r_gaap1p0Mag - i_gaap1p0Mag) > -0.1 * z_cModelMag + 3.1"
    red_seq_sigma: float = 0.4  # Width of red sequence in mags

    # Only ECDFS? (which has deepest data and best photo-zs)
    only_ecdfs: bool = False

    @staticmethod
    def cut_photoz(cat: pd.DataFrame) -> pd.DataFrame:
        """Define the photo-z quality cut."""
        # Use galaxies with high-quality FlexZBoost and LePhare photo-zs
        max_sig = 0.25
        cut = (cat["fzboost_z_err68_high"] - cat["fzboost_z_err68_low"]) / 2 < max_sig
        cut &= (cat["lephare_z_err68_high"] - cat["lephare_z_err68_low"]) / 2 < max_sig

        # Require agreement between the two photo-z estimates
        max_dz = 0.2 * (1 + cat["z_phot"])
        cut &= np.abs(cat["fzboost_z_mode"] - cat["lephare_z_mode"]) < max_dz

        return cat[cut]

    def run(self, force: bool = False) -> None:
        """Run selection."""
        # Set directories
        in_dir = Path("data")
        out_dir = Path(f"results/{self.name}")

        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check for expected output files and run
        full = out_dir / "selected_catalog.fits"
        fg = out_dir / "foreground_catalog.fits"
        bg = out_dir / "background_catalog.fits"
        if force or not full.exists() or not fg.exists() or not bg.exists():
            print(f"Running selection for variant: {self.name}")

            # Save the config
            with open(out_dir / "config_selector.yaml", "w") as file:
                yaml.dump(asdict(self), file, sort_keys=False)

            # Run selection
            ...

            # Create plots to visualize selection
            ...

        else:
            print(f"Selection already done for variant: {self.name}")
