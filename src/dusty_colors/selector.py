"""Class to perform selection."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord

from .utils import select_ecdfs


@dataclass
class Selector:
    """Class to perform selection on the processed catalog."""

    name: str = "default"

    # Choose the catalog
    catalog: str = "dp1"

    # Only ECDFS? (which has deepest data and best photo-zs)
    only_ecdfs: bool = False

    # Cuts on the raw catalog
    min_snr_u: float = 5
    min_snr_g: float = 5
    min_snr_r: float = 5
    min_snr_i: float = 10
    min_snr_z: float = 5
    min_snr_y: float = 5
    bright_cut: float = 18.0  # Don't use objects brighter than this in i (mag)
    bright_radius: float = 0  # Radius around bright objects to mask (arcsec)
    blendedness_cut: float = 0.42  # Maximum blendedness

    # Photo-z quality cuts
    pz_max_sig: float = 0.25  # Max photo-z uncertainty (1-sigma)
    pz_max_diff: float = 0.2  # Max difference between FZB and LePhare

    # Foreground/background selection
    fg_zmin: float = 0.2
    fg_zmax: float = 0.5
    bg_zmin: float = 0.6
    bg_zmax: float = 1.5

    # Red sequence selection
    bg_red_seq: bool = False  # If true, select bg galaxies on the red sequence
    red_seq_cut: str = (  # Color-magnitude cut for red sequence
        "((r_gaap1p0Mag - i_gaap1p0Mag) > -0.1 * z_cModelMag + 3.1) and "
        "((r_gaap1p0Mag - i_gaap1p0Mag) < -0.1 * z_cModelMag + 3.5)"
    )

    def __post_init__(self) -> None:
        """Post-init processing."""
        self.in_dir = Path("data")
        self.out_dir = Path(f"results/catalogs/{self.name}")
        self.file_galaxies = self.out_dir / "galaxy_catalog.parquet"
        self.file_foreground = self.out_dir / "galaxy_catalog_foreground.parquet"
        self.file_background = self.out_dir / "galaxy_catalog_background.parquet"
        self.file_stars = self.out_dir / "star_catalog.parquet"

    def cut_photoz(self, cat: pd.DataFrame) -> pd.DataFrame:
        """Define the photo-z quality cut."""
        # Save photo-z estimate that is weighted mean of FlexZBoost and LePhare
        fzb = cat["fzboost_z_mode"]
        fzb_sig = (cat["fzboost_z_err68_high"] - cat["fzboost_z_err68_low"]) / 2
        lph = cat["lephare_z_mode"]
        lph_sig = (cat["lephare_z_err68_high"] - cat["lephare_z_err68_low"]) / 2
        cat = cat.assign(
            z_phot=(fzb / fzb_sig**2 + lph / lph_sig**2)
            / (1 / fzb_sig**2 + 1 / lph_sig**2)
        )

        # Use galaxies with high-quality FlexZBoost and LePhare photo-zs
        cut = (
            cat["fzboost_z_err68_high"] - cat["fzboost_z_err68_low"]
        ) / 2 < self.pz_max_sig
        cut &= (
            cat["lephare_z_err68_high"] - cat["lephare_z_err68_low"]
        ) / 2 < self.pz_max_sig

        # Require agreement between the two photo-z estimates
        max_dz = self.pz_max_diff * (1 + cat["z_phot"])
        cut &= np.abs(cat["fzboost_z_mode"] - cat["lephare_z_mode"]) < max_dz

        return cat[cut]

    def run(self, force: bool = False) -> None:
        """Run selection."""
        # Set directories
        in_dir = Path("data")
        out_dir = Path(f"results/catalogs/{self.name}")

        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check for expected output files and run
        if (
            force
            or not self.file_galaxies.exists()
            or not self.file_foreground.exists()
            or not self.file_background.exists()
            or not self.file_stars.exists()
        ):
            print(f"Running selection for variant: {self.name}")

            # Save the config
            with open(out_dir / "config_selector.yaml", "w") as file:
                yaml.dump(asdict(self), file, sort_keys=False)

            # Load the processed catalog
            cat = pd.read_parquet(in_dir / "dp1_catalog_processed.parquet")
            print("   starting length:", len(cat))

            # Only ECDFS?
            if self.only_ecdfs:
                cat = select_ecdfs(cat)
                print("   after ECDFS cut:", len(cat))

            # Cut on SNR
            for band in ["u", "g", "r", "i", "z", "y"]:
                cat = cat.query(f"{band}_snr > @self.min_snr_{band}")
            print("   after SNR cuts:", len(cat))

            # Bright mask
            if self.bright_radius > 0:
                # Create SkyCoord objects for bright sources and all sources
                bright = cat.query("i_cModelMag < @self.bright_cut")
                bright_coords = SkyCoord(
                    ra=bright["coord_ra"].values,
                    dec=bright["coord_dec"].values,
                    unit="deg",
                )
                all_coords = SkyCoord(
                    ra=cat["coord_ra"].values,
                    dec=cat["coord_dec"].values,
                    unit="deg",
                )

                # Find all sources within bright_radius of any bright source
                idx_within_radius = set()
                for bc in bright_coords:
                    sep = all_coords.separation(bc)
                    idx_within_radius.update(
                        np.where(sep.arcsec < self.bright_radius)[0]
                    )

                # Mask out sources within bright_radius of any bright source
                mask = np.ones(len(cat), dtype=bool)
                mask[list(idx_within_radius)] = False
                cat = cat[mask]
                print("   after bright mask:", len(cat))
            else:
                cat = cat.query("i_cModelMag > @self.bright_cut")
                print("   after brightness cut:", len(cat))

            # Cut on blendedness
            cat = cat.query("i_blendedness < @self.blendedness_cut")
            print("   after blendedness cut:", len(cat))

            # Separate stars and galaxies
            stars = cat.query("refExtendedness < 0.5 and i_cModelMag < 22")
            galaxies = cat.query("refExtendedness > 0.5")
            print(f"   stars/galaxies: {len(stars)}, {len(galaxies)}")

            # Photo-z quality cut
            galaxies = self.cut_photoz(galaxies)
            print("   after photo-z quality cut:", len(galaxies))

            # Foreground selection
            fg_cat = galaxies.query("z_phot > @self.fg_zmin and z_phot < @self.fg_zmax")
            print("   foreground galaxies:", len(fg_cat))

            # Background selection
            bg_cat = galaxies.query("z_phot > @self.bg_zmin and z_phot < @self.bg_zmax")
            print("   background galaxies:", len(bg_cat))

            # Red sequence selection
            if self.bg_red_seq:
                # Fit the red sequence
                bg_cat = bg_cat.query(self.red_seq_cut)
                print("   background galaxies after red sequence:", len(bg_cat))

            # Save catalogs
            galaxies.to_parquet(self.file_galaxies)
            fg_cat.to_parquet(self.file_foreground)
            bg_cat.to_parquet(self.file_background)
            stars.to_parquet(self.file_stars)

            print("   selection complete")

        else:
            print(f"Selection already done for variant: {self.name}")


# Alias default variant
Default = Selector


# Other variants
@dataclass
class BrightCut20(Default):
    name: str = "bright_cut_20"
    bright_cut: float = 20.0


@dataclass
class BrightMaskR20(Default):
    name: str = "bright_mask_r20"
    bright_radius: float = 20.0


@dataclass
class BrightMaskR50(Default):
    name: str = "bright_mask_r50"
    bright_radius: float = 50.0


@dataclass
class BrightMaskR100(Default):
    name: str = "bright_mask_r100"
    bright_radius: float = 100.0


@dataclass
class BlendCut100(Default):
    name: str = "blend_cut_100"
    blendedness_cut: float = 1.00


@dataclass
class BlendCut20(Default):
    name: str = "blend_cut_20"
    blendedness_cut: float = 0.20


@dataclass
class BlendCut10(Default):
    name: str = "blend_cut_10"
    blendedness_cut: float = 0.10


@dataclass
class SNRi20(Default):
    name: str = "snr_i_20"
    min_snr_i: float = 20.0


@dataclass
class SNRConservative(Default):
    name: str = "snr_conservative"
    min_snr_u: float = 5.0
    min_snr_g: float = 10.0
    min_snr_r: float = 10.0
    min_snr_i: float = 20.0
    min_snr_z: float = 10.0
    min_snr_y: float = 5.0


@dataclass
class RedSeq(Default):
    name: str = "red_seq"
    bg_red_seq: bool = True


@dataclass
class EcdfsOnly(Default):
    name: str = "ecdfs_only"
    only_ecdfs: bool = True


@dataclass
class FgLowZ(Default):
    name: str = "fg_low_z"
    fg_zmin: float = 0.1
    fg_zmax: float = 0.4
    bg_zmin: float = 0.5


@dataclass
class BgLowZ(Default):
    name: str = "bg_low_z"
    fg_zmin: float = 0.1
    fg_zmax: float = 0.4
    bg_zmin: float = 0.5
    bg_zmax: float = 1.0


@dataclass
class DzGap0p2(Default):
    name: str = "dz_gap_0p2"
    fg_zmin: float = 0.2
    fg_zmax: float = 0.5
    bg_zmin: float = 0.7
    bg_zmax: float = 1.5


@dataclass
class DzGap0p3(Default):
    name: str = "dz_gap_0p3"
    fg_zmin: float = 0.2
    fg_zmax: float = 0.5
    bg_zmin: float = 0.8
    bg_zmax: float = 1.5


@dataclass
class PZStrict(Default):
    name: str = "pz_strict"
    pz_max_sig: float = 0.1
    pz_max_diff: float = 0.1
