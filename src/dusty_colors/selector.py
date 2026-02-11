"""Class to perform selection."""

from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from astropy.coordinates import SkyCoord

from .utils import clean_data, select_ecdfs


@dataclass
class Selector:
    """Class to perform selection on the processed catalog."""

    name: str = "default"

    # Choose the catalog
    catalog: str = "dp1"

    # Only ECDFS? (which has deepest data and best photo-zs)
    only_ecdfs: bool = False

    # Cuts on faint end
    i_cut: float = 24.0  # Maximum i-band magnitude
    min_snr_u: float = 1
    min_snr_g: float = 5
    min_snr_r: float = 5
    min_snr_i: float = 10
    min_snr_z: float = 5
    min_snr_y: float = 1
    min_pixel_quantile: float = 0.05  # Minimum pixel depth quantile to accept

    # Cuts on bright end
    bright_cut: float = 18.0  # Don't use objects brighter than this in i (mag)
    bright_radius: float = 0  # Radius around bright objects to mask (arcsec)

    # Cuts on blendedness
    blendedness_cut: float = 0.42  # Maximum blendedness

    # Photo-z quality cuts
    # Note if you increase either beyond 0.2, process_dp1.py needs to be
    # re-run with appropriately larger cuts on z_phot_err and z_phot_diff
    pz_max_sig: float = 0.1  # Max photo-z uncertainty (1-sigma)
    pz_max_diff: float = 0.1  # Max difference between FZB and LePhare

    # Foreground/background selection
    # Note if you increase fg_zmax to > 0.5, the kcorrections in process_dp1.py
    # need to be re-run with a larger zmax, including the "broad" template setup:
    # https://kcorrect.readthedocs.io/en/stable/templates.html
    fg_zmin: float = 0.2
    fg_zmax: float = 0.5
    bg_zmin: float = 0.7
    bg_zmax: float = 1.4

    # Parameters for background cleaning
    clean_nonuniformity: bool = True  # Clean non-uniformity in background?
    clean_ztrends: bool = True  # Clean redshift trends in background?
    clean_outliers: bool = True  # Remove outliers from background?

    # Red sequence selection
    bg_red_seq: bool = False  # If true, select bg galaxies on the red sequence
    red_seq_cut: str = (  # Color-magnitude cut for red sequence
        "((r_gaap1p0Mag - i_gaap1p0Mag) > -0.1 * z_cModelMag + 3.1) and "
        "((r_gaap1p0Mag - i_gaap1p0Mag) < -0.1 * z_cModelMag + 3.5)"
    )

    def __post_init__(self) -> None:
        """Post-init processing."""
        current_file = Path(__file__).resolve()
        root = current_file.parents[2]
        self.in_dir = root / "data"
        self.out_dir = root / f"results/catalogs/{self.name}"
        self.file_galaxies = self.out_dir / "galaxy_catalog.parquet"
        self.file_galaxies_pzcut = self.out_dir / "galaxy_catalog_pzcut.parquet"
        self.file_foreground = self.out_dir / "galaxy_catalog_foreground.parquet"
        self.file_background = self.out_dir / "galaxy_catalog_background.parquet"
        self.file_foreground_cleaned = (
            self.out_dir / "galaxy_catalog_foreground_cleaned.parquet"
        )
        self.file_background_cleaned = (
            self.out_dir / "galaxy_catalog_background_cleaned.parquet"
        )
        self.file_stars = self.out_dir / "star_catalog.parquet"
        self.file_figure_photoz = self.out_dir / "fig_photoz.png"

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

    def create_figure_photoz(self) -> None:
        """Create 3panel figure showing photo-z selection."""
        # Load photo-z catalogs
        fg_cat = pd.read_parquet(self.file_foreground)
        bg_cat = pd.read_parquet(self.file_background)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2), dpi=150)

        # Panel 1: Spec-z vs photo-z
        settings = dict(extent=(0, 2, 0, 2), gridsize=50, norm="log", edgecolors="none")
        ax1.hexbin(
            fg_cat["redshift"][~fg_cat["redshift.mask"]],
            fg_cat["z_phot"][~fg_cat["redshift.mask"]],
            cmap="Blues",
            **settings,
        )
        ax1.hexbin(
            bg_cat["redshift"][~bg_cat["redshift.mask"]],
            bg_cat["z_phot"][~bg_cat["redshift.mask"]],
            cmap="Reds",
            **settings,
        )
        ax1.set(
            xlim=(0, 1.5),
            ylim=(0, 1.5),
            xlabel="Spec-z",
            ylabel="Photo-z",
            aspect="equal",
        )
        ax1.plot([0, 2], [0, 2], c="k", lw=1, ls="--", zorder=0)

        # Panel 2: FlexZBoost vs LePhare photo-z
        ax2.hexbin(
            fg_cat["fzboost_z_mode"],
            fg_cat["lephare_z_mode"],
            cmap="Blues",
            **settings,
        )
        ax2.hexbin(
            bg_cat["fzboost_z_mode"],
            bg_cat["lephare_z_mode"],
            cmap="Reds",
            **settings,
        )
        ax2.set(
            xlim=(0, 1.5),
            ylim=(0, 1.5),
            xlabel="FlexZBoost photo-z",
            ylabel="LePhare photo-z",
            aspect="equal",
        )
        ax2.plot([0, 2], [0, 2], c="k", lw=1, ls="--", zorder=0)

        # Panel 3: Redshift histograms
        # Settings for all histograms
        settings = dict(range=(0, 1.5), bins=50, density=True, histtype="step")

        # Foreground sample
        ax3.hist(
            fg_cat["z_phot"],
            **settings,
            color="C0",
            ls="-",
        )
        ax3.hist(
            fg_cat["redshift"][~fg_cat["redshift.mask"]],
            **settings,
            color="C0",
            ls="--",
        )

        # Background sample
        ax3.hist(
            bg_cat["z_phot"],
            **settings,
            color="C3",
            ls="-",
        )
        ax3.hist(
            bg_cat["redshift"][~bg_cat["redshift.mask"]],
            **settings,
            color="C3",
            ls="--",
        )

        ax3.set(
            xlim=(0, 1.5),
            ylim=(0, 6.5),
            xlabel="Redshift",
            ylabel="Frequency",
            aspect=1.5 / 6.5,
        )

        ax3.hist([], histtype="step", color="k", ls="-", label="Photo-z")
        ax3.hist([], histtype="step", color="k", ls="--", label="Spec-z")
        ax3.legend(handlelength=1, frameon=False, fontsize=8)

        fig.subplots_adjust(wspace=0.4)

        fig.savefig(self.file_figure_photoz, dpi=500, bbox_inches="tight")
        print(f"   saved figure to {self.file_figure_photoz}")

    def run(self, force: bool = False) -> None:
        """Run selection."""
        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Check for expected output files and run
        ran_selection = False
        if (
            force
            or not self.file_galaxies.exists()
            or not self.file_galaxies_pzcut.exists()
            or not self.file_foreground.exists()
            or not self.file_background.exists()
            or not self.file_foreground_cleaned.exists()
            or not self.file_background_cleaned.exists()
            or not self.file_stars.exists()
        ):
            print(f"Running selection for variant: {self.name}")

            # Save the config
            with open(self.out_dir / "config_selector.yaml", "w") as file:
                yaml.dump(asdict(self), file, sort_keys=False)

            # Load the processed catalog
            cat = pd.read_parquet(self.in_dir / "dp1_catalog_processed.parquet")
            print("   starting length:", len(cat))

            # Only ECDFS?
            if self.only_ecdfs:
                cat = select_ecdfs(cat)
                print("   after ECDFS cut:", len(cat))

            # Cut bad pixels
            for band in ["u", "g", "r", "i", "z", "y"]:
                cat = cat.query(f"({band}5_pixel > 20) & ({band}5_pixel < 30)")
            print("   after cutting bad pixels:", len(cat))

            # Apply depth cut
            # Need to determine minimum depth such that the i_cut
            # matches the specified i-band SNR cut
            i5_cut = self.i_cut + 2.5 * np.log10(self.min_snr_i / 5)
            cat = cat.query(f"i5_pixel > {i5_cut}")
            for band in "ugrzy":
                qcut = np.quantile(
                    np.unique(cat[f"{band}5_pixel"]),
                    self.min_pixel_quantile,
                )
                cat = cat.query(f"{band}5_pixel > {qcut}")
            print("   after cutting shallow pixels:", len(cat))

            # Cut on SNR
            for band in "ugrizy":
                cat = cat.query(f"{band}_snr > @self.min_snr_{band}")
            print("   after SNR cuts:", len(cat))

            # Cut on i-band magnitude
            cat = cat.query("i_cModelMag < @self.i_cut")
            print("   after i-band magnitude cut:", len(cat))

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
            galaxies_pzcut = self.cut_photoz(galaxies)
            print("   after photo-z quality cut:", len(galaxies))

            # Foreground selection
            fg_cat = galaxies_pzcut.query(
                "z_phot > @self.fg_zmin and z_phot < @self.fg_zmax"
            )
            print("   foreground galaxies:", len(fg_cat))

            # Background selection
            bg_cat = galaxies_pzcut.query(
                "z_phot > @self.bg_zmin and z_phot < @self.bg_zmax"
            )
            print("   background galaxies:", len(bg_cat))

            # Red sequence selection
            if self.bg_red_seq:
                # Fit the red sequence
                bg_cat = bg_cat.query(self.red_seq_cut)
                print("   background galaxies after red sequence:", len(bg_cat))

            # Clean the catalogs
            print("   cleaning the foreground...")
            fg_cleaned = clean_data(
                fg_cat,
                nonuniformity=self.clean_nonuniformity,
                ztrends=self.clean_ztrends,
                outliers=self.clean_outliers,
            )
            print("   cleaning the background...")
            bg_cleaned = clean_data(
                bg_cat,
                nonuniformity=self.clean_nonuniformity,
                ztrends=self.clean_ztrends,
                outliers=self.clean_outliers,
            )

            # Save catalogs
            galaxies.to_parquet(self.file_galaxies)
            galaxies_pzcut.to_parquet(self.file_galaxies_pzcut)
            fg_cat.to_parquet(self.file_foreground)
            bg_cat.to_parquet(self.file_background)
            fg_cleaned.to_parquet(self.file_foreground_cleaned)
            bg_cleaned.to_parquet(self.file_background_cleaned)
            stars.to_parquet(self.file_stars)

            print("   selection complete")

            ran_selection = True

        else:
            print(f"Selection already done for variant: {self.name}")

        if ran_selection or not self.file_figure_photoz.exists():
            print(f"Creating photo-z plots for variant: {self.name}")
            self.create_figure_photoz()
        else:
            print(f"Photo-z plots already exist for variant: {self.name}")


# Alias default variant
Default = Selector


# Other variants
@dataclass
class iCut23p0(Default):
    name: str = "i_cut_23p0"
    i_cut: float = 23.0


@dataclass
class iCut23p5(Default):
    name: str = "i_cut_23p5"
    i_cut: float = 23.5


@dataclass
class iCut24p5(Default):
    name: str = "i_cut_24p5"
    i_cut: float = 24.5


@dataclass
class iCut25p0(Default):
    name: str = "i_cut_25p0"
    i_cut: float = 25.0


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
class BlendCut01(Default):
    name: str = "blend_cut_01"
    blendedness_cut: float = 0.01


@dataclass
class RedSeq(Default):
    name: str = "red_seq"
    bg_red_seq: bool = True


@dataclass
class EcdfsOnly(Default):
    name: str = "ecdfs_only"
    only_ecdfs: bool = True


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


@dataclass
class FgLowZ(Default):
    name: str = "fg_low_z"
    fg_zmin: float = 0.1
    fg_zmax: float = 0.4
    bg_zmin: float = 0.5
    bg_zmax: float = 1.5


@dataclass
class BgLowZ(Default):
    name: str = "bg_low_z"
    fg_zmin: float = 0.1
    fg_zmax: float = 0.4
    bg_zmin: float = 0.5
    bg_zmax: float = 1.0
