from dataclasses import dataclass, field
from types import SimpleNamespace

from .selector import Selector
from .stacker import Stacker


class VariantList(SimpleNamespace):
    """Container for analysis variants."""

    def __iter__(self):
        return iter(self.__dict__.items())


@dataclass
class AnalysisVariant:
    """Class to define a full analysis variant."""

    name: str = "default"

    selector: Selector = field(default_factory=Selector)
    stacking: Stacker = field(default_factory=Stacker)

    def run(self, force_selection: bool = False, force_stacking: bool = False) -> None:
        """Run the analysis for this variant."""
        self.selector.run()
        self.stacking.run()


# Define selection variants
selection_variants = VariantList()
selection_variants.default = Selector()
selection_variants.bright_cut_20 = Selector(name="bright_cut_20", bright_cut=20.0)
selection_variants.bright_mask_r20 = Selector(name="bright_mask_r20", bright_radius=20)
selection_variants.bright_mask_r50 = Selector(name="bright_mask_r50", bright_radius=50)
selection_variants.blend_cut_40 = Selector(name="blend_cut_40", blendedness_cut=0.40)
selection_variants.blend_cut_30 = Selector(name="blend_cut_30", blendedness_cut=0.30)
selection_variants.blend_cut_20 = Selector(name="blend_cut_20", blendedness_cut=0.20)
selection_variants.blend_cut_10 = Selector(name="blend_cut_10", blendedness_cut=0.10)
selection_variants.snr_i_20 = Selector(name="snr_i_20", min_snr_i=20)
selection_variants.snr_conservative = Selector(
    name="snr_conservative",
    min_snr_u=5,
    min_snr_g=10,
    min_snr_r=10,
    min_snr_i=20,
    min_snr_z=10,
    min_snr_y=5,
)
selection_variants.no_tomo = Selector(name="no_tomo", tomographic=False)
selection_variants.red_seq = Selector(name="red_seq", bg_red_seq=True)
selection_variants.ecdfs_only = Selector(name="ecdfs_only", only_ecdfs=True)
selection_variants.pz_gap_0p2 = Selector(name="pz_gap_0p2", bg_zmin=0.7)
selection_variants.pz_gap_0p3 = Selector(name="pz_gap_0p3", bg_zmin=0.8)
selection_variants.fg_low_z = Selector(name="fg_low_z", fg_zmin=0.1)
selection_variants.bg_low_z = Selector(
    name="bg_low_z", fg_zmin=0.1, fg_zmax=0.4, bg_zmin=0.5
)

# Define stacking variants
stacking_variants = VariantList()
stacking_variants.default = Stacker()
stacking_variants.bin_by_angle = Stacker(name="bin_by_angle", bin_by_angle=True)
stacking_variants.free_fluxes = Stacker(name="free_fluxes", free_fluxes=True)
stacking_variants.null_galaxies_flip = Stacker(name="null_galaxies_flip", flip=True)
stacking_variants.null_stars = Stacker(name="null_stars", fg_stars=True)
stacking_variants.null_stars_flip = Stacker(
    name="null_stars_flip", fg_stars=True, flip=True
)
stacking_variants.stack_mags = Stacker(name="stack_mags", stack_mags=True)
