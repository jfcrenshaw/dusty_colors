"""Main script for running analysis variants"""

from dusty_colors import selector, stacker


variants = [
    stacker.Default(),
    stacker.Default(
        name="i_cut_23p0",
        selector=selector.iCut23p0(),
    ),
    stacker.Default(
        name="i_cut_23p5",
        selector=selector.iCut23p5(),
    ),
    stacker.Default(
        name="i_cut_24p5",
        selector=selector.iCut24p5(),
    ),
    stacker.Default(
        name="i_cut_25p0",
        selector=selector.iCut25p0(),
    ),
    stacker.Default(
        name="snr_i_20",
        selector=selector.SNRi20(),
    ),
    stacker.Default(
        name="snr_relaxed",
        selector=selector.SNRRelaxed(),
    ),
    stacker.Default(
        name="snr_conservative",
        selector=selector.SNRConservative(),
    ),
    stacker.Default(
        name="bright_cut_16",
        selector=selector.BrightCut16(),
    ),
    stacker.Default(
        name="bright_cut_20",
        selector=selector.BrightCut20(),
    ),
    stacker.Default(
        name="bright_mask_r20",
        selector=selector.BrightMaskR20(),
    ),
    stacker.Default(
        name="bright_mask_r60",
        selector=selector.BrightMaskR60(),
    ),
    stacker.Default(
        name="blend_cut_100",
        selector=selector.BlendCut100(),
    ),
    stacker.Default(
        name="blend_cut_10",
        selector=selector.BlendCut10(),
    ),
    stacker.Default(
        name="dz_gap_0p1",
        selector=selector.DzGap0p1(),
    ),
    stacker.Default(
        name="dz_gap_0p3",
        selector=selector.DzGap0p3(),
    ),
    stacker.Default(
        name="pz_relaxed",
        selector=selector.PZRelaxed(),
    ),
    stacker.Default(
        name="fg_red_sequence",
        selector=selector.FGRedSequence(),
    ),
    stacker.Default(
        name="fg_blue_cloud",
        selector=selector.FGBlueCloud(),
    ),
]

for variant in variants:
    variant.run()
