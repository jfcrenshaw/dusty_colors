from dusty_colors import selector, stacker

variants = [
    stacker.Default(),
    stacker.Default(
        name="snr_conservative",
        selector=selector.SNRConservative(),
    ),
    stacker.Default(
        name="snr_i_20",
        selector=selector.SNRi20(),
    ),
    stacker.SNRMax20(),
    stacker.SNRMax500(),
    stacker.SNRMaxInf(),
    stacker.Default(
        name="ecdfs_only",
        selector=selector.EcdfsOnly(),
    ),
    stacker.Default(
        name="red_sequence",
        selector=selector.RedSeq(),
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
        name="bright_mask_r50",
        selector=selector.BrightMaskR50(),
    ),
    stacker.Default(
        name="bright_mask_r100",
        selector=selector.BrightMaskR100(),
    ),
    stacker.Default(
        name="blend_cut_100",
        selector=selector.BlendCut100(),
    ),
    stacker.Default(
        name="blend_cut_20",
        selector=selector.BlendCut20(),
    ),
    stacker.Default(
        name="blend_cut_10",
        selector=selector.BlendCut10(),
    ),
    stacker.Default(
        name="fg_lowz",
        selector=selector.FgLowZ(),
    ),
    stacker.Default(
        name="bg_lowz",
        selector=selector.BgLowZ(),
    ),
    stacker.Default(
        name="DzGap0p2",
        selector=selector.DzGap0p2(),
    ),
    stacker.Default(
        name="DzGap0p3",
        selector=selector.DzGap0p3(),
    ),
    stacker.Default(
        name="pz_strict",
        selector=selector.PZStrict(),
    ),
    stacker.FreeFluxes(),
    stacker.NullGalaxiesFlip(),
    # stacker.NullStars(),
    # stacker.NullStarsFlip(),
    # stacker.BinByAngle(),
    # stacker.StackMags(),
]

for variant in variants:
    variant.run()
