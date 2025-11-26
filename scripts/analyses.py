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
        name="snr_conservative",
        selector=selector.SNRConservative(),
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
        name="blend_cut_01",
        selector=selector.BlendCut01(),
    ),
    stacker.Default(
        name="red_sequence",
        selector=selector.RedSeq(),
    ),
    stacker.Default(
        name="pz_strict",
        selector=selector.PZStrict(),
    ),
    stacker.Default(
        name="dz_gap_0p2",
        selector=selector.DzGap0p2(),
    ),
    stacker.Default(
        name="dz_gap_0p3",
        selector=selector.DzGap0p3(),
    ),
    stacker.Default(
        name="fg_lowz",
        selector=selector.FgLowZ(),
    ),
    stacker.Default(
        name="bg_lowz",
        selector=selector.BgLowZ(),
    ),
    stacker.Nbins5(),
    stacker.Nbins20(),
    stacker.SNRMax20(),
    stacker.SNRMax500(),
    stacker.SNRMaxInf(),
    stacker.FreeFluxes(),
    # stacker.NullGalaxiesFlip(),
    # stacker.NullStars(),
    # stacker.NullStarsFlip(),
    # stacker.Default(
    #    name="i_cut_25p0_snr_i_20",
    #    selector=selector.Default(
    #        name="i_cut_25p0_snr_i_20",
    #        i_cut=25.0,
    #        min_snr_i=20.0,
    #    ),
    # ),
    stacker.Default(
        name="i_cut_25p5",
        selector=selector.Default(name="i_cut_25p5", i_cut=25.5),
    ),
    stacker.Default(
        name="snr_i_40",
        selector=selector.Default(name="snr_i_40", min_snr_i=40.0),
    ),
]

variants.append(stacker.RandPositions())
for seed in [123, 231, 312, 321, 213, 132, 11, 13, 96]:
    variants.append(
        stacker.RandPositions(name=f"rand_positions_{seed}", random_seed=seed)
    )

for variant in variants:
    variant.run()
