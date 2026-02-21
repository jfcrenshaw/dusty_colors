from dusty_colors import selector, stacker
import numpy as np


if False:
    stacker.Default(
        name="v1",
        r_bins=[0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 4, 8, 12],
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

if False:
    stacker.Default(
        name="v2",
        r_bins=[1, 2, 4, 8, 12],
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )


# I LIKE THIS ONE!
if False:
    stacker.Default(
        name="v3",
        r_bins=[1, 2, 4, 8, 12],
        selector=selector.Default(
            name="unclean_nonuniformity",
            clean_nonuniformity=False,
        ),
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# NO
if False:
    stacker.Default(
        name="v4",
        r_bins=[1, 2, 4, 8, 12],
        selector=selector.Default(
            name="unclean_ztrends",
            clean_ztrends=False,
        ),
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# NO
if False:
    stacker.Default(
        name="v5",
        r_bins=[1, 2, 4, 8, 12],
        selector=selector.Default(
            name="unclean_outliers",
            clean_outliers=False,
        ),
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# NO
if False:
    stacker.Default(
        name="v6",
        r_bins=[1, 2, 4, 8, 12],
        selector=selector.Default(
            name="unclean_all",
            clean_nonuniformity=False,
            clean_ztrends=False,
            clean_outliers=False,
        ),
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# This is a variant of v3 with more bins
if False:
    stacker.Default(
        name="v7",
        r_bins=[0.2, 0.6, 1.0, 1.4, 1.8, 2.2, 4, 8, 12],
        selector=selector.Default(
            name="unclean_nonuniformity",
            clean_nonuniformity=False,
        ),
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# This is a variant of v3 with more bins
# Dont like it because it goes out a little too far!
if False:
    stacker.Default(
        name="v8",
        r_bins=np.arange(0.2, 13, 0.4),
        r_max=14,
        selector=selector.Default(
            name="unclean_nonuniformity",
            clean_nonuniformity=False,
        ),
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# Variant of v8 that doesn't go so far out.
if False:
    stacker.Default(
        name="v9",
        r_bins=np.arange(0.2, 10.4, 0.4),
        r_max=12,
        selector=selector.Default(
            name="unclean_nonuniformity",
            clean_nonuniformity=False,
        ),
        bootstrap=True,
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# Variant of v9 with fewer bins
# NEW BASELINE!!!
if False:
    stacker.Default(
        name="v10",
        r_bins=[0.5, 1, 2, 3, 4, 5, 6],
        r_max=8,
        r_norm=4.9,
        selector=selector.Default(
            name="unclean_nonuniformity",
            clean_nonuniformity=False,
        ),
        bootstrap=True,
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# Variant of v10 with snr_max=20
if False:
    stacker.Default(
        name="v11",
        r_bins=[0.5, 1, 2, 3, 4, 5, 6],
        r_max=8,
        r_norm=4.9,
        selector=selector.Default(
            name="unclean_nonuniformity",
            clean_nonuniformity=False,
        ),
        bootstrap=True,
        snr_max=20,
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# Variant of v10 without inverse-variance weighting
if False:
    stacker.Default(
        name="v12",
        r_bins=[0.5, 1, 2, 3, 4, 5, 6],
        r_max=8,
        r_norm=4.9,
        selector=selector.Default(
            name="unclean_nonuniformity",
            clean_nonuniformity=False,
        ),
        bootstrap=True,
        weighted=False,
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# Variant of v10 with snr_max=inf
if False:
    stacker.Default(
        name="v13",
        r_bins=[0.5, 1, 2, 3, 4, 5, 6],
        r_max=8,
        r_norm=4.9,
        selector=selector.Default(
            name="unclean_nonuniformity",
            clean_nonuniformity=False,
        ),
        bootstrap=True,
        snr_max=np.inf,
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )


# Variant of v10 with red sequence foreground
if False:
    stacker.Default(
        name="v14",
        r_bins=[0.5, 1, 2, 3, 4, 5, 6],
        r_max=8,
        r_norm=4.9,
        selector=selector.Default(
            name="red_sequence",
            clean_nonuniformity=False,
            fg_query="(g_absmag - r_absmag) > 0.5",
        ),
        bootstrap=True,
        snr_max=np.inf,
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )

# Variant of v10 with blue cloud foreground
if False:
    stacker.Default(
        name="v15",
        r_bins=[0.5, 1, 2, 3, 4, 5, 6],
        r_max=8,
        r_norm=4.9,
        selector=selector.Default(
            name="blue_cloud",
            clean_nonuniformity=False,
            fg_query="(g_absmag - r_absmag) < 0.5",
        ),
        bootstrap=True,
        snr_max=np.inf,
    ).run(
        force_selector=False,
        force_pairer=False,
        force_stacker=False,
        force_plotter=True,
    )