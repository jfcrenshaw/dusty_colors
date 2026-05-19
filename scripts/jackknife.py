from dusty_colors import selector, stacker
import numpy as np

variants = [
    stacker.Default(
        name=f"jackknife_{i}",
        exclude_jk=i,
        r_bins=np.geomspace(1e-2, 1, 5),
        r_aper_min=2,
        r_aper_max=4,
    )
    for i in range(0, 9)
]

for variant in variants:
    variant.run(force_stacker=True)
