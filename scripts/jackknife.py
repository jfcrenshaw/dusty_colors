from dusty_colors import selector, stacker
import numpy as np

variants = [
    stacker.Default(
        name=f"jackknife_{i}", exclude_jk=i, r_bins=np.geomspace(2e-2, 20, 10)
    )
    for i in range(0, 18)
]

for variant in variants:
    variant.run()
