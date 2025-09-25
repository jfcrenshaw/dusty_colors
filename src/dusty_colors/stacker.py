"""Class to perform stacking."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass
class Stacker:
    """Class to define variants for stacking."""

    name: str = "default"

    bin_by_angle: bool = False
    fg_stars: bool = False  # Whether to use stars as foreground objects
    flip: bool = False  # Whether to flip foreground/background samples
    free_fluxes: bool = False  # Whether to use "free" flux variants
    stack_mags: bool = False  # Whether to stack in mags instead of fluxes

    def run(self, force: bool = False) -> None:
        """Run stacking."""
        # Set directories
        out_dir = Path(f"results/{self.name}")
        in_dir = out_dir  # Stacker input is selector output

        # Create output directory
        out_dir.mkdir(parents=True, exist_ok=True)

        # Check for expected output files and run
        flux = out_dir / "stack_flux.npz"
        color = out_dir / "stack_color.npz"
        shear = out_dir / "stack_shear.npz"
        if force or not flux.exists() or not color.exists() or not shear.exists():
            print(f"Running stacking for variant: {self.name}")

            # Save the config
            with open(out_dir / "config_stacker.yaml", "w") as file:
                yaml.dump(asdict(self), file, sort_keys=False)

            # Run stacking
            ...

            # Create plots to visualize stacking
            ...
        else:
            print(f"Stacking already done for variant: {self.name}")
