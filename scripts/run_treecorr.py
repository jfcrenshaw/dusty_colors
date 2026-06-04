"""Run the TreeCorr stacker with the default selector."""

import argparse

from dusty_colors import selector
from dusty_colors.treecorr_stacker import TreeCorrStacker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default="treecorr",
        help="Name for the stack output directory under results/stacks.",
    )
    parser.add_argument(
        "--force-selector",
        action="store_true",
        help="Re-run catalog selection before stacking.",
    )
    parser.add_argument(
        "--force-stacker",
        action="store_true",
        help="Re-run TreeCorr stacks even if output files already exist.",
    )
    parser.add_argument(
        "--random-correction",
        dest="random_correction",
        action="store_true",
        default=True,
        help=(
            "Subtract the same estimator measured around random "
            "HEALPix-footprint positions. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no-random-correction",
        dest="random_correction",
        action="store_false",
        help="Disable random subtraction for diagnostic runs.",
    )
    parser.add_argument(
        "--random-multiplier",
        type=float,
        default=5.0,
        help="Number of random positions to generate per real object in each patch.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used for HEALPix-footprint random positions.",
    )
    parser.add_argument(
        "--r-aper-min",
        type=float,
        default=2.0,
        help="Inner radius of the reference annulus in Mpc.",
    )
    parser.add_argument(
        "--r-aper-max",
        type=float,
        default=4.0,
        help="Outer radius of the reference annulus in Mpc.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stacker = TreeCorrStacker(
        name=args.name,
        selector=selector.Default(),
        random_correction=args.random_correction,
        random_multiplier=args.random_multiplier,
        random_seed=args.random_seed,
        r_aper_min=args.r_aper_min,
        r_aper_max=args.r_aper_max,
    )
    stacker.run(
        force_selector=args.force_selector,
        force_stacker=args.force_stacker,
    )


if __name__ == "__main__":
    main()
