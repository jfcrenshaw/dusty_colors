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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stacker = TreeCorrStacker(
        name=args.name,
        selector=selector.Default(),
    )
    stacker.run(
        force_selector=args.force_selector,
        force_stacker=args.force_stacker,
    )


if __name__ == "__main__":
    main()
