"""Run the YAML-first Dusty Colors stacking pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.pipeline import ForceOptions, PipelineError, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to an analysis YAML config.")
    parser.add_argument(
        "--force-catalog",
        action="store_true",
        help="Recompute catalog outputs plus dependent sample and stack outputs.",
    )
    parser.add_argument(
        "--force-sample",
        action="store_true",
        help="Recompute sample outputs plus dependent stack outputs.",
    )
    parser.add_argument(
        "--force-stack",
        action="store_true",
        help="Recompute stack outputs only.",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Recompute every stage in the resolved graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute() and not config_path.exists():
        config_path = ROOT / config_path

    force = ForceOptions(
        catalog=args.force_catalog,
        sample=args.force_sample,
        stack=args.force_stack,
        all=args.force_all,
    )

    try:
        result = run_pipeline(config_path, root=ROOT, force=force)
    except PipelineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    for stage in result.stages:
        print(
            f"{stage.kind}: {stage.action} ({stage.reason}) -> "
            f"{stage.output_dir.relative_to(ROOT)}"
        )


if __name__ == "__main__":
    main()
