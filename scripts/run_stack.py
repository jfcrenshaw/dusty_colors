"""Run the YAML-first Dusty Colors stacking pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.pipeline import (  # noqa: E402
    ForceOptions,
    PipelineError,
    run_pipeline,
    run_post_run_only,
)


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
    parser.add_argument(
        "--only-postrun",
        action="store_true",
        help=(
            "Run the post-run analyses and refuse to run any stage. An ordinary "
            "run already reruns them, so this is only needed to guarantee that "
            "nothing gets rebuilt."
        ),
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

    if args.only_postrun:
        if any((force.catalog, force.sample, force.stack, force.all)):
            print(
                "Error: --only-postrun cannot be combined with a --force flag.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        try:
            outputs = run_post_run_only(config_path, root=ROOT)
        except PipelineError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
        for path in outputs:
            print(f"postrun: {path.relative_to(ROOT)}")
        return

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
