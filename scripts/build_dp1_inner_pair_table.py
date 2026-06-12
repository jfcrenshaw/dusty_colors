"""Build the DP1 default inner-bin pair table for postage-stamp review."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.postage_stamps import (
    DEFAULT_EXPECTED_INNER_PAIR_COUNT,
    DEFAULT_INNER_R_MAX_KPC,
    DEFAULT_INNER_R_MIN_KPC,
    build_projected_pair_table,
    pair_table_summary,
    write_review_labels_template,
    write_summary_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-dir",
        default=ROOT / "results" / "samples" / "dp1_default",
        type=Path,
        help="Prepared sample directory containing foreground/background parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        default=ROOT / "results" / "postage_stamps" / "dp1_default",
        type=Path,
        help="Directory for pair-table and review-template outputs.",
    )
    parser.add_argument(
        "--analysis-id",
        default="dp1_default",
        help="Prefix used in generated pair IDs.",
    )
    parser.add_argument(
        "--bin-label",
        default="inner",
        help="Label used in generated pair IDs.",
    )
    parser.add_argument(
        "--r-min-kpc",
        default=DEFAULT_INNER_R_MIN_KPC,
        type=float,
        help="Minimum projected separation in kpc.",
    )
    parser.add_argument(
        "--r-max-kpc",
        default=DEFAULT_INNER_R_MAX_KPC,
        type=float,
        help="Maximum projected separation in kpc.",
    )
    parser.add_argument(
        "--expected-count",
        default=DEFAULT_EXPECTED_INNER_PAIR_COUNT,
        type=int,
        help="Expected pair count; set to a negative value to disable the check.",
    )
    parser.add_argument(
        "--no-parquet",
        action="store_true",
        help="Only write CSV and JSON outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = _resolve_path(args.sample_dir)
    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    foreground = _read_sample(sample_dir / "foreground.parquet")
    background = _read_sample(sample_dir / "background.parquet")
    pairs = build_projected_pair_table(
        foreground,
        background,
        r_min_kpc=args.r_min_kpc,
        r_max_kpc=args.r_max_kpc,
        analysis_id=args.analysis_id,
        bin_label=args.bin_label,
    )
    if args.expected_count >= 0 and len(pairs) != args.expected_count:
        raise SystemExit(
            f"Expected {args.expected_count} pairs, but found {len(pairs)}"
        )

    csv_path = out_dir / "inner_bin_pairs.csv"
    pairs.to_csv(csv_path, index=False)
    if not args.no_parquet:
        pairs.to_parquet(out_dir / "inner_bin_pairs.parquet", index=False)

    summary = pair_table_summary(pairs)
    write_summary_json(summary, out_dir / "inner_bin_pairs_summary.json")
    write_review_labels_template(pairs, out_dir / "review_labels.csv")

    print(f"pairs: {len(pairs)}")
    print(f"wrote: {_display_path(csv_path)}")
    print(
        "theta_arcsec: "
        + ", ".join(
            f"{key}={value:.6g}" for key, value in summary["theta_arcsec"].items()
        )
    )


def _resolve_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _read_sample(path: Path) -> pd.DataFrame:
    columns = ["object_id", "ra", "dec", "z_phot", "field"]
    return pd.read_parquet(path, columns=columns)


if __name__ == "__main__":
    main()
