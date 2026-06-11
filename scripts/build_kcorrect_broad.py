"""Build a saved kcorrect model from the Pai & Blanton broad templates."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.enrichments import KcorrectEnrichment

BANDS = ("u", "g", "r", "i", "z", "y")
DEFAULT_BANDS = ("g", "r", "i", "z")
DEFAULT_TAG = "v1.9.1"
DEFAULT_TEMPLATES = "data/kcorrect/templates_broad.fits"
DEFAULT_OUTPUT = "data/kcorrect/kcorrect_broad.fits"
DEFAULT_REDSHIFT_RANGE = (0.0, 0.5)
DEFAULT_NREDSHIFT = 1000


@dataclass(frozen=True)
class BuildConfig:
    templates: Path
    output: Path
    responses: list[str]
    bands: tuple[str, ...]
    redshift_range: tuple[float, float]
    nredshift: int
    abcorrect: bool
    interpolate_templates: bool
    force: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--band",
        action="append",
        choices=BANDS,
        help=(
            "Rubin band to include; may be repeated. Defaults to griz, in that "
            "order."
        ),
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="Rubin bandpass tag used in data/bandpasses filenames.",
    )
    parser.add_argument(
        "--templates",
        default=DEFAULT_TEMPLATES,
        help="Pai & Blanton broad template FITS file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output kcorrect FITS model.",
    )
    parser.add_argument(
        "--redshift-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=DEFAULT_REDSHIFT_RANGE,
        help="Redshift grid bounds for the precomputed kcorrect model.",
    )
    parser.add_argument(
        "--nredshift",
        type=int,
        default=DEFAULT_NREDSHIFT,
        help="Number of redshift grid points.",
    )
    parser.add_argument(
        "--abcorrect",
        action="store_true",
        help="Ask kcorrect to apply AB corrections when building the model.",
    )
    parser.add_argument(
        "--interpolate-templates",
        action="store_true",
        help=(
            "Store the full template interpolation object. This is faster at "
            "runtime but substantially more memory-hungry."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved build inputs without running kcorrect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = config_from_args(args)
        print_build_summary(config)
        if args.dry_run:
            return
        try:
            preflight(config)
        except FileExistsError:
            print()
            print(f"Refusing to replace existing output: {config.output}", flush=True)
            print("Run again with --force to rebuild and replace it.", flush=True)
            raise SystemExit(1)
        output = build_response_file(config)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Wrote {output}")


def config_from_args(args: argparse.Namespace) -> BuildConfig:
    bands = normalize_bands(args.band or list(DEFAULT_BANDS))
    validate_tag(args.tag)
    redshift_range = tuple(float(value) for value in args.redshift_range)
    if redshift_range[0] >= redshift_range[1]:
        raise ValueError("--redshift-range MIN must be less than MAX")
    if int(args.nredshift) < 2:
        raise ValueError("--nredshift must be at least 2")

    return BuildConfig(
        templates=resolve_repo_path(args.templates),
        output=resolve_repo_path(args.output),
        responses=responses_for_bands(bands, args.tag),
        bands=bands,
        redshift_range=(redshift_range[0], redshift_range[1]),
        nredshift=int(args.nredshift),
        abcorrect=bool(args.abcorrect),
        interpolate_templates=bool(args.interpolate_templates),
        force=bool(args.force),
    )


def normalize_bands(bands: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out = []
    for band in bands:
        if band not in BANDS:
            raise ValueError(f"Unknown Rubin band: {band}")
        if band in seen:
            raise ValueError(f"Duplicate Rubin band: {band}")
        seen.add(band)
        out.append(band)
    return tuple(out)


def responses_for_bands(bands: tuple[str, ...], tag: str) -> list[str]:
    paths = [
        ROOT / "data" / "bandpasses" / f"rubin_bandpass_{band}_{tag}.dat"
        for band in bands
    ]
    return KcorrectEnrichment.resolve_response_names([str(path) for path in paths])


def build_response_file(config: BuildConfig) -> Path:
    preflight(config)

    config.output.parent.mkdir(parents=True, exist_ok=True)
    prepare_runtime_cache()
    Template, Kcorrect = load_kcorrect_classes()
    tmp_path = temporary_output_path(config.output)

    try:
        print("Loading Pai & Blanton broad templates...", flush=True)
        templates = Template(
            filename=str(config.templates),
            interpolate=config.interpolate_templates,
        )
        print(
            "Precomputing kcorrect response matrices; this can take a while...",
            flush=True,
        )
        kc = Kcorrect(
            responses=config.responses,
            templates=templates,
            redshift_range=list(config.redshift_range),
            nredshift=config.nredshift,
            abcorrect=config.abcorrect,
        )
        print(f"Writing temporary model: {tmp_path}", flush=True)
        kc.tofits(str(tmp_path))
        print("Validating saved response list...", flush=True)
        validate_saved_model(tmp_path, config.responses)
        print(f"Replacing output: {config.output}", flush=True)
        tmp_path.replace(config.output)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return config.output


def preflight(config: BuildConfig) -> None:
    ensure_inputs_exist(config)
    if config.output.exists() and not config.force:
        raise FileExistsError(f"{config.output} exists; pass --force to replace it")


def ensure_inputs_exist(config: BuildConfig) -> None:
    missing = [config.templates]
    missing.extend(Path(f"{response}.dat") for response in config.responses)
    missing = [path for path in missing if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing kcorrect build input(s):\n{formatted}")


def validate_saved_model(path: Path, responses: list[str]) -> None:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        saved = [str(value[0]) for value in hdul["RESPONSES"].data]
    if saved != responses:
        raise ValueError(
            "Saved kcorrect model responses do not match requested responses: "
            f"{saved} != {responses}"
        )


def load_kcorrect_classes() -> tuple[type, type]:
    from kcorrect.kcorrect import Kcorrect
    from kcorrect.template import Template

    return Template, Kcorrect


def prepare_runtime_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "dusty_colors_kcorrect_cache"
    matplotlib_cache = cache_root / "matplotlib"
    xdg_cache = cache_root / "xdg"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))


def temporary_output_path(output: Path) -> Path:
    return output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")


def resolve_repo_path(path: str | Path) -> Path:
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    return out.resolve()


def validate_tag(tag: str) -> None:
    if "/" in tag or "\\" in tag:
        raise ValueError("Rubin bandpass tag must not contain path separators")


def print_build_summary(config: BuildConfig) -> None:
    print("Building kcorrect model", flush=True)
    print(f"  templates: {config.templates}", flush=True)
    print(f"  output: {config.output}", flush=True)
    print(f"  bands: {''.join(config.bands)}", flush=True)
    print(f"  responses: {len(config.responses)}", flush=True)
    for response in config.responses:
        print(f"    - {response}", flush=True)
    print(
        "  redshift grid: "
        f"{config.redshift_range[0]}..{config.redshift_range[1]} "
        f"({config.nredshift} points)",
        flush=True,
    )
    print(f"  interpolate templates: {config.interpolate_templates}", flush=True)


if __name__ == "__main__":
    main()
