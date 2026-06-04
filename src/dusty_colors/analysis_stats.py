"""Catalog summary statistics for resolved analysis configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import load_resolved_config
from .pipeline import build_stage_specs, check_manifest, expected_manifest

FULL_SKY_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2


@dataclass(frozen=True)
class AnalysisCatalogStats:
    """Sky-area, count, and density summary for an analysis sample."""

    analysis_id: str
    sample_id: str
    catalog_id: str
    sample_dir: Path
    nside: int
    pixel_col: str
    occupied_pixels: int
    area_deg2: float
    foreground_galaxies: int
    background_galaxies: int
    foreground_density_deg2: float
    background_density_deg2: float
    jackknife: bool
    patch_col: str | None
    dropped_foreground_bad_position: int
    dropped_background_bad_position: int
    dropped_foreground_jackknife: int
    dropped_background_jackknife: int

    def to_dict(self, *, root: str | Path | None = None) -> dict[str, Any]:
        data = asdict(self)
        sample_dir = self.sample_dir
        if root is not None:
            root_path = Path(root).resolve()
            try:
                sample_dir = sample_dir.resolve().relative_to(root_path)
            except ValueError:
                sample_dir = sample_dir.resolve()
        data["sample_dir"] = str(sample_dir)
        data["area_unit"] = "deg^2"
        data["density_unit"] = "deg^-2"
        return data


def stats_for_analysis(
    analysis_path: str | Path,
    *,
    root: str | Path | None = None,
    nside: int | None = None,
    require_current: bool = True,
    pixel_col: str = "pixel",
) -> AnalysisCatalogStats:
    """Return final stacker-facing catalog stats for an analysis YAML.

    The counts match the prepared foreground/background samples after the
    stacker's global position and jackknife-patch filtering. Per-color
    observable masks are not applied because they are mode/color-specific rather
    than a single final catalog.
    """

    root_path = Path.cwd().resolve() if root is None else Path(root).resolve()
    resolved = load_resolved_config(analysis_path, root=root_path)
    specs = build_stage_specs(resolved, root=root_path)
    sample_spec = specs["sample"]
    sample_dir = sample_spec.output_dir

    if require_current:
        check = check_manifest(
            sample_spec.output_dir,
            sample_spec.expected_outputs,
            expected_manifest(sample_spec, resolved),
        )
        if check.action != "skip":
            raise ValueError(
                "Prepared sample outputs are not current for this analysis "
                f"({check.reason}). Run scripts/run_stack.py first, or pass "
                "--allow-stale to summarize the existing parquet files."
            )

    foreground, background = _read_samples(sample_dir)
    stack_config = _stack_config(resolved.analysis.data)
    final = _finalize_for_stacking(foreground, background, stack_config)

    healpix_nside = nside or _catalog_nside(resolved.catalog.data)
    pixels = _occupied_pixels(
        final.foreground,
        final.background,
        pixel_col=pixel_col,
        nside=healpix_nside,
    )
    pixel_area = _healpix_pixel_area_deg2(healpix_nside)
    area = float(len(pixels) * pixel_area)
    if area <= 0:
        raise ValueError("Occupied HEALPix area is zero")

    foreground_count = len(final.foreground)
    background_count = len(final.background)
    return AnalysisCatalogStats(
        analysis_id=resolved.analysis.id,
        sample_id=resolved.sample.id,
        catalog_id=resolved.catalog.id,
        sample_dir=sample_dir,
        nside=healpix_nside,
        pixel_col=pixel_col,
        occupied_pixels=len(pixels),
        area_deg2=area,
        foreground_galaxies=foreground_count,
        background_galaxies=background_count,
        foreground_density_deg2=foreground_count / area,
        background_density_deg2=background_count / area,
        jackknife=final.jackknife,
        patch_col=final.patch_col,
        dropped_foreground_bad_position=final.dropped_foreground_bad_position,
        dropped_background_bad_position=final.dropped_background_bad_position,
        dropped_foreground_jackknife=final.dropped_foreground_jackknife,
        dropped_background_jackknife=final.dropped_background_jackknife,
    )


def format_stats(stats: AnalysisCatalogStats, *, root: str | Path | None = None) -> str:
    """Format stats as a compact, human-readable report."""

    data = stats.to_dict(root=root)
    lines = [
        f"analysis: {data['analysis_id']}",
        f"sample: {data['sample_id']} ({data['sample_dir']})",
        f"catalog: {data['catalog_id']}",
        f"nside: {data['nside']}",
        f"pixel_col: {data['pixel_col']}",
        f"occupied_pixels: {data['occupied_pixels']}",
        f"area_deg2: {data['area_deg2']:.6f}",
        f"foreground_galaxies: {data['foreground_galaxies']}",
        f"background_galaxies: {data['background_galaxies']}",
        f"foreground_density_deg^-2: {data['foreground_density_deg2']:.6f}",
        f"background_density_deg^-2: {data['background_density_deg2']:.6f}",
    ]

    dropped = {
        "foreground_bad_position": stats.dropped_foreground_bad_position,
        "background_bad_position": stats.dropped_background_bad_position,
        "foreground_jackknife": stats.dropped_foreground_jackknife,
        "background_jackknife": stats.dropped_background_jackknife,
    }
    if any(dropped.values()):
        lines.append("dropped_rows:")
        lines.extend(f"  {name}: {count}" for name, count in dropped.items())

    return "\n".join(lines)


def save_analysis_catalog_stats(
    analysis_path: str | Path,
    output_dir: str | Path,
    *,
    root: str | Path | None = None,
    nside: int | None = None,
    require_current: bool = True,
    pixel_col: str = "pixel",
) -> tuple[Path, Path]:
    """Write catalog stats text and JSON reports for an analysis."""

    root_path = Path.cwd().resolve() if root is None else Path(root).resolve()
    stats = stats_for_analysis(
        analysis_path,
        root=root_path,
        nside=nside,
        require_current=require_current,
        pixel_col=pixel_col,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    text_path = output_path / "analysis_catalog_stats.txt"
    json_path = output_path / "analysis_catalog_stats.json"
    text_path.write_text(format_stats(stats, root=root_path) + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(stats.to_dict(root=root_path), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return text_path, json_path


@dataclass(frozen=True)
class _FinalSamples:
    foreground: pd.DataFrame
    background: pd.DataFrame
    jackknife: bool
    patch_col: str | None
    dropped_foreground_bad_position: int
    dropped_background_bad_position: int
    dropped_foreground_jackknife: int
    dropped_background_jackknife: int


def _read_samples(sample_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    foreground_path = sample_dir / "foreground.parquet"
    background_path = sample_dir / "background.parquet"
    missing = [path for path in (foreground_path, background_path) if not path.exists()]
    if missing:
        missing_names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Prepared sample outputs are missing: {missing_names}")
    return pd.read_parquet(foreground_path), pd.read_parquet(background_path)


def _stack_config(analysis_data: Mapping[str, Any]) -> Mapping[str, Any]:
    stack_config = analysis_data.get("stack", {})
    if not isinstance(stack_config, Mapping):
        raise ValueError("Analysis config 'stack' must be a mapping")
    return stack_config


def _finalize_for_stacking(
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    stack_config: Mapping[str, Any],
) -> _FinalSamples:
    foreground, dropped_fg_bad = _drop_bad_foreground_positions(foreground)
    background, dropped_bg_bad = _drop_bad_background_positions(background)
    if len(foreground) == 0 or len(background) == 0:
        raise ValueError("Foreground and background samples must both be non-empty")

    jackknife = bool(stack_config.get("jackknife", True))
    patch_col = None
    dropped_fg_jk = 0
    dropped_bg_jk = 0
    if jackknife:
        patch_col = str(stack_config.get("patch_col", "jackknife_region"))
        foreground, background, dropped_fg_jk, dropped_bg_jk = (
            _keep_shared_jackknife_patches(foreground, background, patch_col)
        )

    return _FinalSamples(
        foreground=foreground,
        background=background,
        jackknife=jackknife,
        patch_col=patch_col,
        dropped_foreground_bad_position=dropped_fg_bad,
        dropped_background_bad_position=dropped_bg_bad,
        dropped_foreground_jackknife=dropped_fg_jk,
        dropped_background_jackknife=dropped_bg_jk,
    )


def _drop_bad_foreground_positions(
    foreground: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    required = {"ra", "dec", "z_phot"}
    _require_columns(foreground, required, "foreground")
    z_phot = foreground["z_phot"].to_numpy(float)
    good = (
        np.isfinite(foreground["ra"].to_numpy(float))
        & np.isfinite(foreground["dec"].to_numpy(float))
        & np.isfinite(z_phot)
        & (z_phot > 0)
    )
    return foreground.loc[good].reset_index(drop=True), int(np.sum(~good))


def _drop_bad_background_positions(
    background: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    _require_columns(background, {"ra", "dec"}, "background")
    good = np.isfinite(background["ra"].to_numpy(float)) & np.isfinite(
        background["dec"].to_numpy(float)
    )
    return background.loc[good].reset_index(drop=True), int(np.sum(~good))


def _keep_shared_jackknife_patches(
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    patch_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    _require_columns(foreground, {patch_col}, "foreground")
    _require_columns(background, {patch_col}, "background")

    fg_patch = foreground[patch_col].to_numpy(int)
    bg_patch = background[patch_col].to_numpy(int)
    patches = np.intersect1d(np.unique(fg_patch), np.unique(bg_patch))
    if len(patches) < 2:
        raise ValueError(
            "TreeCorr jackknife covariance needs at least two shared patches"
        )

    fg_keep = np.isin(fg_patch, patches)
    bg_keep = np.isin(bg_patch, patches)
    dropped_fg = int(np.sum(~fg_keep))
    dropped_bg = int(np.sum(~bg_keep))
    return (
        foreground.loc[fg_keep].reset_index(drop=True),
        background.loc[bg_keep].reset_index(drop=True),
        dropped_fg,
        dropped_bg,
    )


def _occupied_pixels(
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    *,
    pixel_col: str,
    nside: int,
) -> np.ndarray:
    _require_columns(foreground, {pixel_col}, "foreground")
    _require_columns(background, {pixel_col}, "background")

    values = pd.concat(
        [foreground[pixel_col], background[pixel_col]],
        ignore_index=True,
    )
    if values.isna().any():
        raise ValueError(f"Cannot compute occupied area with null {pixel_col} values")

    pixels = np.unique(values.to_numpy(int))
    max_pixel = _healpix_npix(nside)
    invalid = pixels[(pixels < 0) | (pixels >= max_pixel)]
    if len(invalid):
        raise ValueError(
            f"Pixel values are outside the valid range for nside={nside}: "
            f"{invalid[:5].tolist()}"
        )
    return pixels


def _catalog_nside(catalog_data: Mapping[str, Any]) -> int:
    footprint = catalog_data.get("footprint", {})
    if isinstance(footprint, Mapping) and "nside" in footprint:
        return int(footprint["nside"])
    raise ValueError(
        "Could not determine HEALPix nside from catalog.footprint.nside; "
        "pass --nside."
    )


def _healpix_npix(nside: int) -> int:
    if nside < 1:
        raise ValueError("HEALPix nside must be positive")
    return 12 * nside * nside


def _healpix_pixel_area_deg2(nside: int) -> float:
    return FULL_SKY_DEG2 / _healpix_npix(nside)


def _require_columns(
    catalog: pd.DataFrame,
    columns: set[str],
    label: str,
) -> None:
    missing = sorted(columns - set(catalog.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
