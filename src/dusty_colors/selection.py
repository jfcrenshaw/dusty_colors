"""Sample selection for canonical catalogs."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cleaning import apply_minimal_cleaning, clean_sample
from .footprint import assign_jackknife_regions

MAGSYS_ZEROPOINT = 31.4
FULL_SKY_DEG2 = 4.0 * np.pi * (180.0 / np.pi) ** 2


class SampleSelector:
    """Select foreground/background samples from one canonical catalog.

    The order mirrors the analysis logic in the sample YAML: start from valid
    catalog rows, apply cuts shared by both samples, then make foreground and
    background samples with their own optional cleaning.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        config: Mapping[str, Any],
        *,
        bands: list[str] | None = None,
        photometry: str | None = None,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.selection = dict(config.get("selection", {}))
        self.cleaning = dict(config.get("cleaning", {}))
        self.bands = bands
        self.photometry = photometry

    def select(self) -> dict[str, pd.DataFrame]:
        return self.select_with_report().samples

    def select_with_report(self) -> SampleSelectionResult:
        selected = self._valid_catalog_rows()
        stages = [
            _stage_summary("catalog_input", self.catalog),
            _stage_summary("minimal_valid_rows", selected),
        ]
        selected, shared_stages = _trace_shared_selection(selected, self.selection)
        stages.extend(shared_stages)
        selected = self._assign_jackknife(selected)
        stages.append(_stage_summary("jackknife_assignment", selected))
        footprint = self._footprint(selected)

        foreground, foreground_stages = self._sample_with_stages(selected, "foreground")
        background, background_stages = self._sample_with_stages(selected, "background")
        if len(foreground) == 0 or len(background) == 0:
            raise ValueError("Foreground and background samples must both be non-empty")

        samples = {"foreground": foreground, "background": background}
        if footprint is not None:
            samples["footprint"] = footprint
        report = _sample_report(
            self.config,
            samples,
            stages=stages,
            foreground_stages=foreground_stages,
            background_stages=background_stages,
        )
        return SampleSelectionResult(samples=samples, report=report)

    def _valid_catalog_rows(self) -> pd.DataFrame:
        return apply_minimal_cleaning(
            self.catalog,
            bands=self.bands,
            photometry=self.photometry,
        )

    def _apply_shared_selection(self, catalog: pd.DataFrame) -> pd.DataFrame:
        mask = SharedSelectionCuts(catalog, self.selection).mask()
        return catalog.loc[mask].reset_index(drop=True)

    def _assign_jackknife(self, catalog: pd.DataFrame) -> pd.DataFrame:
        jackknife = _enabled_option(self.config, "jackknife")
        if jackknife is None:
            return catalog
        return _assign_sample_jackknife(catalog, jackknife)

    def _footprint(self, catalog: pd.DataFrame) -> pd.DataFrame | None:
        return _sample_footprint(catalog, self.config.get("footprint"))

    def _sample(self, catalog: pd.DataFrame, sample: str) -> pd.DataFrame:
        return self._sample_with_stages(catalog, sample)[0]

    def _sample_with_stages(
        self,
        catalog: pd.DataFrame,
        sample: str,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        sample_catalog = _redshift_slice(catalog, self.selection[f"{sample}_z"])
        stages = [_stage_summary(f"{sample}_redshift_window", sample_catalog)]
        query = self.selection.get(f"{sample}_query")
        if query:
            sample_catalog = sample_catalog.query(str(query)).reset_index(drop=True)
            stages.append(_stage_summary(f"{sample}_query", sample_catalog))
        cleaned = clean_sample(
            sample_catalog,
            _cleaning_for_sample(self.cleaning, sample),
        )
        stages.append(_stage_summary(f"{sample}_cleaning", cleaned))
        return cleaned, stages


class SampleSelectionResult:
    """Prepared samples plus their human/report metadata."""

    def __init__(
        self,
        *,
        samples: dict[str, pd.DataFrame],
        report: dict[str, Any],
    ) -> None:
        self.samples = samples
        self.report = report


def select_samples(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    bands: list[str] | None = None,
    photometry: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Return foreground/background samples from an in-memory catalog."""
    return SampleSelector(
        catalog,
        config,
        bands=bands,
        photometry=photometry,
    ).select()


def select_samples_with_report(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    bands: list[str] | None = None,
    photometry: str | None = None,
) -> SampleSelectionResult:
    """Return foreground/background samples and a stage-by-stage report."""
    return SampleSelector(
        catalog,
        config,
        bands=bands,
        photometry=photometry,
    ).select_with_report()


def write_sample_outputs(
    samples: Mapping[str, pd.DataFrame],
    output_dir: str | Path,
    *,
    report: Mapping[str, Any] | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples["foreground"].to_parquet(output_dir / "foreground.parquet", index=False)
    samples["background"].to_parquet(output_dir / "background.parquet", index=False)
    if "footprint" in samples:
        samples["footprint"].to_parquet(output_dir / "footprint.parquet", index=False)
    if report is not None:
        write_sample_report(report, output_dir)


def write_sample_report(report: Mapping[str, Any], output_dir: str | Path) -> None:
    """Write readable and machine-readable sample selection reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_report.md").write_text(
        format_sample_report(report),
        encoding="utf-8",
    )
    (output_dir / "sample_report.json").write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_sample(
    catalog_path_or_dir: str | Path,
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    bands: list[str] | None = None,
    photometry: str | None = None,
    nside: int | None = None,
) -> None:
    """Select and write foreground/background samples from canonical parquet."""
    path = Path(catalog_path_or_dir)
    catalog_path = path / "catalog.parquet" if path.is_dir() else path
    catalog = pd.read_parquet(catalog_path)
    result = select_samples_with_report(
        catalog,
        config,
        bands=bands,
        photometry=photometry,
    )
    if nside is not None:
        result.report["nside"] = int(nside)
        _add_area_metrics(result.report, result.samples, int(nside))
    write_sample_outputs(result.samples, output_dir, report=result.report)


def _trace_shared_selection(
    catalog: pd.DataFrame,
    selection: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    mask = np.ones(len(catalog), dtype=bool)
    stages: list[dict[str, Any]] = []

    if selection.get("shared_query"):
        mask &= catalog.eval(str(selection["shared_query"])).astype(bool).to_numpy()
        stages.append(_stage_summary("shared_query", catalog.loc[mask]))

    pixel_depth_cuts = _enabled_option(selection, "pixel_depth_cuts")
    if pixel_depth_cuts is not None:
        if not isinstance(pixel_depth_cuts, Mapping):
            raise ValueError("selection.pixel_depth_cuts must be a mapping")
        depth_cuts = PixelDepthCuts(catalog, pixel_depth_cuts, mask)
        if pixel_depth_cuts.get("valid_range"):
            mask = depth_cuts._apply_valid_range(mask)
            stages.append(_stage_summary("pixel_depth_valid_range", catalog.loc[mask]))
        if pixel_depth_cuts.get("min_occupancy") is not None:
            mask = depth_cuts._apply_min_occupancy(mask)
            stages.append(
                _stage_summary("pixel_depth_min_occupancy", catalog.loc[mask])
            )
        if pixel_depth_cuts.get("complete_to"):
            mask = depth_cuts._apply_completeness(mask)
            stages.append(_stage_summary("pixel_depth_complete_to", catalog.loc[mask]))
        if pixel_depth_cuts.get("drop_shallowest"):
            mask = depth_cuts._drop_shallowest_pixels(mask)
            stages.append(
                _stage_summary("pixel_depth_drop_shallowest", catalog.loc[mask])
            )

    pixel_occupancy_min = _enabled_option(selection, "pixel_occupancy_min")
    if pixel_occupancy_min is not None:
        mask &= _pixel_occupancy_mask(catalog, pixel_occupancy_min, mask)
        stages.append(_stage_summary("pixel_occupancy_min", catalog.loc[mask]))

    snr_min = _enabled_option(selection, "snr_min")
    if snr_min is not None:
        mask &= _snr_min_mask(catalog, snr_min)
        stages.append(_stage_summary("snr_min", catalog.loc[mask]))

    blendedness_max = _enabled_option(selection, "blendedness_max")
    if blendedness_max is not None:
        mask &= _max_column_mask(
            catalog,
            blendedness_max,
            default_column="blendedness_i",
            option_name="blendedness_max",
        )
        stages.append(_stage_summary("blendedness_max", catalog.loc[mask]))

    magnitude_limits = _enabled_option(selection, "magnitude_limits")
    if magnitude_limits is not None:
        mask &= _magnitude_limit_mask(catalog, magnitude_limits)
        stages.append(_stage_summary("magnitude_limits", catalog.loc[mask]))

    if "photoz_max_sigma" in selection:
        mask &= catalog["z_phot"].notna().to_numpy()
        mask &= catalog["z_phot_err"].to_numpy(float) < float(
            selection["photoz_max_sigma"]
        )
        stages.append(_stage_summary("photoz_max_sigma", catalog.loc[mask]))

    photoz_estimate_sigma = _enabled_option(selection, "photoz_estimate_max_sigma")
    if photoz_estimate_sigma is not None:
        mask &= _photoz_estimate_sigma_mask(catalog, photoz_estimate_sigma)
        stages.append(_stage_summary("photoz_estimate_max_sigma", catalog.loc[mask]))

    photoz_diff_norm = _enabled_option(selection, "photoz_max_diff_norm")
    if photoz_diff_norm is not None:
        mask &= _photoz_diff_norm_mask(catalog, photoz_diff_norm)
        stages.append(_stage_summary("photoz_max_diff_norm", catalog.loc[mask]))

    if "photoz_max_sigma_norm" in selection:
        max_sigma = float(selection["photoz_max_sigma_norm"])
        mask &= catalog["z_phot_err"].to_numpy(float) < max_sigma * (
            1.0 + catalog["z_phot"].to_numpy(float)
        )
        stages.append(_stage_summary("photoz_max_sigma_norm", catalog.loc[mask]))

    return catalog.loc[mask].reset_index(drop=True), stages


def _stage_summary(name: str, catalog: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"stage": name, "rows": int(len(catalog))}
    if "field" in catalog:
        counts = catalog["field"].fillna("unknown").value_counts().sort_index()
        summary["field_counts"] = {
            str(field): int(count) for field, count in counts.items()
        }
    if "pixel" in catalog:
        summary["unique_pixels"] = int(catalog["pixel"].dropna().nunique())
    return summary


def _sample_report(
    config: Mapping[str, Any],
    samples: Mapping[str, pd.DataFrame],
    *,
    stages: list[dict[str, Any]],
    foreground_stages: list[dict[str, Any]],
    background_stages: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "sample_id": str(config.get("id", "")),
        "catalog": str(config.get("catalog", "")),
        "stages": stages,
        "foreground_stages": foreground_stages,
        "background_stages": background_stages,
        "final": {
            "foreground_rows": int(len(samples["foreground"])),
            "background_rows": int(len(samples["background"])),
        },
    }
    if "footprint" in samples:
        report["final"]["footprint_rows"] = int(len(samples["footprint"]))
        report["final"]["footprint_pixels"] = (
            int(samples["footprint"]["pixel"].dropna().nunique())
            if "pixel" in samples["footprint"]
            else None
        )
    report["fields"] = _field_report(samples)
    return report


def _field_report(samples: Mapping[str, pd.DataFrame]) -> list[dict[str, Any]]:
    foreground = samples["foreground"]
    background = samples["background"]
    footprint = samples.get("footprint")
    field_values: set[str] = set()
    for frame in (foreground, background, footprint):
        if frame is not None and "field" in frame:
            field_values.update(
                str(value) for value in frame["field"].dropna().unique()
            )

    rows = []
    for field in sorted(field_values):
        fg = _field_slice(foreground, field)
        bg = _field_slice(background, field)
        fp = _field_slice(footprint, field) if footprint is not None else None
        row: dict[str, Any] = {
            "field": field,
            "foreground_rows": int(len(fg)),
            "background_rows": int(len(bg)),
            "total_sample_rows": int(len(fg) + len(bg)),
        }
        if fp is not None:
            row["footprint_rows"] = int(len(fp))
            if "pixel" in fp:
                row["footprint_pixels"] = int(fp["pixel"].dropna().nunique())
        for label, frame in (("foreground", fg), ("background", bg)):
            if "jackknife_region" in frame:
                row[f"{label}_jackknife_regions"] = [
                    int(value)
                    for value in sorted(frame["jackknife_region"].dropna().unique())
                ]
        if (
            "foreground_jackknife_regions" in row
            and "background_jackknife_regions" in row
        ):
            row["shared_jackknife_regions"] = sorted(
                set(row["foreground_jackknife_regions"])
                & set(row["background_jackknife_regions"])
            )
        rows.append(row)
    return rows


def _field_slice(catalog: pd.DataFrame | None, field: str) -> pd.DataFrame:
    if catalog is None or "field" not in catalog:
        return pd.DataFrame()
    return catalog.loc[catalog["field"].astype(str) == field]


def _add_area_metrics(
    report: dict[str, Any],
    samples: Mapping[str, pd.DataFrame],
    nside: int,
) -> None:
    pixel_area = FULL_SKY_DEG2 / (12.0 * nside**2)
    report["nside"] = int(nside)
    report["pixel_area_deg2"] = float(pixel_area)
    footprint = samples.get("footprint")
    if footprint is not None and "pixel" in footprint:
        pixels = int(footprint["pixel"].dropna().nunique())
        report["final"]["footprint_area_deg2"] = pixels * pixel_area

    field_lookup = {row["field"]: row for row in report.get("fields", [])}
    if footprint is not None and {"field", "pixel"}.issubset(footprint.columns):
        for field, group in footprint.groupby("field", dropna=False):
            key = str(field)
            row = field_lookup.get(key)
            if row is None:
                continue
            pixels = int(group["pixel"].dropna().nunique())
            row["footprint_area_deg2"] = pixels * pixel_area
            total_rows = max(1, int(report["final"].get("footprint_rows", 0)))
            total_area = max(
                float(report["final"].get("footprint_area_deg2", 0.0)), 0.0
            )
            row["footprint_row_fraction"] = row.get("footprint_rows", 0) / total_rows
            row["footprint_area_fraction"] = (
                row["footprint_area_deg2"] / total_area if total_area > 0 else np.nan
            )


def format_sample_report(report: Mapping[str, Any]) -> str:
    """Format a sample report as IDE-readable Markdown."""
    sample_id = report.get("sample_id") or "sample"
    lines = [
        f"# Sample Report: {sample_id}",
        "",
    ]
    if report.get("catalog"):
        lines.extend([f"- Catalog: `{report['catalog']}`"])
    if report.get("nside") is not None:
        lines.append(f"- HEALPix nside: `{report['nside']}`")
    final = report.get("final", {})
    lines.extend(
        [
            f"- Foreground rows: `{int(final.get('foreground_rows', 0))}`",
            f"- Background rows: `{int(final.get('background_rows', 0))}`",
        ]
    )
    if final.get("footprint_area_deg2") is not None:
        lines.append(
            f"- Footprint area: `{float(final['footprint_area_deg2']):.6f} deg^2`"
        )
    lines.append("")

    lines.extend(
        _markdown_stage_table("Shared Selection Funnel", report.get("stages", []))
    )
    lines.extend(
        _markdown_stage_table(
            "Foreground Funnel",
            report.get("foreground_stages", []),
        )
    )
    lines.extend(
        _markdown_stage_table(
            "Background Funnel",
            report.get("background_stages", []),
        )
    )
    lines.extend(_markdown_field_table(report.get("fields", [])))
    return "\n".join(lines).rstrip() + "\n"


def _markdown_stage_table(title: str, stages: Any) -> list[str]:
    stages = list(stages or [])
    if not stages:
        return []
    fields = _report_fields(stages)
    lines = [f"## {title}", ""]
    header = ["stage", "rows", *fields]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for stage in stages:
        counts = dict(stage.get("field_counts", {}))
        row = [
            str(stage.get("stage", "")),
            str(int(stage.get("rows", 0))),
            *[str(int(counts.get(field, 0))) for field in fields],
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return lines


def _markdown_field_table(fields: Any) -> list[str]:
    rows = list(fields or [])
    if not rows:
        return []
    columns = [
        "field",
        "footprint_rows",
        "footprint_pixels",
        "footprint_area_deg2",
        "foreground_rows",
        "background_rows",
        "total_sample_rows",
        "shared_jackknife_regions",
    ]
    lines = ["## Field Summary", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for item in rows:
        values = []
        for col in columns:
            value = item.get(col, "")
            if isinstance(value, float):
                value = f"{value:.6f}" if np.isfinite(value) else ""
            elif isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def _report_fields(stages: list[Mapping[str, Any]]) -> list[str]:
    fields = set()
    for stage in stages:
        fields.update(str(field) for field in stage.get("field_counts", {}))
    return sorted(fields)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(val) for val in value]
    if isinstance(value, tuple):
        return [_json_ready(val) for val in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(val) for val in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


class SharedSelectionCuts:
    """Build the mask for cuts shared by foreground and background samples."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        selection: Mapping[str, Any],
    ) -> None:
        self.catalog = catalog
        self.selection = selection
        self.current_mask = np.ones(len(catalog), dtype=bool)

    def mask(self) -> np.ndarray:
        self._apply_shared_query()
        self._apply_pixel_cuts()
        self._apply_photometry_cuts()
        self._apply_photoz_cuts()
        return self.current_mask

    def _apply_shared_query(self) -> None:
        if not self.selection.get("shared_query"):
            return
        query_mask = (
            self.catalog.eval(str(self.selection["shared_query"]))
            .astype(bool)
            .to_numpy()
        )
        self.current_mask &= query_mask

    def _apply_pixel_cuts(self) -> None:
        pixel_depth_cuts = _enabled_option(self.selection, "pixel_depth_cuts")
        if pixel_depth_cuts is not None:
            self.current_mask &= _pixel_depth_cut_mask(
                self.catalog,
                pixel_depth_cuts,
                self.current_mask,
            )

        pixel_occupancy_min = _enabled_option(
            self.selection,
            "pixel_occupancy_min",
        )
        if pixel_occupancy_min is not None:
            self.current_mask &= _pixel_occupancy_mask(
                self.catalog,
                pixel_occupancy_min,
                self.current_mask,
            )

    def _apply_photometry_cuts(self) -> None:
        snr_min = _enabled_option(self.selection, "snr_min")
        if snr_min is not None:
            self.current_mask &= _snr_min_mask(self.catalog, snr_min)

        blendedness_max = _enabled_option(self.selection, "blendedness_max")
        if blendedness_max is not None:
            self.current_mask &= _max_column_mask(
                self.catalog,
                blendedness_max,
                default_column="blendedness_i",
                option_name="blendedness_max",
            )

        magnitude_limits = _enabled_option(self.selection, "magnitude_limits")
        if magnitude_limits is not None:
            self.current_mask &= _magnitude_limit_mask(
                self.catalog,
                magnitude_limits,
            )

    def _apply_photoz_cuts(self) -> None:
        if "photoz_max_sigma" in self.selection:
            self.current_mask &= self.catalog["z_phot_err"].to_numpy(float) < float(
                self.selection["photoz_max_sigma"]
            )

        photoz_estimate_sigma = _enabled_option(
            self.selection,
            "photoz_estimate_max_sigma",
        )
        if photoz_estimate_sigma is not None:
            self.current_mask &= _photoz_estimate_sigma_mask(
                self.catalog,
                photoz_estimate_sigma,
            )

        photoz_diff_norm = _enabled_option(self.selection, "photoz_max_diff_norm")
        if photoz_diff_norm is not None:
            self.current_mask &= _photoz_diff_norm_mask(
                self.catalog,
                photoz_diff_norm,
            )

        if "photoz_max_sigma_norm" in self.selection:
            max_sigma = float(self.selection["photoz_max_sigma_norm"])
            self.current_mask &= self.catalog["z_phot_err"].to_numpy(
                float
            ) < max_sigma * (1.0 + self.catalog["z_phot"].to_numpy(float))


def _redshift_slice(catalog: pd.DataFrame, bounds: list[float]) -> pd.DataFrame:
    if len(bounds) != 2:
        raise ValueError("Redshift bounds must be [min, max]")
    lo, hi = float(bounds[0]), float(bounds[1])
    z_phot = catalog["z_phot"].to_numpy(float)
    return catalog.loc[(z_phot > lo) & (z_phot < hi)].reset_index(drop=True)


def _enabled_option(
    config: Mapping[str, Any],
    name: str,
) -> Any:
    value = config.get(name)
    if value is None or value is False:
        return None
    if isinstance(value, Mapping) and not bool(value.get("enabled", True)):
        return None
    return value


def _photoz_estimate_sigma_mask(catalog: pd.DataFrame, config: Any) -> np.ndarray:
    if isinstance(config, Mapping):
        max_sigma = float(config.get("value", config.get("max", 0.1)))
        columns = [str(col) for col in config.get("columns", [])]
    else:
        max_sigma = float(config)
        columns = []
    if not columns:
        columns = sorted(
            col for col in catalog.columns if col.startswith("photoz_sigma_")
        )
    if not columns:
        raise ValueError(
            "selection.photoz_estimate_max_sigma requires photoz_sigma_* columns"
        )

    mask = np.ones(len(catalog), dtype=bool)
    for col in columns:
        if col not in catalog:
            raise ValueError(f"Photo-z sigma cut requested missing column: {col}")
        mask &= catalog[col].to_numpy(float) < max_sigma
    return mask


def _photoz_diff_norm_mask(catalog: pd.DataFrame, config: Any) -> np.ndarray:
    if isinstance(config, Mapping):
        max_norm = float(config.get("value", config.get("max", 0.1)))
        diff_col = str(config.get("column", config.get("diff_col", "z_phot_diff")))
    else:
        max_norm = float(config)
        diff_col = "z_phot_diff"
    if diff_col not in catalog:
        raise ValueError(f"Photo-z difference cut requested missing column: {diff_col}")
    return catalog[diff_col].to_numpy(float) < max_norm * (
        1.0 + catalog["z_phot"].to_numpy(float)
    )


def _max_column_mask(
    catalog: pd.DataFrame,
    config: Any,
    *,
    default_column: str,
    option_name: str,
) -> np.ndarray:
    if isinstance(config, Mapping):
        column = str(config.get("column", default_column))
        value = float(config.get("value", config.get("max")))
    else:
        column = default_column
        value = float(config)
    if column not in catalog:
        raise ValueError(f"selection.{option_name} requested missing column: {column}")
    return catalog[column].to_numpy(float) < value


def _magnitude_limit_mask(catalog: pd.DataFrame, config: Any) -> np.ndarray:
    configs = config if isinstance(config, list) else [config]
    mask = np.ones(len(catalog), dtype=bool)
    for item in configs:
        if not item or item is False:
            continue
        if not isinstance(item, Mapping):
            raise ValueError("selection.magnitude_limits entries must be mappings")
        if not bool(item.get("enabled", True)):
            continue
        mag = _magnitude_values(catalog, item)
        if "max" in item:
            mask &= mag < float(item["max"])
        if "min" in item:
            mask &= mag > float(item["min"])
    return mask


def _magnitude_values(catalog: pd.DataFrame, config: Mapping[str, Any]) -> np.ndarray:
    if "mag_col" in config:
        mag_col = str(config["mag_col"])
        if mag_col not in catalog:
            raise ValueError(f"Magnitude limit requested missing column: {mag_col}")
        return catalog[mag_col].to_numpy(float)

    if "flux_col" in config:
        flux_col = str(config["flux_col"])
    else:
        band = str(config.get("band", ""))
        if not band:
            raise ValueError(
                "Magnitude limit needs either 'band', 'mag_col', or 'flux_col'"
            )
        if f"mag_{band}" in catalog:
            return catalog[f"mag_{band}"].to_numpy(float)
        if f"cmodel_flux_{band}" in catalog:
            flux_col = f"cmodel_flux_{band}"
        else:
            flux_col = f"flux_{band}"

    if flux_col not in catalog:
        raise ValueError(f"Magnitude limit requested missing flux column: {flux_col}")
    zero_point = float(config.get("zero_point", MAGSYS_ZEROPOINT))
    flux = catalog[flux_col].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return zero_point - 2.5 * np.log10(flux)


def _snr_min_mask(catalog: pd.DataFrame, config: Any) -> np.ndarray:
    if not isinstance(config, Mapping):
        raise ValueError("selection.snr_min must be a mapping")
    flux_template = str(config.get("flux_template", "flux_{band}"))
    fluxerr_template = str(config.get("fluxerr_template", "fluxerr_{band}"))
    bands = config.get("bands")
    if not isinstance(bands, Mapping):
        raise ValueError("selection.snr_min.bands must map band names to thresholds")

    mask = np.ones(len(catalog), dtype=bool)
    for band, threshold in bands.items():
        band = str(band)
        flux_col = flux_template.format(band=band)
        fluxerr_col = fluxerr_template.format(band=band)
        missing = [col for col in (flux_col, fluxerr_col) if col not in catalog]
        if missing:
            raise ValueError(f"SNR cut requested missing columns: {missing}")
        flux = catalog[flux_col].to_numpy(float)
        fluxerr = catalog[fluxerr_col].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            snr = flux / fluxerr
        mask &= snr > float(threshold)
    return mask


def _pixel_occupancy_mask(
    catalog: pd.DataFrame,
    config: Any,
    current_mask: np.ndarray,
) -> np.ndarray:
    if isinstance(config, Mapping):
        pixel_col = str(config.get("pixel_col", "pixel"))
        minimum = int(config.get("value", config.get("min", config.get("minimum", 1))))
    else:
        pixel_col = "pixel"
        minimum = int(config)
    if minimum < 1:
        raise ValueError("selection.pixel_occupancy_min must be positive")
    if pixel_col not in catalog:
        raise ValueError(f"Pixel occupancy cut requested missing column: {pixel_col}")

    pixels = catalog[pixel_col]
    counts = pixels.loc[current_mask].value_counts()
    accepted = set(counts[counts >= minimum].index.tolist())
    return pixels.isin(accepted).to_numpy()


class PixelDepthCuts:
    """Apply depth cuts that are measured per HEALPix footprint pixel."""

    def __init__(
        self,
        catalog: pd.DataFrame,
        config: Mapping[str, Any],
        current_mask: np.ndarray,
    ) -> None:
        self.catalog = catalog
        self.config = config
        self.current_mask = current_mask
        self.pixel_col = str(config.get("pixel_col", "pixel"))
        if self.pixel_col not in catalog:
            raise ValueError(
                f"Pixel depth cuts requested missing column: {self.pixel_col}"
            )
        self.fluxerr_template = str(config.get("fluxerr_template", "fluxerr_{band}"))
        self.depth_sigma = float(config.get("depth_sigma", 10.0))
        if self.depth_sigma <= 0:
            raise ValueError("selection.pixel_depth_cuts.depth_sigma must be positive")
        aggregate = str(config.get("aggregate", "median"))
        if aggregate != "median":
            raise ValueError("selection.pixel_depth_cuts.aggregate must be 'median'")
        self._depth_cache: dict[str, np.ndarray] = {}

    def mask(self) -> np.ndarray:
        mask = self.current_mask.copy()
        mask = self._apply_valid_range(mask)
        mask = self._apply_min_occupancy(mask)
        mask = self._apply_completeness(mask)
        mask = self._drop_shallowest_pixels(mask)
        return mask

    def depth_for(self, band: str) -> np.ndarray:
        if band not in self._depth_cache:
            self._depth_cache[band] = _pixel_depth_values(
                self.catalog,
                pixel_col=self.pixel_col,
                fluxerr_col=self.fluxerr_template.format(band=band),
                depth_sigma=self.depth_sigma,
                rows_mask=self.current_mask,
            )
        return self._depth_cache[band]

    def _apply_valid_range(self, mask: np.ndarray) -> np.ndarray:
        valid_range = self.config.get("valid_range")
        if not valid_range:
            return mask
        if not isinstance(valid_range, Mapping):
            raise ValueError("selection.pixel_depth_cuts.valid_range must be a mapping")
        bands = [str(band) for band in valid_range.get("bands", [])]
        if not bands:
            raise ValueError("selection.pixel_depth_cuts.valid_range.bands is required")
        lo = valid_range.get("min")
        hi = valid_range.get("max")
        for band in bands:
            depth = self.depth_for(band)
            if lo is not None:
                mask &= depth > float(lo)
            if hi is not None:
                mask &= depth < float(hi)
        return mask

    def _apply_min_occupancy(self, mask: np.ndarray) -> np.ndarray:
        occupancy = self.config.get("min_occupancy")
        if occupancy is None:
            return mask
        return mask & _pixel_occupancy_mask(
            self.catalog,
            {"pixel_col": self.pixel_col, "value": occupancy},
            mask,
        )

    def _apply_completeness(self, mask: np.ndarray) -> np.ndarray:
        complete_to = self.config.get("complete_to")
        if not complete_to:
            return mask
        complete_configs = (
            complete_to if isinstance(complete_to, list) else [complete_to]
        )
        for item in complete_configs:
            if not isinstance(item, Mapping):
                raise ValueError(
                    "pixel_depth_cuts.complete_to entries must be mappings"
                )
            band = str(item.get("band", ""))
            if not band:
                raise ValueError("pixel_depth_cuts.complete_to.band is required")
            if "magnitude" not in item:
                raise ValueError("pixel_depth_cuts.complete_to.magnitude is required")
            mask &= self.depth_for(band) > float(item["magnitude"])
        return mask

    def _drop_shallowest_pixels(self, mask: np.ndarray) -> np.ndarray:
        drop_shallowest = self.config.get("drop_shallowest")
        if not drop_shallowest:
            return mask
        if not isinstance(drop_shallowest, Mapping):
            raise ValueError(
                "selection.pixel_depth_cuts.drop_shallowest must be a mapping"
            )
        bands = [str(band) for band in drop_shallowest.get("bands", [])]
        if not bands:
            raise ValueError(
                "selection.pixel_depth_cuts.drop_shallowest.bands is required"
            )
        q = float(drop_shallowest.get("fraction", 0.05))
        if not 0.0 <= q < 1.0:
            raise ValueError(
                "selection.pixel_depth_cuts.drop_shallowest.fraction must be in [0, 1)"
            )
        unique = bool(drop_shallowest.get("unique_pixels", True))
        base_mask = mask.copy()
        drop_mask = np.zeros(len(self.catalog), dtype=bool)
        for band in bands:
            depth = self.depth_for(band)
            finite = base_mask & np.isfinite(depth)
            values = depth[finite]
            if unique:
                values = np.unique(values)
            if len(values) == 0:
                return np.zeros(len(self.catalog), dtype=bool)
            threshold = float(np.nanquantile(values, q))
            drop_mask |= base_mask & (~np.isfinite(depth) | (depth <= threshold))
        return mask & ~drop_mask


def _pixel_depth_cut_mask(
    catalog: pd.DataFrame,
    config: Any,
    current_mask: np.ndarray,
) -> np.ndarray:
    if not isinstance(config, Mapping):
        raise ValueError("selection.pixel_depth_cuts must be a mapping")
    return PixelDepthCuts(catalog, config, current_mask).mask()


def _pixel_depth_values(
    catalog: pd.DataFrame,
    *,
    pixel_col: str,
    fluxerr_col: str,
    depth_sigma: float,
    rows_mask: np.ndarray,
) -> np.ndarray:
    if fluxerr_col not in catalog:
        raise ValueError(f"Pixel depth cut requested missing column: {fluxerr_col}")
    fluxerr = catalog[fluxerr_col].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = MAGSYS_ZEROPOINT - 2.5 * np.log10(depth_sigma * fluxerr)
    finite = rows_mask & np.isfinite(depth) & (fluxerr > 0)
    return _median_pixel_depth(catalog, pixel_col=pixel_col, depth=depth, finite=finite)


def _median_pixel_depth(
    catalog: pd.DataFrame,
    *,
    pixel_col: str,
    depth: np.ndarray,
    finite: np.ndarray,
) -> np.ndarray:
    data = pd.DataFrame(
        {
            "pixel": catalog[pixel_col],
            "depth": depth,
        }
    )
    grouped = data.loc[finite].groupby("pixel")["depth"].median()
    return catalog[pixel_col].map(grouped).to_numpy(float)


def _sample_footprint(
    catalog: pd.DataFrame,
    config: Any,
) -> pd.DataFrame | None:
    if not config:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("Sample footprint config must be a mapping")
    if not bool(config.get("enabled", True)):
        return None
    columns = [str(col) for col in config.get("columns", [])]
    if not columns:
        columns = ["ra", "dec", "field", "pixel", "jackknife_region"]
    columns = [col for col in columns if col in catalog]
    if "pixel" not in columns:
        raise ValueError("Sample footprint output requires a 'pixel' column")
    return catalog[columns].copy().reset_index(drop=True)


def _assign_sample_jackknife(catalog: pd.DataFrame, config: Any) -> pd.DataFrame:
    if not isinstance(config, Mapping):
        raise ValueError("Sample jackknife config must be a mapping")
    regions_per_field = config.get("regions_per_field")
    if regions_per_field is None:
        return catalog
    return assign_jackknife_regions(
        catalog,
        regions_per_field=int(regions_per_field),
        field_col=str(config.get("field_col", "field")),
        output_col=str(config.get("output_col", "jackknife_region")),
    )


def _cleaning_for_sample(
    cleaning: Mapping[str, Any],
    sample: str,
) -> Mapping[str, Any]:
    if sample not in {"foreground", "background"}:
        raise ValueError(f"Unknown sample for cleaning config: {sample}")
    if sample not in cleaning:
        return cleaning

    sample_config = cleaning.get(sample)
    if sample_config is None or sample_config is False:
        return {"enabled": False}
    if not isinstance(sample_config, Mapping):
        raise ValueError(f"cleaning.{sample} must be a mapping or false")

    base = {
        key: value
        for key, value in cleaning.items()
        if key not in {"foreground", "background"}
    }
    merged = dict(base)
    merged.update(sample_config)
    return merged
