"""Utilities for DP1 close-pair postage-stamp blend review."""

from __future__ import annotations

import html
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from astropy.cosmology import Planck18 as DEFAULT_COSMO
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


ARCSEC_PER_RADIAN = 206264.80624709636
DEFAULT_REVIEW_BANDS = ("r", "i")
DEFAULT_INNER_R_MIN_KPC = 10.0
DEFAULT_INNER_R_MAX_KPC = 15.0
DEFAULT_EXPECTED_INNER_PAIR_COUNT = 111

PAIR_TABLE_COLUMNS = [
    "pair_id",
    "foreground_index",
    "background_index",
    "foreground_object_id",
    "background_object_id",
    "foreground_ra",
    "foreground_dec",
    "foreground_z",
    "background_ra",
    "background_dec",
    "background_z",
    "foreground_field",
    "background_field",
    "midpoint_ra",
    "midpoint_dec",
    "theta_arcsec",
    "r_perp_kpc",
]

BASE_OBJECT_COLUMNS = [
    "objectId",
    "coord_ra",
    "coord_dec",
    "detect_fromBlend",
    "detect_isIsolated",
    "parentObjectId",
    "deblend_failed",
]


def build_projected_pair_table(
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    *,
    r_min_kpc: float = DEFAULT_INNER_R_MIN_KPC,
    r_max_kpc: float = DEFAULT_INNER_R_MAX_KPC,
    cosmo: Any = DEFAULT_COSMO,
    analysis_id: str = "dp1_default",
    bin_label: str = "inner",
) -> pd.DataFrame:
    """Return foreground-background pairs in a projected physical annulus.

    This mirrors the TreeCorr ``Rlens`` convention used by the stacker: angular
    separation is multiplied by each foreground galaxy's angular-diameter
    distance, so the same physical annulus maps to different sky angles for
    different foreground redshifts.
    """
    if r_min_kpc <= 0 or r_max_kpc <= r_min_kpc:
        raise ValueError("Projected radial limits must satisfy max > min > 0")
    _require_columns(foreground, {"ra", "dec", "z_phot"}, "foreground")
    _require_columns(background, {"ra", "dec"}, "background")

    if foreground.empty or background.empty:
        return pd.DataFrame(columns=PAIR_TABLE_COLUMNS)

    fg = foreground.reset_index(drop=True)
    bg = background.reset_index(drop=True)
    fg_vectors = _unit_vectors(fg["ra"].to_numpy(float), fg["dec"].to_numpy(float))
    bg_vectors = _unit_vectors(bg["ra"].to_numpy(float), bg["dec"].to_numpy(float))
    bg_tree = cKDTree(bg_vectors)
    fg_da = cosmo.angular_diameter_distance(fg["z_phot"].to_numpy(float)).to_value(
        "kpc"
    )

    rows: list[dict[str, Any]] = []
    for fg_index, (fg_vector, da_kpc) in enumerate(zip(fg_vectors, fg_da)):
        if not np.isfinite(da_kpc) or da_kpc <= 0:
            continue
        max_theta = r_max_kpc / da_kpc
        max_chord = 2.0 * np.sin(0.5 * max_theta)
        neighbors = bg_tree.query_ball_point(fg_vector, max_chord)
        if not neighbors:
            continue

        bg_index = np.asarray(sorted(neighbors), dtype=int)
        chords = np.linalg.norm(bg_vectors[bg_index] - fg_vector, axis=1)
        theta = 2.0 * np.arcsin(np.clip(0.5 * chords, 0.0, 1.0))
        r_perp = theta * da_kpc
        use = (r_perp >= r_min_kpc) & (r_perp < r_max_kpc)
        if not np.any(use):
            continue

        for local_index in np.flatnonzero(use):
            j = int(bg_index[local_index])
            midpoint_ra, midpoint_dec = _spherical_midpoint(
                fg_vectors[fg_index],
                bg_vectors[j],
            )
            rows.append(
                {
                    "foreground_index": fg_index,
                    "background_index": j,
                    "foreground_object_id": _value_or_none(fg, fg_index, "object_id"),
                    "background_object_id": _value_or_none(bg, j, "object_id"),
                    "foreground_ra": float(fg.loc[fg_index, "ra"]),
                    "foreground_dec": float(fg.loc[fg_index, "dec"]),
                    "foreground_z": float(fg.loc[fg_index, "z_phot"]),
                    "background_ra": float(bg.loc[j, "ra"]),
                    "background_dec": float(bg.loc[j, "dec"]),
                    "background_z": _value_or_nan(bg, j, "z_phot"),
                    "foreground_field": _value_or_none(fg, fg_index, "field"),
                    "background_field": _value_or_none(bg, j, "field"),
                    "midpoint_ra": midpoint_ra,
                    "midpoint_dec": midpoint_dec,
                    "theta_arcsec": float(theta[local_index] * ARCSEC_PER_RADIAN),
                    "r_perp_kpc": float(r_perp[local_index]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=PAIR_TABLE_COLUMNS)

    pair_table = pd.DataFrame(rows)
    pair_table = pair_table.sort_values(
        ["foreground_object_id", "background_object_id", "theta_arcsec"],
        kind="mergesort",
    ).reset_index(drop=True)
    width = max(4, len(str(len(pair_table))))
    pair_table.insert(
        0,
        "pair_id",
        [
            f"{analysis_id}_{bin_label}_{index + 1:0{width}d}"
            for index in range(len(pair_table))
        ],
    )
    return pair_table[PAIR_TABLE_COLUMNS]


def pair_table_summary(
    pair_table: pd.DataFrame,
    *,
    quantiles: Sequence[float] = (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0),
) -> dict[str, Any]:
    """Return compact count and quantile diagnostics for a pair table."""
    summary: dict[str, Any] = {"count": int(len(pair_table))}
    for column in ("theta_arcsec", "r_perp_kpc"):
        values = np.asarray(pair_table.get(column, []), dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            summary[column] = {}
            continue
        summary[column] = {
            _quantile_label(q): float(np.quantile(values, q)) for q in quantiles
        }
    return summary


def requested_blend_columns(
    bands: Sequence[str] = DEFAULT_REVIEW_BANDS,
) -> list[str]:
    """Return DP1 Object-table columns useful for blend triage."""
    columns = list(BASE_OBJECT_COLUMNS)
    for band in bands:
        columns.extend(
            [
                f"{band}_blendedness",
                f"{band}_blendedness_flag",
                f"{band}_deblend_blendedness",
                f"{band}_deblend_fluxOverlap",
                f"{band}_deblend_fluxOverlapFraction",
            ]
        )
    return _unique(columns)


def available_tap_columns(service: Any, table_name: str = "dp1.Object") -> list[str]:
    """Query TAP schema metadata for available columns in a DP1 table."""
    query = (
        "SELECT column_name FROM tap_schema.columns "
        f"WHERE table_name = '{_escape_adql_literal(table_name)}'"
    )
    result = service.search(query).to_table()
    return [str(value) for value in result["column_name"]]


def select_existing_columns(
    requested: Sequence[str],
    available: Iterable[str],
    *,
    required: Sequence[str] = ("objectId",),
) -> list[str]:
    """Filter requested column names to columns present in a TAP table."""
    available_set = set(available)
    missing_required = [column for column in required if column not in available_set]
    if missing_required:
        raise ValueError(f"Required TAP columns are missing: {missing_required}")
    return _unique(
        [column for column in [*required, *requested] if column in available_set]
    )


def build_object_metadata_query(
    object_ids: Iterable[int],
    columns: Sequence[str],
    *,
    table_name: str = "dp1.Object",
) -> str:
    """Build a small ADQL query for selected Object-table rows."""
    safe_columns = [_safe_adql_identifier(column) for column in _unique(columns)]
    if "objectId" not in safe_columns:
        safe_columns.insert(0, "objectId")
    safe_table = _safe_adql_table_name(table_name)
    ids = sorted({int(object_id) for object_id in object_ids if pd.notna(object_id)})
    if not ids:
        raise ValueError("At least one object_id is required")
    id_list = ", ".join(str(object_id) for object_id in ids)
    return (
        f"SELECT {', '.join(safe_columns)} "
        f"FROM {safe_table} "
        f"WHERE objectId IN ({id_list})"
    )


def fetch_object_metadata(
    service: Any,
    object_ids: Iterable[int],
    columns: Sequence[str],
    *,
    table_name: str = "dp1.Object",
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """Fetch selected Object-table metadata with TAP, chunking long ID lists."""
    ids = sorted({int(object_id) for object_id in object_ids if pd.notna(object_id)})
    if not ids:
        return pd.DataFrame(columns=_unique(["objectId", *columns]))
    chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]
    tables = []
    for chunk in chunks:
        query = build_object_metadata_query(chunk, columns, table_name=table_name)
        tables.append(service.search(query).to_table().to_pandas())
    if not tables:
        return pd.DataFrame(columns=_unique(["objectId", *columns]))
    return pd.concat(tables, ignore_index=True)


def score_pair_blend_risk(
    pair_table: pd.DataFrame,
    object_metadata: pd.DataFrame,
    *,
    bands: Sequence[str] = DEFAULT_REVIEW_BANDS,
    blendedness_threshold: float = 0.1,
    overlap_fraction_threshold: float = 0.1,
) -> pd.DataFrame:
    """Attach blend metadata and assign a simple triage risk level."""
    _require_columns(
        pair_table,
        {"pair_id", "foreground_object_id", "background_object_id", "theta_arcsec"},
        "pair_table",
    )
    if object_metadata.empty:
        review = pair_table.copy()
        review["blend_risk"] = "unknown"
        review["blend_risk_score"] = np.nan
        return review

    meta = object_metadata.copy()
    if "object_id" in meta.columns and "objectId" not in meta.columns:
        meta = meta.rename(columns={"object_id": "objectId"})
    _require_columns(meta, {"objectId"}, "object_metadata")

    review = pair_table.copy()
    review = _merge_prefixed_metadata(
        review,
        meta,
        left_on="foreground_object_id",
        prefix="foreground",
    )
    review = _merge_prefixed_metadata(
        review,
        meta,
        left_on="background_object_id",
        prefix="background",
    )

    fg_parent = _numeric_series(review, "foreground_parentObjectId")
    bg_parent = _numeric_series(review, "background_parentObjectId")
    same_parent = (
        np.isfinite(fg_parent)
        & np.isfinite(bg_parent)
        & (fg_parent > 0)
        & (fg_parent == bg_parent)
    )

    not_isolated = np.zeros(len(review), dtype=bool)
    from_blend = np.zeros(len(review), dtype=bool)
    if "foreground_detect_isIsolated" in review:
        not_isolated |= ~_bool_series(review, "foreground_detect_isIsolated")
    if "background_detect_isIsolated" in review:
        not_isolated |= ~_bool_series(review, "background_detect_isIsolated")
    if "foreground_detect_fromBlend" in review:
        from_blend |= _bool_series(review, "foreground_detect_fromBlend")
    if "background_detect_fromBlend" in review:
        from_blend |= _bool_series(review, "background_detect_fromBlend")

    failed_cols = _matching_metadata_columns(review, ["deblend_failed"])
    deblend_failed = _row_any_bool(review, failed_cols)

    overlap_cols = _matching_metadata_columns(review, ["fluxOverlapFraction"])
    blendedness_cols = _blend_value_columns(review, bands)
    max_overlap = _row_max_numeric(review, overlap_cols)
    max_blendedness = _row_max_numeric(review, blendedness_cols)

    high = (
        same_parent
        | deblend_failed
        | (np.nan_to_num(max_overlap, nan=0.0) >= overlap_fraction_threshold)
    )
    medium = (
        not_isolated
        | from_blend
        | (np.nan_to_num(max_blendedness, nan=0.0) >= blendedness_threshold)
    )
    risk = np.full(len(review), "low", dtype=object)
    risk[medium] = "medium"
    risk[high] = "high"

    score = (
        same_parent.astype(float) * 100.0
        + deblend_failed.astype(float) * 50.0
        + not_isolated.astype(float) * 20.0
        + from_blend.astype(float) * 10.0
        + np.nan_to_num(max_overlap, nan=0.0) * 10.0
        + np.nan_to_num(max_blendedness, nan=0.0)
    )

    review["same_parent"] = same_parent
    review["either_not_isolated"] = not_isolated
    review["either_from_blend"] = from_blend
    review["either_deblend_failed"] = deblend_failed
    review["max_overlap_fraction"] = max_overlap
    review["max_blendedness"] = max_blendedness
    review["blend_risk"] = risk
    review["blend_risk_score"] = score
    return sort_review_table(review)


def sort_review_table(review_table: pd.DataFrame) -> pd.DataFrame:
    """Sort by the review priority requested for the contact sheet."""
    table = review_table.copy()
    for column, default in (
        ("same_parent", False),
        ("theta_arcsec", np.inf),
        ("max_blendedness", -np.inf),
        ("pair_id", ""),
    ):
        if column not in table:
            table[column] = default
    return table.sort_values(
        ["same_parent", "theta_arcsec", "max_blendedness", "pair_id"],
        ascending=[False, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def write_review_labels_template(pair_table: pd.DataFrame, path: str | Path) -> Path:
    """Write a blank manual-review CSV next to the generated stamps."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = pair_table[["pair_id"]].copy()
    labels["blend_review"] = ""
    labels["notes"] = ""
    labels.to_csv(path, index=False)
    return path


def write_summary_json(summary: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON summary in a stable, human-readable form."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def write_contact_sheet_html(
    review_table: pd.DataFrame,
    output_path: str | Path,
    *,
    image_dir: str | Path,
    image_template: str = "{pair_id}_stamps.png",
    title: str = "DP1 Inner-Bin Pair Blend Review",
) -> Path:
    """Write a simple HTML contact sheet for stamp PNGs and blend metadata."""
    output_path = Path(output_path)
    image_dir = Path(image_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in sort_review_table(review_table).iterrows():
        pair_id = str(row["pair_id"])
        image_name = image_template.format(pair_id=pair_id)
        image_path = image_dir / image_name
        image_rel = _relative_href(image_path, output_path.parent)
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(pair_id)}</strong><br>"
            f"risk: {html.escape(str(row.get('blend_risk', 'unknown')))}<br>"
            f"theta: {_format_number(row.get('theta_arcsec'))} arcsec<br>"
            f"r_perp: {_format_number(row.get('r_perp_kpc'))} kpc<br>"
            f"same parent: {html.escape(str(row.get('same_parent', '')))}</td>"
            f'<td><img src="{html.escape(image_rel)}" alt="{html.escape(pair_id)}"></td>'
            "<td>"
            f"fg: {html.escape(str(row.get('foreground_object_id', '')))}<br>"
            f"bg: {html.escape(str(row.get('background_object_id', '')))}<br>"
            f"max blendedness: {_format_number(row.get('max_blendedness'))}<br>"
            f"max overlap fraction: {_format_number(row.get('max_overlap_fraction'))}"
            "</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 1.5rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    td {{ border-top: 1px solid #ddd; padding: 0.75rem; vertical-align: top; }}
    img {{ max-width: 760px; width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p>Allowed manual labels in review_labels.csv: ok, maybe, reject.</p>
  <table>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(document)
    return output_path


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cos_dec = np.cos(dec)
    return np.column_stack([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)])


def _spherical_midpoint(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    midpoint = first + second
    norm = np.linalg.norm(midpoint)
    if not np.isfinite(norm) or norm == 0:
        midpoint = first
        norm = np.linalg.norm(midpoint)
    midpoint = midpoint / norm
    ra = float(np.rad2deg(np.arctan2(midpoint[1], midpoint[0])) % 360.0)
    dec = float(np.rad2deg(np.arcsin(np.clip(midpoint[2], -1.0, 1.0))))
    return ra, dec


def _value_or_none(frame: pd.DataFrame, index: int, column: str) -> Any:
    if column not in frame:
        return None
    value = frame.loc[index, column]
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _value_or_nan(frame: pd.DataFrame, index: int, column: str) -> float:
    if column not in frame:
        return float("nan")
    value = frame.loc[index, column]
    return float(value) if pd.notna(value) else float("nan")


def _quantile_label(value: float) -> str:
    if value == 0.0:
        return "min"
    if value == 1.0:
        return "max"
    return f"q{int(round(value * 100)):02d}"


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _escape_adql_literal(value: str) -> str:
    return value.replace("'", "''")


def _safe_adql_identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Unsafe ADQL identifier: {value!r}")
    return value


def _safe_adql_table_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", value):
        raise ValueError(f"Unsafe ADQL table name: {value!r}")
    return value


def _merge_prefixed_metadata(
    pair_table: pd.DataFrame,
    object_metadata: pd.DataFrame,
    *,
    left_on: str,
    prefix: str,
) -> pd.DataFrame:
    prefixed = object_metadata.add_prefix(f"{prefix}_")
    return pair_table.merge(
        prefixed,
        how="left",
        left_on=left_on,
        right_on=f"{prefix}_objectId",
    )


def _bool_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.zeros(len(frame), dtype=bool)
    return frame[column].fillna(False).astype(bool).to_numpy()


def _numeric_series(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float)


def _matching_metadata_columns(
    frame: pd.DataFrame, name_fragments: Sequence[str]
) -> list[str]:
    lowered = [(column, column.lower()) for column in frame.columns]
    return [
        column
        for column, lower in lowered
        if column.startswith(("foreground_", "background_"))
        and all(fragment.lower() in lower for fragment in name_fragments)
    ]


def _blend_value_columns(frame: pd.DataFrame, bands: Sequence[str]) -> list[str]:
    columns: list[str] = []
    for prefix in ("foreground", "background"):
        for band in bands:
            for suffix in ("blendedness", "deblend_blendedness"):
                column = f"{prefix}_{band}_{suffix}"
                if column in frame:
                    columns.append(column)
    return columns


def _row_any_bool(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.zeros(len(frame), dtype=bool)
    values = frame.loc[:, list(columns)].fillna(False).astype(bool).to_numpy()
    return np.any(values, axis=1)


def _row_max_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if not columns:
        return np.full(len(frame), np.nan, dtype=float)
    values = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return values.max(axis=1, skipna=True).to_numpy(float)


def _relative_href(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _format_number(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(numeric):
        return ""
    return f"{numeric:.4g}"
