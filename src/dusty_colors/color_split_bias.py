"""Color-contrast split diagnostics for close-pair stacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .observables import (
    flux_ratio_observable,
    magnitude_color_observable,
    parse_color,
)
from .postage_stamps import build_projected_pair_table, pair_table_summary


FG_REDDER_GROUP = "fg_redder_than_bg"
BG_REDDER_GROUP = "bg_redder_than_fg"
INVALID_GROUP = "invalid_or_tie"
GROUPS = (FG_REDDER_GROUP, BG_REDDER_GROUP)
HIGH_GROUP = "high"
LOW_GROUP = "low"
PROXY_GROUPS = (HIGH_GROUP, LOW_GROUP)


def radial_bin_label(bin_index: int, r_min_kpc: float, r_max_kpc: float) -> str:
    """Return a compact label for a 0-indexed radial bin."""
    return f"bin{bin_index + 1}_{r_min_kpc:g}_{r_max_kpc:g}kpc"


def build_radial_pair_table(
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    *,
    stack_id: str,
    bin_index: int,
    r_edges_kpc: Sequence[float],
) -> pd.DataFrame:
    """Build foreground-background pair table for one radial bin."""
    r_edges = np.asarray(r_edges_kpc, dtype=float)
    if bin_index < 0 or bin_index >= len(r_edges) - 1:
        raise ValueError("bin_index must select one interval in r_edges_kpc")
    r_min = float(r_edges[bin_index])
    r_max = float(r_edges[bin_index + 1])
    return build_projected_pair_table(
        foreground[["object_id", "ra", "dec", "z_phot", "field"]],
        background[["object_id", "ra", "dec", "z_phot", "field"]],
        r_min_kpc=r_min,
        r_max_kpc=r_max,
        analysis_id=stack_id,
        bin_label=f"bin{bin_index + 1}",
    )


def attach_color_split_columns(
    pairs: pd.DataFrame,
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    *,
    color: str = "g-z",
    snr_max: float | None = 100.0,
) -> pd.DataFrame:
    """Attach foreground/background color and split labels to a pair table."""
    if pairs.empty:
        out = pairs.copy()
        out["color_group"] = pd.Series(dtype=object)
        return out

    fg_rows = foreground.iloc[pairs["foreground_index"].to_numpy(int)].reset_index(
        drop=True
    )
    bg_rows = background.iloc[pairs["background_index"].to_numpy(int)].reset_index(
        drop=True
    )

    fg_color, fg_color_err = magnitude_color_observable(fg_rows, color, snr_max=snr_max)
    bg_color, bg_color_err = magnitude_color_observable(bg_rows, color, snr_max=snr_max)
    bg_ratio, bg_ratio_err = flux_ratio_observable(bg_rows, color, snr_max=snr_max)

    label = color.replace("-", "_")
    out = pairs.copy()
    out[f"foreground_{label}"] = fg_color
    out[f"foreground_{label}_err"] = fg_color_err
    out[f"background_{label}"] = bg_color
    out[f"background_{label}_err"] = bg_color_err
    out[f"foreground_minus_background_{label}"] = (
        out[f"foreground_{label}"] - out[f"background_{label}"]
    )
    out[f"background_flux_ratio_{label}"] = bg_ratio
    out[f"background_flux_ratio_{label}_err"] = bg_ratio_err

    valid_color = (
        np.isfinite(out[f"foreground_{label}"])
        & np.isfinite(out[f"background_{label}"])
        & np.isfinite(out[f"background_flux_ratio_{label}"])
        & np.isfinite(out[f"background_flux_ratio_{label}_err"])
        & (out[f"background_flux_ratio_{label}"] > 0)
        & (out[f"background_flux_ratio_{label}_err"] > 0)
    )
    contrast = out[f"foreground_minus_background_{label}"]
    out["valid_color_split"] = valid_color
    out["color_group"] = INVALID_GROUP
    out.loc[valid_color & (contrast > 0), "color_group"] = FG_REDDER_GROUP
    out.loc[valid_color & (contrast < 0), "color_group"] = BG_REDDER_GROUP
    return out


def attach_pair_observables(
    pairs: pd.DataFrame,
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    *,
    color: str = "g-z",
    snr_max: float | None = 100.0,
) -> pd.DataFrame:
    """Attach pair-level colors, flux ratios, and brightness metrics."""
    if pairs.empty:
        return pairs.copy()

    fg_rows = foreground.iloc[pairs["foreground_index"].to_numpy(int)].reset_index(
        drop=True
    )
    bg_rows = background.iloc[pairs["background_index"].to_numpy(int)].reset_index(
        drop=True
    )
    label = color.replace("-", "_")
    fg_color, fg_color_err = magnitude_color_observable(fg_rows, color, snr_max=snr_max)
    bg_color, bg_color_err = magnitude_color_observable(bg_rows, color, snr_max=snr_max)
    fg_ratio, fg_ratio_err = flux_ratio_observable(fg_rows, color, snr_max=snr_max)
    bg_ratio, bg_ratio_err = flux_ratio_observable(bg_rows, color, snr_max=snr_max)

    out = pairs.copy()
    out[f"foreground_{label}"] = fg_color
    out[f"foreground_{label}_err"] = fg_color_err
    out[f"background_{label}"] = bg_color
    out[f"background_{label}_err"] = bg_color_err
    out[f"foreground_minus_background_{label}"] = (
        out[f"foreground_{label}"] - out[f"background_{label}"]
    )
    out[f"foreground_flux_ratio_{label}"] = fg_ratio
    out[f"foreground_flux_ratio_{label}_err"] = fg_ratio_err
    out[f"background_flux_ratio_{label}"] = bg_ratio
    out[f"background_flux_ratio_{label}_err"] = bg_ratio_err
    band1, band2 = parse_color(color)
    for band in (band1, band2):
        flux_col = f"flux_{band}"
        if flux_col in fg_rows:
            out[f"foreground_{band}_flux"] = fg_rows[flux_col].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"log10_foreground_{band}_flux"] = np.log10(
                    out[f"foreground_{band}_flux"]
                )
        if flux_col in bg_rows:
            out[f"background_{band}_flux"] = bg_rows[flux_col].to_numpy(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[f"log10_background_{band}_flux"] = np.log10(
                    out[f"background_{band}_flux"]
                )

    if "flux_z" in fg_rows and "flux_z" in bg_rows:
        out["foreground_z_flux"] = fg_rows["flux_z"].to_numpy(float)
        out["background_z_flux"] = bg_rows["flux_z"].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["foreground_background_z_flux_ratio"] = (
                out["foreground_z_flux"] / out["background_z_flux"]
            )
            out["log10_foreground_background_z_flux_ratio"] = np.log10(
                out["foreground_background_z_flux_ratio"]
            )
    out["valid_background_stack_color"] = (
        np.isfinite(out[f"background_flux_ratio_{label}"])
        & np.isfinite(out[f"background_flux_ratio_{label}_err"])
        & (out[f"background_flux_ratio_{label}"] > 0)
        & (out[f"background_flux_ratio_{label}_err"] > 0)
    )
    out["valid_flipped_corrected_stack_color"] = (
        out["valid_background_stack_color"]
        & np.isfinite(out[f"foreground_flux_ratio_{label}"])
        & np.isfinite(out[f"foreground_flux_ratio_{label}_err"])
        & (out[f"foreground_flux_ratio_{label}"] > 0)
        & (out[f"foreground_flux_ratio_{label}_err"] > 0)
    )
    return out


def attach_object_metadata(
    pairs: pd.DataFrame,
    object_metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach foreground/background Object metadata to a pair table."""
    out = pairs.copy()
    if object_metadata is None or object_metadata.empty:
        return out
    metadata = object_metadata.copy()
    if "object_id" in metadata.columns and "objectId" not in metadata.columns:
        metadata = metadata.rename(columns={"object_id": "objectId"})
    if "objectId" not in metadata.columns:
        return out
    foreground_meta = metadata.add_prefix("foreground_")
    background_meta = metadata.add_prefix("background_")
    out = out.merge(
        foreground_meta,
        how="left",
        left_on="foreground_object_id",
        right_on="foreground_objectId",
    )
    out = out.merge(
        background_meta,
        how="left",
        left_on="background_object_id",
        right_on="background_objectId",
    )
    return out


def attach_overlap_proxy(
    pairs: pd.DataFrame,
    *,
    bands: Sequence[str] = ("r", "i"),
) -> pd.DataFrame:
    """Attach max deblend flux-overlap fraction from pair Object metadata."""
    out = pairs.copy()
    columns = []
    for prefix in ("foreground", "background"):
        for band in bands:
            column = f"{prefix}_{band}_deblend_fluxOverlapFraction"
            if column in out.columns:
                columns.append(column)
    if columns:
        values = out.loc[:, columns].apply(pd.to_numeric, errors="coerce")
        out["max_deblend_fluxOverlapFraction"] = values.max(axis=1, skipna=True)
    else:
        out["max_deblend_fluxOverlapFraction"] = np.nan
    return out


def attach_independent_proxy_columns(
    pairs: pd.DataFrame,
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    *,
    color: str = "g-z",
    snr_max: float | None = 100.0,
    object_metadata: pd.DataFrame | None = None,
    overlap_bands: Sequence[str] = ("r", "i"),
) -> pd.DataFrame:
    """Attach inputs needed for contamination-proxy split diagnostics."""
    out = attach_pair_observables(
        pairs,
        foreground,
        background,
        color=color,
        snr_max=snr_max,
    )
    out = attach_object_metadata(out, object_metadata)
    out = attach_overlap_proxy(out, bands=overlap_bands)
    return out


def proxy_definitions(color: str = "g-z") -> list[dict[str, Any]]:
    """Return the independent contamination proxies evaluated by the notebook."""
    band1, band2 = parse_color(color)
    label = color.replace("-", "_")
    return [
        {
            "proxy": "deblend_fluxOverlapFraction",
            "metric_col": "max_deblend_fluxOverlapFraction",
            "high_label": "high_overlap",
            "low_label": "low_overlap",
            "description": (
                "Catalog deblend overlap: high values indicate the deblender "
                "assigned substantial neighbor flux in at least one object footprint."
            ),
            "expected_positive_delta": True,
            "requires_metadata": True,
        },
        {
            "proxy": f"background_{band1}_brightness",
            "metric_col": f"log10_background_{band1}_flux",
            "high_label": f"bg_bright_in_{band1}",
            "low_label": f"bg_faint_in_{band1}",
            "description": (
                f"Background {band1}-band flux only: high values mean the "
                f"background is brighter in the blue band. For a color defined "
                "as blue-red, blue foreground leakage should have a smaller "
                "fractional effect on high-background-flux objects."
            ),
            "expected_positive_delta": True,
            "requires_metadata": False,
        },
        {
            "proxy": f"background_{band2}_brightness",
            "metric_col": f"log10_background_{band2}_flux",
            "high_label": f"bg_bright_in_{band2}",
            "low_label": f"bg_faint_in_{band2}",
            "description": (
                f"Background {band2}-band flux only: high values mean the "
                f"background is brighter in the red band. For a color defined "
                "as blue-red, red foreground leakage should have a smaller "
                "fractional effect on high-background-flux objects."
            ),
            "expected_positive_delta": False,
            "requires_metadata": False,
        },
        {
            "proxy": "foreground_gz_color",
            "metric_col": f"foreground_{label}",
            "high_label": "red_foreground",
            "low_label": "blue_foreground",
            "description": (
                "Foreground g-z color only: high values isolate red foregrounds "
                "without conditioning on the observed background color."
            ),
            "expected_positive_delta": True,
            "requires_metadata": False,
        },
    ]


def attach_proxy_groups(
    pairs: pd.DataFrame,
    *,
    metric_col: str,
    group_col: str,
    high_label: str = HIGH_GROUP,
    low_label: str = LOW_GROUP,
    split_quantile: float = 0.5,
) -> tuple[pd.DataFrame, float]:
    """Split valid pairs into high/low groups by a proxy metric quantile."""
    out = pairs.copy()
    values = pd.to_numeric(out.get(metric_col, np.nan), errors="coerce")
    stack_valid = out.get(
        "valid_flipped_corrected_stack_color",
        out.get("valid_background_stack_color", False),
    )
    valid = np.isfinite(values) & stack_valid
    if not np.any(valid):
        out[group_col] = INVALID_GROUP
        return out, np.nan

    threshold = float(np.nanquantile(values[valid], split_quantile))
    out[group_col] = INVALID_GROUP
    out.loc[valid & (values >= threshold), group_col] = high_label
    out.loc[valid & (values < threshold), group_col] = low_label
    return out, threshold


def fcolor_stack(values: Sequence[float], errors: Sequence[float]) -> dict[str, float]:
    """Compute the same one-bin fcolors average used by TreeCorrStacker."""
    value = np.asarray(values, dtype=float)
    error = np.asarray(errors, dtype=float)
    good = np.isfinite(value) & np.isfinite(error) & (value > 0) & (error > 0)
    value = value[good]
    error = error[good]
    if len(value) == 0:
        return {
            "n": 0,
            "raw_ratio": np.nan,
            "raw_ratio_err": np.nan,
            "gz_color": np.nan,
            "gz_color_err": np.nan,
            "variance_floor": np.nan,
        }

    mean = np.mean(value)
    mean2 = np.mean(value**2)
    variance_floor = max(mean2 - mean**2, 0.0)
    weight = 1.0 / (error**2 + variance_floor)
    raw_ratio = np.sum(value * weight) / np.sum(weight)
    raw_ratio_err = np.sqrt(1.0 / np.sum(weight))
    color = -2.5 * np.log10(raw_ratio)
    color_err = 2.5 / np.log(10.0) * raw_ratio_err / raw_ratio
    return {
        "n": int(len(value)),
        "raw_ratio": float(raw_ratio),
        "raw_ratio_err": float(raw_ratio_err),
        "gz_color": float(color),
        "gz_color_err": float(color_err),
        "variance_floor": float(variance_floor),
    }


def flipped_corrected_fcolor_stack(
    split_pairs: pd.DataFrame,
    *,
    color: str = "g-z",
) -> dict[str, float]:
    """Compute a no-random/no-reference forward-minus-flipped pair stack."""
    label = color.replace("-", "_")
    required = [
        f"background_flux_ratio_{label}",
        f"background_flux_ratio_{label}_err",
        f"foreground_flux_ratio_{label}",
        f"foreground_flux_ratio_{label}_err",
    ]
    if not set(required).issubset(split_pairs.columns):
        return {
            "n": 0,
            "gz_color": np.nan,
            "gz_color_err": np.nan,
            "forward_gz_color": np.nan,
            "forward_gz_color_err": np.nan,
            "flipped_gz_color": np.nan,
            "flipped_gz_color_err": np.nan,
            "forward_raw_ratio": np.nan,
            "flipped_raw_ratio": np.nan,
        }

    values = split_pairs.loc[:, required].apply(pd.to_numeric, errors="coerce")
    good = (
        np.isfinite(values[required[0]])
        & np.isfinite(values[required[1]])
        & np.isfinite(values[required[2]])
        & np.isfinite(values[required[3]])
        & (values[required[0]] > 0)
        & (values[required[1]] > 0)
        & (values[required[2]] > 0)
        & (values[required[3]] > 0)
    )
    if not np.any(good):
        return {
            "n": 0,
            "gz_color": np.nan,
            "gz_color_err": np.nan,
            "forward_gz_color": np.nan,
            "forward_gz_color_err": np.nan,
            "flipped_gz_color": np.nan,
            "flipped_gz_color_err": np.nan,
            "forward_raw_ratio": np.nan,
            "flipped_raw_ratio": np.nan,
        }

    forward = fcolor_stack(values.loc[good, required[0]], values.loc[good, required[1]])
    flipped = fcolor_stack(values.loc[good, required[2]], values.loc[good, required[3]])
    corrected = forward["gz_color"] - flipped["gz_color"]
    corrected_err = np.hypot(forward["gz_color_err"], flipped["gz_color_err"])
    return {
        "n": int(np.sum(good)),
        "gz_color": float(corrected),
        "gz_color_err": float(corrected_err),
        "forward_gz_color": forward["gz_color"],
        "forward_gz_color_err": forward["gz_color_err"],
        "flipped_gz_color": flipped["gz_color"],
        "flipped_gz_color_err": flipped["gz_color_err"],
        "forward_raw_ratio": forward["raw_ratio"],
        "flipped_raw_ratio": flipped["raw_ratio"],
    }


def stack_for_group(
    split_pairs: pd.DataFrame,
    group: str,
    *,
    color: str = "g-z",
) -> dict[str, Any]:
    """Compute a one-bin fcolors stack for one color-contrast group."""
    label = color.replace("-", "_")
    use = split_pairs["color_group"].eq(group)
    result = fcolor_stack(
        split_pairs.loc[use, f"background_flux_ratio_{label}"],
        split_pairs.loc[use, f"background_flux_ratio_{label}_err"],
    )
    result["color_group"] = group
    return result


def stack_for_proxy_group(
    split_pairs: pd.DataFrame,
    *,
    group_col: str,
    group: str,
    color: str = "g-z",
) -> dict[str, Any]:
    """Compute a flipped-corrected one-bin fcolors stack for one proxy group."""
    use = split_pairs[group_col].eq(group)
    result = flipped_corrected_fcolor_stack(split_pairs.loc[use], color=color)
    result["group"] = group
    return result


def bootstrap_group_color(
    split_pairs: pd.DataFrame,
    group: str,
    *,
    color: str = "g-z",
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bootstrap group mini-stack colors by resampling pairs."""
    rng = np.random.default_rng() if rng is None else rng
    label = color.replace("-", "_")
    columns = [f"background_flux_ratio_{label}", f"background_flux_ratio_{label}_err"]
    sub = split_pairs.loc[split_pairs["color_group"].eq(group), columns].dropna()
    values = sub[columns[0]].to_numpy(float)
    errors = sub[columns[1]].to_numpy(float)
    if len(values) == 0:
        return np.full(n_samples, np.nan)

    out = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        sample = rng.integers(0, len(values), len(values))
        out[i] = fcolor_stack(values[sample], errors[sample])["gz_color"]
    return out


def permutation_differences(
    split_pairs: pd.DataFrame,
    *,
    color: str = "g-z",
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Permute split labels and recompute fg-redder minus bg-redder color."""
    rng = np.random.default_rng() if rng is None else rng
    label = color.replace("-", "_")
    use = split_pairs["color_group"].isin(GROUPS)
    columns = [
        "color_group",
        f"background_flux_ratio_{label}",
        f"background_flux_ratio_{label}_err",
    ]
    sub = split_pairs.loc[use, columns].copy()
    values = sub[f"background_flux_ratio_{label}"].to_numpy(float)
    errors = sub[f"background_flux_ratio_{label}_err"].to_numpy(float)
    labels = sub["color_group"].to_numpy(str)
    n_fg_redder = int(np.sum(labels == FG_REDDER_GROUP))

    out = np.empty(n_samples, dtype=float)
    indices = np.arange(len(values))
    for i in range(n_samples):
        fg_idx = rng.choice(indices, size=n_fg_redder, replace=False)
        mask = np.zeros(len(values), dtype=bool)
        mask[fg_idx] = True
        fg_color = fcolor_stack(values[mask], errors[mask])["gz_color"]
        bg_color = fcolor_stack(values[~mask], errors[~mask])["gz_color"]
        out[i] = fg_color - bg_color
    return out


def bootstrap_proxy_group_color(
    split_pairs: pd.DataFrame,
    *,
    group_col: str,
    group: str,
    color: str = "g-z",
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bootstrap proxy group flipped-corrected colors by resampling pairs."""
    rng = np.random.default_rng() if rng is None else rng
    sub = split_pairs.loc[split_pairs[group_col].eq(group)].copy()
    if sub.empty:
        return np.full(n_samples, np.nan)

    out = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        sample = rng.integers(0, len(sub), len(sub))
        out[i] = flipped_corrected_fcolor_stack(
            sub.iloc[sample],
            color=color,
        )["gz_color"]
    return out


def proxy_permutation_differences(
    split_pairs: pd.DataFrame,
    *,
    group_col: str,
    high_label: str,
    low_label: str,
    color: str = "g-z",
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Permute proxy labels and recompute corrected high-minus-low color."""
    rng = np.random.default_rng() if rng is None else rng
    label = color.replace("-", "_")
    use = split_pairs[group_col].isin([high_label, low_label])
    columns = [
        group_col,
        f"background_flux_ratio_{label}",
        f"background_flux_ratio_{label}_err",
        f"foreground_flux_ratio_{label}",
        f"foreground_flux_ratio_{label}_err",
    ]
    sub = split_pairs.loc[use, columns].copy()
    labels = sub[group_col].to_numpy(str)
    n_high = int(np.sum(labels == high_label))

    out = np.empty(n_samples, dtype=float)
    indices = np.arange(len(sub))
    for i in range(n_samples):
        high_idx = rng.choice(indices, size=n_high, replace=False)
        mask = np.zeros(len(sub), dtype=bool)
        mask[high_idx] = True
        high_color = flipped_corrected_fcolor_stack(sub.loc[mask], color=color)[
            "gz_color"
        ]
        low_color = flipped_corrected_fcolor_stack(sub.loc[~mask], color=color)[
            "gz_color"
        ]
        out[i] = high_color - low_color
    return out


def analyze_proxy_split_bin(
    pair_inputs: pd.DataFrame,
    *,
    stack_id: str,
    bin_index: int,
    r_min_kpc: float,
    r_max_kpc: float,
    proxy: str,
    metric_col: str,
    high_label: str,
    low_label: str,
    color: str = "g-z",
    split_quantile: float = 0.5,
    bootstrap_samples: int = 5000,
    permutation_samples: int = 5000,
    rng: np.random.Generator | None = None,
    expected_positive_delta: bool = True,
) -> dict[str, Any]:
    """Compute high-vs-low proxy split stacks and significance metrics."""
    rng = np.random.default_rng() if rng is None else rng
    group_col = f"{proxy}_group"
    split_pairs, threshold = attach_proxy_groups(
        pair_inputs,
        metric_col=metric_col,
        group_col=group_col,
        high_label=high_label,
        low_label=low_label,
        split_quantile=split_quantile,
    )

    rows = [
        stack_for_proxy_group(
            split_pairs,
            group_col=group_col,
            group=high_label,
            color=color,
        ),
        stack_for_proxy_group(
            split_pairs,
            group_col=group_col,
            group=low_label,
            color=color,
        ),
    ]
    valid = split_pairs[group_col].isin([high_label, low_label])
    all_valid = flipped_corrected_fcolor_stack(split_pairs.loc[valid], color=color)
    all_valid["group"] = "all_valid_proxy_pairs"
    rows.append(all_valid)
    stack_table = pd.DataFrame(rows).set_index("group")

    high_color = stack_table.loc[high_label, "gz_color"]
    low_color = stack_table.loc[low_label, "gz_color"]
    observed_diff = high_color - low_color
    valid_metric = pd.to_numeric(pair_inputs.get(metric_col, np.nan), errors="coerce")
    counts = split_pairs[group_col].value_counts()
    n_high = int(counts.get(high_label, 0))
    n_low = int(counts.get(low_label, 0))
    has_comparison = n_high > 0 and n_low > 0 and np.isfinite(observed_diff)
    if has_comparison:
        high_boot = bootstrap_proxy_group_color(
            split_pairs,
            group_col=group_col,
            group=high_label,
            color=color,
            n_samples=bootstrap_samples,
            rng=rng,
        )
        low_boot = bootstrap_proxy_group_color(
            split_pairs,
            group_col=group_col,
            group=low_label,
            color=color,
            n_samples=bootstrap_samples,
            rng=rng,
        )
        diff_boot = high_boot - low_boot
        diff_sigma = np.nanstd(diff_boot, ddof=1)
        z_score = observed_diff / diff_sigma if diff_sigma > 0 else np.nan
        ci16, ci50, ci84 = np.nanpercentile(diff_boot, [16, 50, 84])
        ci025, ci975 = np.nanpercentile(diff_boot, [2.5, 97.5])

        perm_diff = proxy_permutation_differences(
            split_pairs,
            group_col=group_col,
            high_label=high_label,
            low_label=low_label,
            color=color,
            n_samples=permutation_samples,
            rng=rng,
        )
        p_two_sided = (1.0 + np.sum(np.abs(perm_diff) >= abs(observed_diff))) / (
            len(perm_diff) + 1.0
        )
        p_positive = (1.0 + np.sum(perm_diff >= observed_diff)) / (len(perm_diff) + 1.0)
        expected_sign_detected = (
            observed_diff > 0 if expected_positive_delta else observed_diff < 0
        )
    else:
        diff_boot = np.full(bootstrap_samples, np.nan)
        perm_diff = np.full(permutation_samples, np.nan)
        diff_sigma = np.nan
        z_score = np.nan
        ci16 = ci50 = ci84 = np.nan
        ci025 = ci975 = np.nan
        p_two_sided = np.nan
        p_positive = np.nan
        expected_sign_detected = False

    summary = {
        "stack_id": stack_id,
        "color": color,
        "bin_index": bin_index,
        "bin_number": bin_index + 1,
        "bin_label": radial_bin_label(bin_index, r_min_kpc, r_max_kpc),
        "r_min_kpc": r_min_kpc,
        "r_max_kpc": r_max_kpc,
        "proxy": proxy,
        "estimator": "forward_minus_flipped_no_random_no_reference",
        "metric_col": metric_col,
        "high_label": high_label,
        "low_label": low_label,
        "split_threshold": threshold,
        "n_positional_pairs": int(len(split_pairs)),
        "n_metric_finite": int(np.isfinite(valid_metric).sum()),
        "n_proxy_valid": int(valid.sum()),
        "n_high": n_high,
        "n_low": n_low,
        "n_invalid_or_unavailable": int(counts.get(INVALID_GROUP, 0)),
        "high_stack_gz": high_color,
        "low_stack_gz": low_color,
        "all_valid_stack_gz": stack_table.loc["all_valid_proxy_pairs", "gz_color"],
        "delta_high_minus_low_gz_mag": observed_diff,
        "bootstrap_sigma_mag": diff_sigma,
        "bootstrap_z": z_score,
        "bootstrap_ci16_mag": ci16,
        "bootstrap_ci50_mag": ci50,
        "bootstrap_ci84_mag": ci84,
        "bootstrap_ci025_mag": ci025,
        "bootstrap_ci975_mag": ci975,
        "permutation_p_two_sided": p_two_sided,
        "permutation_p_positive_delta": p_positive,
        "significant_at_2sigma": bool(np.isfinite(z_score) and abs(z_score) >= 2.0),
        "significant_by_permutation_0p05": bool(
            np.isfinite(p_two_sided) and p_two_sided < 0.05
        ),
        "expected_contamination_sign_detected": bool(expected_sign_detected),
        "effect_sign": (
            "positive"
            if observed_diff > 0
            else "negative" if observed_diff < 0 else "zero"
        ),
    }
    return {
        "summary": summary,
        "split_pairs": split_pairs,
        "stack_table": stack_table,
        "diff_bootstrap": diff_boot,
        "diff_permutation": perm_diff,
        "group_col": group_col,
    }


def analyze_color_split_bin(
    pairs: pd.DataFrame,
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    *,
    stack_id: str,
    bin_index: int,
    r_min_kpc: float,
    r_max_kpc: float,
    color: str = "g-z",
    snr_max: float | None = 100.0,
    bootstrap_samples: int = 5000,
    permutation_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Compute color-split stacks and significance metrics for one bin."""
    rng = np.random.default_rng() if rng is None else rng
    split_pairs = attach_color_split_columns(
        pairs,
        foreground,
        background,
        color=color,
        snr_max=snr_max,
    )
    label = color.replace("-", "_")
    rows = [stack_for_group(split_pairs, group, color=color) for group in GROUPS]
    valid_groups = split_pairs["color_group"].isin(GROUPS)
    all_valid = fcolor_stack(
        split_pairs.loc[valid_groups, f"background_flux_ratio_{label}"],
        split_pairs.loc[valid_groups, f"background_flux_ratio_{label}_err"],
    )
    all_valid["color_group"] = "all_color_valid_pairs"
    rows.append(all_valid)
    stack_table = pd.DataFrame(rows).set_index("color_group")

    fg_boot = bootstrap_group_color(
        split_pairs,
        FG_REDDER_GROUP,
        color=color,
        n_samples=bootstrap_samples,
        rng=rng,
    )
    bg_boot = bootstrap_group_color(
        split_pairs,
        BG_REDDER_GROUP,
        color=color,
        n_samples=bootstrap_samples,
        rng=rng,
    )
    diff_boot = fg_boot - bg_boot

    fg_color = stack_table.loc[FG_REDDER_GROUP, "gz_color"]
    bg_color = stack_table.loc[BG_REDDER_GROUP, "gz_color"]
    observed_diff = fg_color - bg_color
    diff_sigma = np.nanstd(diff_boot, ddof=1)
    z_score = observed_diff / diff_sigma if diff_sigma > 0 else np.nan
    ci16, ci50, ci84 = np.nanpercentile(diff_boot, [16, 50, 84])
    ci025, ci975 = np.nanpercentile(diff_boot, [2.5, 97.5])

    perm_diff = permutation_differences(
        split_pairs,
        color=color,
        n_samples=permutation_samples,
        rng=rng,
    )
    p_two_sided = (1.0 + np.sum(np.abs(perm_diff) >= abs(observed_diff))) / (
        len(perm_diff) + 1.0
    )
    p_positive = (1.0 + np.sum(perm_diff >= observed_diff)) / (len(perm_diff) + 1.0)

    counts = split_pairs["color_group"].value_counts()
    summary = {
        "stack_id": stack_id,
        "color": color,
        "bin_index": bin_index,
        "bin_number": bin_index + 1,
        "bin_label": radial_bin_label(bin_index, r_min_kpc, r_max_kpc),
        "r_min_kpc": r_min_kpc,
        "r_max_kpc": r_max_kpc,
        "n_positional_pairs": int(len(split_pairs)),
        "n_color_valid": int(split_pairs["color_group"].isin(GROUPS).sum()),
        "n_invalid_or_tie": int(counts.get(INVALID_GROUP, 0)),
        "n_fg_redder": int(stack_table.loc[FG_REDDER_GROUP, "n"]),
        "n_bg_redder": int(stack_table.loc[BG_REDDER_GROUP, "n"]),
        "fg_redder_stack_gz": fg_color,
        "bg_redder_stack_gz": bg_color,
        "all_valid_stack_gz": stack_table.loc["all_color_valid_pairs", "gz_color"],
        "delta_gz_mag": observed_diff,
        "bootstrap_sigma_mag": diff_sigma,
        "bootstrap_z": z_score,
        "bootstrap_ci16_mag": ci16,
        "bootstrap_ci50_mag": ci50,
        "bootstrap_ci84_mag": ci84,
        "bootstrap_ci025_mag": ci025,
        "bootstrap_ci975_mag": ci975,
        "permutation_p_two_sided": p_two_sided,
        "permutation_p_positive_delta": p_positive,
        "significant_at_2sigma": bool(np.isfinite(z_score) and abs(z_score) >= 2.0),
        "significant_by_permutation_0p05": bool(p_two_sided < 0.05),
        "contamination_sign": (
            "positive"
            if observed_diff > 0
            else "negative" if observed_diff < 0 else "zero"
        ),
    }
    return {
        "summary": summary,
        "split_pairs": split_pairs,
        "stack_table": stack_table,
        "diff_bootstrap": diff_boot,
        "diff_permutation": perm_diff,
        "pair_summary": pair_table_summary(split_pairs),
    }


def load_saved_stack_bin_context(
    stack_dir: str | Path,
    *,
    color: str = "g-z",
    mode: str = "fcolors",
) -> pd.DataFrame:
    """Load saved full-stack and forward-stack values for comparison."""
    stack_dir = Path(stack_dir)
    stack_file = stack_dir / f"stack_{mode}.npz"
    provenance_file = stack_dir / f"stack_{mode}_provenance.npz"
    if not stack_file.exists() or not provenance_file.exists():
        return pd.DataFrame()
    with np.load(stack_file) as stack_data, np.load(provenance_file) as provenance:
        radius = np.asarray(stack_data[f"{color}_bin_centers"], dtype=float)
        return pd.DataFrame(
            {
                "bin_index": np.arange(len(radius), dtype=int),
                "bin_number": np.arange(len(radius), dtype=int) + 1,
                "bin_center_kpc": radius,
                "saved_forward_gz": np.asarray(
                    stack_data[f"{color}_forward_avg"], dtype=float
                ),
                "saved_forward_gz_err": np.asarray(
                    stack_data[f"{color}_forward_err"], dtype=float
                ),
                "saved_corrected_gz": np.asarray(
                    stack_data[f"{color}_avg"], dtype=float
                ),
                "saved_corrected_gz_err": np.asarray(
                    stack_data[f"{color}_err"], dtype=float
                ),
                "saved_forward_npairs": np.asarray(
                    provenance[f"{color}_forward_npairs"], dtype=float
                ),
            }
        )


def attach_proxy_signal_normalization(
    proxy_summary: pd.DataFrame,
    *,
    delta_col: str = "delta_high_minus_low_gz_mag",
    signal_col: str = "saved_corrected_gz",
    error_col: str = "saved_corrected_gz_err",
    signal_alias: str = "regular_corrected_gz_signal_mag",
    error_alias: str = "regular_corrected_gz_signal_err_mag",
    ratio_col: str = "delta_over_regular_corrected_gz_signal",
    sigma_col: str = "delta_over_regular_corrected_gz_err",
) -> pd.DataFrame:
    """Attach proxy-delta normalization by the regular signal and its error."""
    out = proxy_summary.copy()
    if delta_col not in out or signal_col not in out:
        out[signal_alias] = np.nan
        out[ratio_col] = np.nan
    else:
        delta = pd.to_numeric(out[delta_col], errors="coerce")
        signal = pd.to_numeric(out[signal_col], errors="coerce")
        valid = np.isfinite(delta) & np.isfinite(signal) & (signal != 0)
        out[signal_alias] = signal
        out[ratio_col] = np.nan
        out.loc[valid, ratio_col] = delta.loc[valid] / signal.loc[valid]

    if delta_col not in out or error_col not in out:
        out[error_alias] = np.nan
        out[sigma_col] = np.nan
        return out

    delta = pd.to_numeric(out[delta_col], errors="coerce")
    error = pd.to_numeric(out[error_col], errors="coerce")
    valid = np.isfinite(delta) & np.isfinite(error) & (error > 0)
    out[error_alias] = error
    out[sigma_col] = np.nan
    out.loc[valid, sigma_col] = delta.loc[valid] / error.loc[valid]
    return out


def _bool_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(float).ne(0)
    return (
        values.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y"})
    )


def _top_fraction_mask(
    values: pd.Series,
    valid: pd.Series,
    fraction: float,
) -> tuple[pd.Series, float]:
    finite = valid & np.isfinite(values)
    out = pd.Series(False, index=values.index)
    if not np.any(finite):
        return out, np.nan
    n_remove = int(np.ceil(float(fraction) * int(finite.sum())))
    if n_remove <= 0:
        return out, np.nan
    ordered = values.loc[finite].sort_values(ascending=False)
    selected = ordered.index[:n_remove]
    out.loc[selected] = True
    return out, float(ordered.iloc[n_remove - 1])


def _review_label_map(review_labels: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    labels = review_labels.copy()
    if "pair_id" not in labels or "blend_review" not in labels:
        empty = pd.Series(dtype=object)
        return empty, empty
    normalized = labels["blend_review"].fillna("").astype(str).str.strip().str.lower()
    by_pair_id = pd.Series(normalized.to_numpy(), index=labels["pair_id"].astype(str))
    suffix = labels["pair_id"].astype(str).str.extract(r"_(\d+)$", expand=False)
    by_suffix = pd.Series(normalized.to_numpy(), index=suffix)
    return (
        by_pair_id[~by_pair_id.index.duplicated(keep="last")],
        by_suffix[~by_suffix.index.duplicated(keep="last")],
    )


def attach_inner_bin_exclusion_flags(
    pairs: pd.DataFrame,
    *,
    review_labels: pd.DataFrame | None = None,
    overlap_col: str = "max_deblend_fluxOverlapFraction",
) -> pd.DataFrame:
    """Attach blend-review exclusion flags for inner-bin robustness reruns."""
    out = pairs.copy()
    valid = (
        out["valid_background_stack_color"].fillna(False).astype(bool)
        if "valid_background_stack_color" in out
        else pd.Series(False, index=out.index)
    )
    overlap = pd.to_numeric(out.get(overlap_col, np.nan), errors="coerce")
    top10, top10_threshold = _top_fraction_mask(overlap, valid, 0.10)
    top20, top20_threshold = _top_fraction_mask(overlap, valid, 0.20)

    fg_parent = pd.to_numeric(
        out.get("foreground_parentObjectId", np.nan), errors="coerce"
    )
    bg_parent = pd.to_numeric(
        out.get("background_parentObjectId", np.nan), errors="coerce"
    )
    same_parent = (
        np.isfinite(fg_parent)
        & np.isfinite(bg_parent)
        & (fg_parent > 0)
        & (fg_parent == bg_parent)
    )
    deblend_failed = _bool_column(out, "foreground_deblend_failed") | _bool_column(
        out, "background_deblend_failed"
    )

    manual_label = pd.Series("", index=out.index, dtype=object)
    if review_labels is not None and not review_labels.empty and "pair_id" in out:
        by_pair_id, by_suffix = _review_label_map(review_labels)
        manual_label = out["pair_id"].astype(str).map(by_pair_id).fillna("")
        missing = manual_label.eq("")
        if np.any(missing) and not by_suffix.empty:
            suffix = (
                out.loc[missing, "pair_id"]
                .astype(str)
                .str.extract(r"_(\d+)$", expand=False)
            )
            manual_label.loc[missing] = suffix.map(by_suffix).fillna("")

    out["manual_blend_review"] = manual_label
    out["exclude_overlap_top10pct"] = top10
    out["exclude_overlap_top20pct"] = top20
    out["exclude_same_parent"] = same_parent
    out["exclude_deblend_failed"] = deblend_failed
    out["exclude_manual_reject"] = manual_label.eq("reject")
    out["exclude_manual_maybe_or_reject"] = manual_label.isin({"maybe", "reject"})
    out["exclude_practical_blend_risk"] = (
        out["exclude_overlap_top20pct"]
        | out["exclude_deblend_failed"]
        | out["exclude_manual_maybe_or_reject"]
    )
    out["overlap_top10pct_threshold"] = top10_threshold
    out["overlap_top20pct_threshold"] = top20_threshold
    return out


def inner_bin_exclusion_scenarios() -> list[dict[str, str | None]]:
    """Return default inner-bin exclusion rerun scenarios."""
    return [
        {
            "scenario": "no_exclusion",
            "mask_col": None,
            "description": "All valid inner-bin stack pairs.",
        },
        {
            "scenario": "remove_overlap_top10pct",
            "mask_col": "exclude_overlap_top10pct",
            "description": "Remove the highest 10% in max deblend flux-overlap fraction.",
        },
        {
            "scenario": "remove_overlap_top20pct",
            "mask_col": "exclude_overlap_top20pct",
            "description": "Remove the highest 20% in max deblend flux-overlap fraction.",
        },
        {
            "scenario": "remove_deblend_failed",
            "mask_col": "exclude_deblend_failed",
            "description": "Remove pairs where either Object has deblend_failed=True.",
        },
        {
            "scenario": "remove_manual_reject",
            "mask_col": "exclude_manual_reject",
            "description": "Remove pairs manually labelled reject in review_labels.csv.",
        },
        {
            "scenario": "remove_manual_maybe_or_reject",
            "mask_col": "exclude_manual_maybe_or_reject",
            "description": "Remove pairs manually labelled maybe or reject in review_labels.csv.",
        },
        {
            "scenario": "remove_practical_blend_risk",
            "mask_col": "exclude_practical_blend_risk",
            "description": (
                "Remove top-20% overlap, deblend failures, and manual maybe/reject labels."
            ),
        },
        {
            "scenario": "remove_same_parent",
            "mask_col": "exclude_same_parent",
            "description": (
                "Remove pairs where both Objects share a nonzero parentObjectId; "
                "this may be overly aggressive for close-pair samples."
            ),
        },
    ]


def summarize_inner_bin_exclusion_rerun(
    pairs: pd.DataFrame,
    *,
    saved_context: Mapping[str, Any] | pd.Series | None = None,
    color: str = "g-z",
    scenarios: Sequence[Mapping[str, str | None]] | None = None,
) -> pd.DataFrame:
    """Summarize inner-bin one-bin stacks after excluding suspicious pairs."""
    label = color.replace("-", "_")
    value_col = f"background_flux_ratio_{label}"
    error_col = f"background_flux_ratio_{label}_err"
    valid = (
        pairs["valid_background_stack_color"].fillna(False).astype(bool)
        if "valid_background_stack_color" in pairs
        else pd.Series(False, index=pairs.index)
    )
    full_stack = fcolor_stack(pairs.loc[valid, value_col], pairs.loc[valid, error_col])
    context = {} if saved_context is None else dict(saved_context)
    saved_forward = float(context.get("saved_forward_gz", np.nan))
    saved_corrected = float(context.get("saved_corrected_gz", np.nan))
    saved_corrected_err = float(context.get("saved_corrected_gz_err", np.nan))
    baseline_forward = full_stack["gz_color"]
    scenario_defs = inner_bin_exclusion_scenarios() if scenarios is None else scenarios

    rows = []
    for definition in scenario_defs:
        scenario = str(definition["scenario"])
        mask_col = definition.get("mask_col")
        if mask_col is None:
            exclude = pd.Series(False, index=pairs.index)
        elif mask_col in pairs:
            exclude = pairs[mask_col].fillna(False).astype(bool)
        else:
            exclude = pd.Series(False, index=pairs.index)
        threshold_col = None
        if mask_col == "exclude_overlap_top10pct":
            threshold_col = "overlap_top10pct_threshold"
        elif mask_col in {"exclude_overlap_top20pct", "exclude_practical_blend_risk"}:
            threshold_col = "overlap_top20pct_threshold"
        threshold_values = (
            pd.to_numeric(pairs[threshold_col], errors="coerce").dropna()
            if threshold_col in pairs
            else pd.Series(dtype=float)
        )
        exclusion_threshold = (
            float(threshold_values.iloc[0]) if not threshold_values.empty else np.nan
        )
        kept = valid & ~exclude
        removed_valid = valid & exclude
        kept_stack = fcolor_stack(
            pairs.loc[kept, value_col], pairs.loc[kept, error_col]
        )
        delta_forward = kept_stack["gz_color"] - baseline_forward
        corrected_after_exclusion = (
            saved_corrected + delta_forward
            if np.isfinite(saved_corrected) and np.isfinite(delta_forward)
            else np.nan
        )
        delta_over_signal = (
            delta_forward / saved_corrected
            if np.isfinite(delta_forward)
            and np.isfinite(saved_corrected)
            and saved_corrected != 0
            else np.nan
        )
        kept_over_signal = (
            corrected_after_exclusion / saved_corrected
            if np.isfinite(corrected_after_exclusion)
            and np.isfinite(saved_corrected)
            and saved_corrected != 0
            else np.nan
        )
        delta_over_jackknife_sigma = (
            delta_forward / saved_corrected_err
            if np.isfinite(delta_forward)
            and np.isfinite(saved_corrected_err)
            and saved_corrected_err > 0
            else np.nan
        )
        rows.append(
            {
                "scenario": scenario,
                "description": definition.get("description", ""),
                "n_positional_pairs": int(len(pairs)),
                "n_valid_original": int(valid.sum()),
                "n_excluded_total": int(exclude.sum()),
                "n_excluded_valid": int(removed_valid.sum()),
                "n_kept_valid": int(kept.sum()),
                "excluded_valid_fraction": (
                    float(removed_valid.sum() / valid.sum()) if valid.sum() else np.nan
                ),
                "exclusion_threshold": exclusion_threshold,
                "full_forward_gz": baseline_forward,
                "saved_forward_gz": saved_forward,
                "forward_stack_minus_saved_forward_gz": (
                    baseline_forward - saved_forward
                    if np.isfinite(baseline_forward) and np.isfinite(saved_forward)
                    else np.nan
                ),
                "kept_forward_gz": kept_stack["gz_color"],
                "kept_forward_gz_err": kept_stack["gz_color_err"],
                "delta_kept_minus_full_forward_gz": delta_forward,
                "regular_corrected_gz_signal_mag": saved_corrected,
                "regular_corrected_gz_signal_err_mag": saved_corrected_err,
                "approx_corrected_gz_after_exclusion": corrected_after_exclusion,
                "approx_delta_corrected_gz": delta_forward,
                "approx_delta_over_regular_corrected_gz_err": (
                    delta_over_jackknife_sigma
                ),
                "approx_delta_over_regular_corrected_gz_signal": delta_over_signal,
                "approx_corrected_over_regular_corrected_gz_signal": kept_over_signal,
            }
        )
    return pd.DataFrame(rows)


def summarize_inner_bin_pair_influence(
    pairs: pd.DataFrame,
    *,
    saved_context: Mapping[str, Any] | pd.Series | None = None,
    color: str = "g-z",
) -> pd.DataFrame:
    """Return leave-one-out influence diagnostics for valid inner-bin pairs."""
    label = color.replace("-", "_")
    value_col = f"background_flux_ratio_{label}"
    error_col = f"background_flux_ratio_{label}_err"
    valid = (
        pairs["valid_background_stack_color"].fillna(False).astype(bool)
        if "valid_background_stack_color" in pairs
        else pd.Series(False, index=pairs.index)
    )
    full_stack = fcolor_stack(pairs.loc[valid, value_col], pairs.loc[valid, error_col])
    context = {} if saved_context is None else dict(saved_context)
    saved_corrected = float(context.get("saved_corrected_gz", np.nan))
    saved_corrected_err = float(context.get("saved_corrected_gz_err", np.nan))

    rows = []
    for index, pair in pairs.loc[valid].iterrows():
        kept = valid.copy()
        kept.loc[index] = False
        kept_stack = fcolor_stack(
            pairs.loc[kept, value_col],
            pairs.loc[kept, error_col],
        )
        delta_forward = kept_stack["gz_color"] - full_stack["gz_color"]
        corrected_after_removal = (
            saved_corrected + delta_forward
            if np.isfinite(saved_corrected) and np.isfinite(delta_forward)
            else np.nan
        )
        delta_over_signal = (
            delta_forward / saved_corrected
            if np.isfinite(delta_forward)
            and np.isfinite(saved_corrected)
            and saved_corrected != 0
            else np.nan
        )
        delta_over_jackknife_sigma = (
            delta_forward / saved_corrected_err
            if np.isfinite(delta_forward)
            and np.isfinite(saved_corrected_err)
            and saved_corrected_err > 0
            else np.nan
        )
        row = {
            "pair_id": pair.get("pair_id", index),
            "foreground_object_id": pair.get("foreground_object_id", np.nan),
            "background_object_id": pair.get("background_object_id", np.nan),
            "theta_arcsec": pair.get("theta_arcsec", np.nan),
            "r_perp_kpc": pair.get("r_perp_kpc", np.nan),
            "foreground_g_z": pair.get(f"foreground_{label}", np.nan),
            "background_g_z": pair.get(f"background_{label}", np.nan),
            "foreground_background_z_flux_ratio": pair.get(
                "foreground_background_z_flux_ratio", np.nan
            ),
            "max_deblend_fluxOverlapFraction": pair.get(
                "max_deblend_fluxOverlapFraction", np.nan
            ),
            "manual_blend_review": pair.get("manual_blend_review", ""),
            "exclude_overlap_top10pct": pair.get("exclude_overlap_top10pct", False),
            "exclude_overlap_top20pct": pair.get("exclude_overlap_top20pct", False),
            "exclude_same_parent": pair.get("exclude_same_parent", False),
            "exclude_deblend_failed": pair.get("exclude_deblend_failed", False),
            "full_forward_gz": full_stack["gz_color"],
            "leave_one_out_forward_gz": kept_stack["gz_color"],
            "delta_leave_one_out_minus_full_forward_gz": delta_forward,
            "regular_corrected_gz_signal_mag": saved_corrected,
            "regular_corrected_gz_signal_err_mag": saved_corrected_err,
            "approx_corrected_gz_after_pair_removal": corrected_after_removal,
            "approx_delta_over_regular_corrected_gz_signal": delta_over_signal,
            "approx_delta_over_regular_corrected_gz_err": (delta_over_jackknife_sigma),
            "abs_delta_over_regular_corrected_gz_err": abs(delta_over_jackknife_sigma),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        "abs_delta_over_regular_corrected_gz_err",
        ascending=False,
        na_position="last",
    )


def summarize_pair_influence_table(influence: pd.DataFrame) -> pd.DataFrame:
    """Summarize the largest leave-one-out pair influence."""
    if influence.empty:
        return pd.DataFrame(
            [
                {
                    "n_valid_pairs": 0,
                    "max_abs_delta_mag": np.nan,
                    "max_abs_delta_jackknife_sigma": np.nan,
                    "n_abs_shift_gt_0p5sigma": 0,
                    "n_abs_shift_gt_1sigma": 0,
                }
            ]
        )
    abs_sigma = pd.to_numeric(
        influence["abs_delta_over_regular_corrected_gz_err"], errors="coerce"
    )
    abs_delta = pd.to_numeric(
        influence["delta_leave_one_out_minus_full_forward_gz"], errors="coerce"
    ).abs()
    top = influence.iloc[0]
    return pd.DataFrame(
        [
            {
                "n_valid_pairs": int(len(influence)),
                "max_abs_delta_mag": float(abs_delta.max()),
                "max_abs_delta_jackknife_sigma": float(abs_sigma.max()),
                "n_abs_shift_gt_0p5sigma": int((abs_sigma > 0.5).sum()),
                "n_abs_shift_gt_1sigma": int((abs_sigma > 1.0).sum()),
                "most_influential_pair_id": top.get("pair_id", ""),
                "most_influential_delta_mag": top.get(
                    "delta_leave_one_out_minus_full_forward_gz", np.nan
                ),
                "most_influential_delta_jackknife_sigma": top.get(
                    "approx_delta_over_regular_corrected_gz_err", np.nan
                ),
            }
        ]
    )


def contamination_summary_text(
    comparison_table: pd.DataFrame,
    *,
    inner_bin_number: int = 1,
    baseline_bin_number: int = 2,
) -> str:
    """Summarize what the split diagnostic says about inner-bin contamination."""
    if comparison_table.empty:
        return "No comparison table was available."
    keyed = comparison_table.set_index("bin_number")
    if inner_bin_number not in keyed.index or baseline_bin_number not in keyed.index:
        return "Both bin 1 and bin 2 are required for this comparison."
    inner = keyed.loc[inner_bin_number]
    baseline = keyed.loc[baseline_bin_number]

    if inner["delta_gz_mag"] > 0:
        direction = (
            "has the red-foreground contamination sign: backgrounds in the "
            "fg-redder subset are redder than those in the bg-redder subset"
        )
    elif inner["delta_gz_mag"] < 0:
        direction = (
            "has the opposite sign from red-foreground contamination: backgrounds "
            "in the fg-redder subset are bluer than those in the bg-redder subset"
        )
    else:
        direction = "shows no split difference"

    if baseline["delta_gz_mag"] == 0:
        relative = "Bin 2 has no split difference."
    elif np.sign(inner["delta_gz_mag"]) == np.sign(baseline["delta_gz_mag"]):
        relative = "Bin 2 has the same split-difference sign."
    else:
        relative = "Bin 2 has the opposite split-difference sign."

    strength = (
        "statistically notable"
        if bool(inner.get("significant_at_2sigma", False))
        or bool(inner.get("significant_by_permutation_0p05", False))
        else "not statistically notable"
    )
    return (
        f"Bin 1 {direction}. The bin-1 split is {strength} in this diagnostic "
        f"(delta_gz={inner['delta_gz_mag']:.4g} mag, "
        f"bootstrap_z={inner['bootstrap_z']:.3g}, "
        f"permutation_p={inner['permutation_p_two_sided']:.3g}). "
        f"{relative} Bin 2 has delta_gz={baseline['delta_gz_mag']:.4g} mag "
        f"(bootstrap_z={baseline['bootstrap_z']:.3g}, "
        f"permutation_p={baseline['permutation_p_two_sided']:.3g}). "
        "Because the split is defined using observed background color, these "
        "numbers are descriptive contamination diagnostics rather than "
        "independent causal tests."
    )


def proxy_contamination_summary_table(proxy_summary: pd.DataFrame) -> pd.DataFrame:
    """Return human-readable bin-1/bin-2 proxy contamination interpretation."""
    rows = []
    if proxy_summary.empty:
        return pd.DataFrame(columns=["proxy", "interpretation"])
    for proxy, group in proxy_summary.groupby("proxy", sort=False):
        keyed = group.set_index("bin_number")
        if 1 not in keyed.index or 2 not in keyed.index:
            rows.append(
                {
                    "proxy": proxy,
                    "bin1_expected_sign": np.nan,
                    "bin2_expected_sign": np.nan,
                    "interpretation": "Both bin 1 and bin 2 are required.",
                }
            )
            continue
        inner = keyed.loc[1]
        baseline = keyed.loc[2]
        inner_available = int(inner.get("n_proxy_valid", 0)) > 0 and np.isfinite(
            inner["delta_high_minus_low_gz_mag"]
        )
        baseline_available = int(baseline.get("n_proxy_valid", 0)) > 0 and np.isfinite(
            baseline["delta_high_minus_low_gz_mag"]
        )
        inner_expected = bool(inner["expected_contamination_sign_detected"])
        baseline_expected = bool(baseline["expected_contamination_sign_detected"])
        inner_sig = bool(inner["significant_at_2sigma"]) or bool(
            inner["significant_by_permutation_0p05"]
        )
        baseline_sig = bool(baseline["significant_at_2sigma"]) or bool(
            baseline["significant_by_permutation_0p05"]
        )

        if not inner_available or not baseline_available:
            verdict = (
                "This proxy is unavailable or incomplete for at least one of the "
                "first two bins, so it cannot diagnose inner-bin contamination in "
                "this run."
            )
        elif inner_expected and not baseline_expected:
            verdict = (
                "Bin 1 shows the contamination-expected sign while bin 2 does not; "
                "this is a possible inner-bin-specific warning."
            )
        elif inner_expected and baseline_expected:
            verdict = (
                "Both bins show the contamination-expected sign; this does not by "
                "itself indicate a uniquely inner-bin problem."
            )
        elif (not inner_expected) and baseline_expected:
            verdict = (
                "Bin 1 does not show the contamination-expected sign, while bin 2 "
                "does; this argues against an inner-bin-only contamination signal."
            )
        else:
            verdict = (
                "Neither bin shows the contamination-expected sign; this argues "
                "against this proxy indicating contamination in the expected "
                "direction in bin 1."
            )

        rows.append(
            {
                "proxy": proxy,
                "bin1_delta_gz_mag": inner["delta_high_minus_low_gz_mag"],
                "bin1_regular_corrected_gz_signal_mag": inner.get(
                    "regular_corrected_gz_signal_mag", np.nan
                ),
                "bin1_regular_corrected_gz_signal_err_mag": inner.get(
                    "regular_corrected_gz_signal_err_mag", np.nan
                ),
                "bin1_delta_over_regular_corrected_gz_signal": inner.get(
                    "delta_over_regular_corrected_gz_signal", np.nan
                ),
                "bin1_delta_over_regular_corrected_gz_err": inner.get(
                    "delta_over_regular_corrected_gz_err", np.nan
                ),
                "bin1_bootstrap_z": inner["bootstrap_z"],
                "bin1_permutation_p": inner["permutation_p_two_sided"],
                "bin1_expected_sign": inner_expected,
                "bin1_significant": inner_sig,
                "bin2_delta_gz_mag": baseline["delta_high_minus_low_gz_mag"],
                "bin2_regular_corrected_gz_signal_mag": baseline.get(
                    "regular_corrected_gz_signal_mag", np.nan
                ),
                "bin2_regular_corrected_gz_signal_err_mag": baseline.get(
                    "regular_corrected_gz_signal_err_mag", np.nan
                ),
                "bin2_delta_over_regular_corrected_gz_signal": baseline.get(
                    "delta_over_regular_corrected_gz_signal", np.nan
                ),
                "bin2_delta_over_regular_corrected_gz_err": baseline.get(
                    "delta_over_regular_corrected_gz_err", np.nan
                ),
                "bin2_bootstrap_z": baseline["bootstrap_z"],
                "bin2_permutation_p": baseline["permutation_p_two_sided"],
                "bin2_expected_sign": baseline_expected,
                "bin2_significant": baseline_sig,
                "interpretation": (
                    f"{verdict} Bin 1 delta={inner['delta_high_minus_low_gz_mag']:.4g} "
                    f"mag (delta/sigma_jk="
                    f"{inner.get('delta_over_regular_corrected_gz_err', np.nan):.3g}, "
                    f"sigma_jk={inner.get('regular_corrected_gz_signal_err_mag', np.nan):.4g} mag, "
                    f"delta/E(g-z)="
                    f"{inner.get('delta_over_regular_corrected_gz_signal', np.nan):.3g}, "
                    f"E(g-z)={inner.get('regular_corrected_gz_signal_mag', np.nan):.4g} mag, "
                    f"z={inner['bootstrap_z']:.3g}, "
                    f"p={inner['permutation_p_two_sided']:.3g}); "
                    f"bin 2 delta={baseline['delta_high_minus_low_gz_mag']:.4g} "
                    f"mag (delta/sigma_jk="
                    f"{baseline.get('delta_over_regular_corrected_gz_err', np.nan):.3g}, "
                    f"sigma_jk={baseline.get('regular_corrected_gz_signal_err_mag', np.nan):.4g} mag, "
                    f"delta/E(g-z)="
                    f"{baseline.get('delta_over_regular_corrected_gz_signal', np.nan):.3g}, "
                    f"E(g-z)={baseline.get('regular_corrected_gz_signal_mag', np.nan):.4g} mag, "
                    f"z={baseline['bootstrap_z']:.3g}, "
                    f"p={baseline['permutation_p_two_sided']:.3g})."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_color_split_outputs(
    results: Mapping[int, Mapping[str, Any]],
    out_dir: str | Path,
    *,
    color: str = "g-z",
    comparison_table: pd.DataFrame | None = None,
) -> None:
    """Write per-bin split tables and combined comparison outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = color.replace("-", "_")
    for bin_index, result in results.items():
        label = result["summary"]["bin_label"]
        result["split_pairs"].to_csv(
            out_dir / f"{prefix}_{label}_pair_color_split.csv",
            index=False,
        )
        result["stack_table"].to_csv(
            out_dir / f"{prefix}_{label}_color_split_stack.csv"
        )
    if comparison_table is not None:
        comparison_table.to_csv(
            out_dir / f"{prefix}_bin1_bin2_color_split_bias_summary.csv",
            index=False,
        )
