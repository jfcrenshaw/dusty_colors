"""Post-run analysis products written after pipeline stages complete."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from .config import ResolvedConfig


def run_post_run_analyses(
    resolved: ResolvedConfig,
    *,
    stack_dir: str | Path,
    sample_dir: str | Path,
    catalog_dir: str | Path,
    modes: Sequence[str],
) -> tuple[Path, ...]:
    """Write analysis-level products that depend on completed stack outputs."""

    stack_path = Path(stack_dir)
    sample_path = Path(sample_dir)
    catalog_path = Path(catalog_dir)
    outputs: list[Path] = []

    outputs.extend(
        _write_analysis_catalog_stats(
            resolved,
            stack_dir=stack_path,
        )
    )
    outputs.extend(
        _write_dust_extinction_fits(
            resolved,
            stack_dir=stack_path,
            sample_dir=sample_path,
            catalog_dir=catalog_path,
            modes=modes,
        )
    )
    outputs.extend(
        _write_color_power_law_fits(
            resolved,
            stack_dir=stack_path,
            modes=modes,
        )
    )
    return tuple(outputs)


def _write_analysis_catalog_stats(
    resolved: ResolvedConfig,
    *,
    stack_dir: Path,
) -> tuple[Path, ...]:
    from .analysis_stats import save_analysis_catalog_stats

    try:
        return save_analysis_catalog_stats(
            resolved.analysis.path,
            stack_dir,
            root=resolved.root,
            require_current=True,
        )
    except Exception as exc:
        warnings.warn(
            f"Could not write analysis catalog stats for "
            f"{resolved.analysis.id}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return ()


def _write_dust_extinction_fits(
    resolved: ResolvedConfig,
    *,
    stack_dir: Path,
    sample_dir: Path,
    catalog_dir: Path,
    modes: Sequence[str],
) -> tuple[Path, ...]:
    del catalog_dir
    stack_config = _stack_config(resolved.analysis.data)
    foreground_redshift, redshift_source = _representative_foreground_redshift(
        sample_dir,
        resolved.sample.data,
    )

    from .dust_extinction_fit import save_stack_dust_extinction_fit

    outputs: list[Path] = []
    for mode in modes:
        try:
            path = save_stack_dust_extinction_fit(
                stack_dir,
                stack_dir,
                mode=mode,
                root=resolved.root,
                stack_config=stack_config,
                foreground_redshift=foreground_redshift,
                foreground_redshift_source=redshift_source,
            )
        except (OSError, KeyError, ValueError, ModuleNotFoundError) as exc:
            warnings.warn(
                f"Could not write dust-extinction fit for "
                f"{resolved.analysis.id} {mode}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if path is not None:
            outputs.append(path)
    return tuple(outputs)


def _write_color_power_law_fits(
    resolved: ResolvedConfig,
    *,
    stack_dir: Path,
    modes: Sequence[str],
) -> tuple[Path, ...]:
    stack_config = _stack_config(resolved.analysis.data)

    from .color_power_law_fit import save_stack_color_power_law_fits

    outputs: list[Path] = []
    for mode in modes:
        try:
            path = save_stack_color_power_law_fits(
                stack_dir,
                stack_dir,
                mode=mode,
                root=resolved.root,
                stack_config=stack_config,
            )
        except (OSError, KeyError, ValueError) as exc:
            warnings.warn(
                f"Could not write color power-law fits for "
                f"{resolved.analysis.id} {mode}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if path is not None:
            outputs.append(path)
    return tuple(outputs)


def _stack_config(analysis_data: Mapping[str, Any]) -> Mapping[str, Any]:
    stack_config = analysis_data.get("stack", {})
    return stack_config if isinstance(stack_config, Mapping) else {}


def _representative_foreground_redshift(
    sample_dir: Path,
    sample_config: Mapping[str, Any],
) -> tuple[float | None, str]:
    """Return a single foreground redshift for dust-frame wavelengths."""

    parquet_path = sample_dir / "foreground.parquet"
    if parquet_path.exists():
        try:
            import pandas as pd

            foreground = pd.read_parquet(parquet_path, columns=["z_phot"])
            redshift = foreground["z_phot"].to_numpy(float)
            good = np.isfinite(redshift) & (redshift >= 0)
            if np.any(good):
                return float(np.median(redshift[good])), "foreground_median_z_phot"
        except (OSError, KeyError, ValueError, ImportError):
            pass

    selection = sample_config.get("selection", {})
    if isinstance(selection, Mapping):
        foreground_z = selection.get("foreground_z")
        if (
            isinstance(foreground_z, Sequence)
            and not isinstance(foreground_z, str)
            and len(foreground_z) == 2
        ):
            try:
                lo, hi = float(foreground_z[0]), float(foreground_z[1])
            except (TypeError, ValueError):
                pass
            else:
                if np.isfinite(lo) and np.isfinite(hi):
                    return 0.5 * (lo + hi), "selection_foreground_z_midpoint"

    return None, "default_zero"


__all__ = ["run_post_run_analyses"]
