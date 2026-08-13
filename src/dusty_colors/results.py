"""Loading stack outputs written by the TreeCorr estimator.

A finished stack is a directory of NPZ files plus a resolved config. Reading
one back is a separate concern from plotting it, so this module owns the
loading and the fitting and plotting code both build on top of it.

`StackResults` is the in-memory form: the arrays, the color order they should
be interpreted in, and enough provenance to produce a useful error message
when a key is missing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import load_resolved_config
from .pipeline import build_stage_specs, stack_modes

__all__ = ["StackResults", "load_stack_results", "load_stack_source"]


def load_stack_source(
    source: str | Path,
    *,
    mode: str | None = None,
    root: str | Path | None = None,
) -> "StackResults":
    """Load stack results from either a stack directory or an analysis YAML.

    Parameters
    ----------
    source : str or Path
        A stack output directory, or the analysis config that produced it.
    mode : str, optional
        Color mode to load; inferred from the config when omitted.
    root : str or Path, optional
        Repository root used to resolve relative config paths.

    Returns
    -------
    StackResults
        The loaded arrays and their metadata.
    """
    path = Path(source)
    if path.is_dir():
        return load_stack_results(stack_dir=path, mode=mode, root=root)
    return load_stack_results(path, mode=mode, root=root)


@dataclass(frozen=True)
class StackResults:
    """Loaded stack arrays with the config metadata needed for plotting."""

    stack_dir: Path
    mode: str
    colors: tuple[str, ...]
    arrays: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
    config_path: Path | None = None

    @property
    def first_color(self) -> str:
        if not self.colors:
            raise ValueError("Stack results do not define any colors")
        return self.colors[0]

    def require(self, key: str) -> np.ndarray:
        try:
            return self.arrays[key]
        except KeyError as exc:
            raise KeyError(
                f"{self.stack_dir / f'stack_{self.mode}.npz'} is missing {key!r}"
            ) from exc

    def require_diagnostic(self, key: str) -> np.ndarray:
        try:
            return self.diagnostics[key]
        except KeyError as exc:
            path = _stack_diagnostic_file(self.stack_dir, self.mode)
            raise KeyError(f"{path} is missing {key!r}") from exc


def load_stack_results(
    analysis_config: str | Path | None = None,
    *,
    stack_dir: str | Path | None = None,
    mode: str | None = None,
    root: str | Path | None = None,
    colors: Sequence[str] | None = None,
) -> StackResults:
    """Load one ``stack_<mode>.npz`` file plus plotting metadata.

    Passing an analysis YAML is the most reproducible path: the loader reads the
    configured ``stack.colors`` order and resolves the canonical
    ``results/stacks/<analysis-id>`` directory. Passing ``stack_dir`` directly is
    useful for ad hoc outputs and will infer colors from ``config_resolved.yaml``
    or the NPZ keys.
    """

    config_path: Path | None = None
    configured_colors = tuple(str(color) for color in colors or ())
    configured_modes: tuple[str, ...] = ()

    if analysis_config is not None:
        resolved = load_resolved_config(analysis_config, root=root)
        config_path = resolved.analysis.path
        stack_config = resolved.analysis.data.get("stack", {})
        configured_colors = configured_colors or _stack_colors(stack_config)
        configured_modes = stack_modes(resolved.analysis)
        if stack_dir is None:
            stack_dir = build_stage_specs(resolved, root=resolved.root)[
                "stack"
            ].output_dir

    if stack_dir is None:
        raise ValueError("Provide either analysis_config or stack_dir")

    stack_path = Path(stack_dir).resolve()
    if not configured_colors:
        stack_config = _load_stack_config(stack_path)
        configured_colors = _stack_colors(stack_config)
        configured_modes = tuple(
            str(candidate) for candidate in stack_config.get("modes", ())
        )

    selected_mode = _resolve_mode(stack_path, mode, configured_modes)
    arrays = _read_stack_npz(stack_path / f"stack_{selected_mode}.npz")

    # Diagnostics are optional: the stack only writes them when configured to.
    diagnostic_path = _stack_diagnostic_file(stack_path, selected_mode)
    diagnostics = _read_stack_npz(diagnostic_path) if diagnostic_path.exists() else {}

    if not configured_colors:
        # Fall back to whichever colors have a radial profile in the file.
        configured_colors = tuple(
            key.removesuffix("_bin_centers")
            for key in arrays
            if key.endswith("_bin_centers")
        )
    if not configured_colors:
        raise ValueError(f"Could not infer stack colors from {stack_path}")

    return StackResults(
        stack_dir=stack_path,
        mode=selected_mode,
        colors=configured_colors,
        arrays=arrays,
        diagnostics=diagnostics,
        config_path=config_path,
    )


def _read_stack_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _stack_diagnostic_file(stack_dir: Path, mode: str) -> Path:
    return stack_dir / f"stack_{mode}_diagnostics.npz"


def _resolve_mode(
    stack_dir: Path,
    mode: str | None,
    configured_modes: Sequence[str],
) -> str:
    if mode is not None:
        return str(mode)
    if configured_modes:
        return str(configured_modes[0])
    if (stack_dir / "stack_fcolors.npz").exists():
        return "fcolors"
    if (stack_dir / "stack_mcolors.npz").exists():
        return "mcolors"

    matches = [
        path
        for path in sorted(stack_dir.glob("stack_*.npz"))
        if not path.stem.endswith(("_diagnostics", "_provenance"))
    ]
    if len(matches) == 1:
        return matches[0].stem.removeprefix("stack_")
    raise FileNotFoundError(f"Could not choose a stack mode in {stack_dir}")


def _load_stack_config(stack_dir: Path) -> dict[str, Any]:
    path = stack_dir / "config_resolved.yaml"
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        return {}

    analysis = data.get("analysis")
    if isinstance(analysis, Mapping):
        analysis_data = analysis.get("data")
        if isinstance(analysis_data, Mapping):
            stack = analysis_data.get("stack", {})
            return dict(stack) if isinstance(stack, Mapping) else {}

    stack = data.get("stack")
    if isinstance(stack, Mapping):
        return dict(stack)
    return dict(data)


def _stack_colors(stack_config: Mapping[str, Any]) -> tuple[str, ...]:
    colors = stack_config.get("colors", ())
    if not isinstance(colors, Sequence) or isinstance(colors, str):
        return ()
    return tuple(str(color) for color in colors)
