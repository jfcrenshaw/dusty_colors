"""Registry and shared context for post-run analyses.

A post-run analysis reads a completed stack back off disk and writes a derived
product next to it: a figure, a fit report, a summary table. Each one lives in
its own module under this package and registers itself with :func:`register`,
so adding another is a new file plus one import in ``__init__.py`` rather than
an edit to the pipeline.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ResolvedConfig
from ..results import StackResults, load_stack_source

# A stage receives the context and, for per-mode stages, the color mode it is
# being run for. It returns the paths it wrote, which may be empty when the
# stack lacks the inputs the stage needs.
StageFunc = Callable[..., Iterable[Path]]


@dataclass(frozen=True)
class PostRunContext:
    """Everything a post-run analysis needs, with the expensive parts cached."""

    resolved: ResolvedConfig
    stack_dir: Path
    sample_dir: Path
    catalog_dir: Path
    modes: tuple[str, ...]

    # Caches shared across every stage in one run. Mutating a dict on a frozen
    # dataclass is fine; the frozen-ness only guards attribute rebinding.
    _results_cache: dict[str, StackResults] = field(
        default_factory=dict, repr=False, compare=False
    )
    _redshift_cache: list[tuple[float | None, str]] = field(
        default_factory=list, repr=False, compare=False
    )

    @property
    def root(self) -> Path:
        return self.resolved.root

    @property
    def analysis_id(self) -> str:
        return self.resolved.analysis.id

    def results(self, mode: str) -> StackResults:
        """Return the loaded stack for one mode, reading each file once.

        Several stages want the same arrays, and every ``save_*`` helper accepts
        an already-loaded :class:`StackResults`, so caching here turns one npz
        read per stage into one read per mode.
        """

        if mode not in self._results_cache:
            self._results_cache[mode] = load_stack_source(
                self.stack_dir, mode=mode, root=self.root
            )
        return self._results_cache[mode]

    @property
    def foreground_redshift(self) -> tuple[float | None, str]:
        """Return one representative foreground redshift and where it came from.

        Dust-frame wavelengths need a single redshift to de-redshift the filters,
        and both the extinction fit and the chromaticity curve want the same one.
        """

        if not self._redshift_cache:
            self._redshift_cache.append(
                _representative_foreground_redshift(
                    self.sample_dir, self.resolved.sample.data
                )
            )
        return self._redshift_cache[0]

    @property
    def stack_config(self) -> Mapping[str, Any]:
        return _mapping(self.resolved.analysis.data.get("stack"))

    def options(self, name: str) -> Mapping[str, Any]:
        """Return one stage's option block, preferring ``postrun`` over ``stack``.

        Post-run settings belong under ``postrun`` because that block is excluded
        from the stack config hash. The ``stack`` fallback keeps configs written
        before that split working unchanged.
        """

        raw = _mapping(self.resolved.analysis.data.get("postrun")).get(name)
        if raw is None:
            raw = self.stack_config.get(name)
        return _normalize_options(name, raw)

    def enabled(self, name: str, *, default: bool = True) -> bool:
        """Return whether a stage should run.

        A stage is switched off with either ``<name>: false`` or
        ``<name>: {enabled: false}``; both spellings appear in existing configs.
        """

        raw = _mapping(self.resolved.analysis.data.get("postrun")).get(name)
        if raw is None:
            raw = self.stack_config.get(name)
        if raw is False:
            return False
        if raw is None:
            return default
        return bool(_normalize_options(name, raw).get("enabled", True))


@dataclass(frozen=True)
class PostRunAnalysis:
    """One registered analysis and how the runner should invoke it."""

    name: str
    func: StageFunc
    per_mode: bool = True
    default_enabled: bool = True

    def call(self, context: PostRunContext, mode: str | None) -> tuple[Path, ...]:
        outputs = self.func(context, mode) if self.per_mode else self.func(context)
        return tuple(outputs or ())


# Insertion-ordered, so the run order is the import order in __init__.py and
# stays reproducible from one run to the next.
_REGISTRY: dict[str, PostRunAnalysis] = {}


def register(
    name: str,
    *,
    per_mode: bool = True,
    default_enabled: bool = True,
) -> Callable[[StageFunc], StageFunc]:
    """Register a post-run analysis under ``name``.

    Parameters
    ----------
    name : str
        Config key for this stage, looked up under ``postrun`` in the analysis
        YAML.
    per_mode : bool, optional
        When true the stage is called once per configured color mode and
        receives ``(context, mode)``. When false it is called once with just
        ``(context)``, which suits products derived from the sample rather than
        from a particular stack mode.
    default_enabled : bool, optional
        Whether the stage runs when the config says nothing about it.

    Returns
    -------
    callable
        Decorator returning the function unchanged.
    """

    def decorator(func: StageFunc) -> StageFunc:
        if name in _REGISTRY:
            raise ValueError(
                f"A post-run analysis named {name!r} is already registered"
            )
        _REGISTRY[name] = PostRunAnalysis(
            name=name,
            func=func,
            per_mode=per_mode,
            default_enabled=default_enabled,
        )
        return func

    return decorator


def registered_analyses() -> tuple[PostRunAnalysis, ...]:
    """Return every registered analysis in registration order."""

    return tuple(_REGISTRY.values())


def run_post_run_analyses(context: PostRunContext) -> tuple[Path, ...]:
    """Run every enabled post-run analysis and return the paths written.

    A failing stage warns and is skipped rather than aborting the run: these are
    derived products, and losing a figure should not discard a stack that may
    have taken hours. Holding that policy here is what keeps the stages
    themselves free of error-handling boilerplate.
    """

    outputs: list[Path] = []
    for analysis in registered_analyses():
        if not context.enabled(analysis.name, default=analysis.default_enabled):
            continue
        modes: tuple[str | None, ...] = context.modes if analysis.per_mode else (None,)
        for mode in modes:
            try:
                outputs.extend(analysis.call(context, mode))
            except Exception as exc:  # noqa: BLE001 - deliberate: warn, don't abort
                label = analysis.name if mode is None else f"{analysis.name} {mode}"
                warnings.warn(
                    f"Post-run analysis {label} failed for "
                    f"{context.analysis_id}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return tuple(outputs)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_options(name: str, raw: Any) -> Mapping[str, Any]:
    """Coerce a stage's config value to a mapping.

    ``<name>: true`` and a bare ``<name>:`` both mean "run with defaults", and
    ``<name>: false`` is handled by the caller, so all three collapse to an
    empty mapping here.
    """

    if raw is None or raw is True or raw is False:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"postrun.{name} must be a mapping or boolean")
    return raw


def _representative_foreground_redshift(
    sample_dir: Path,
    sample_config: Mapping[str, Any],
) -> tuple[float | None, str]:
    """Return a single foreground redshift for dust-frame wavelengths."""

    parquet_path = Path(sample_dir) / "foreground.parquet"
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


__all__ = [
    "PostRunAnalysis",
    "PostRunContext",
    "register",
    "registered_analyses",
    "run_post_run_analyses",
]
