"""Pipeline orchestration and manifest handling for YAML-first runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, TypeAlias

import yaml

from .config import (
    ResolvedConfig,
    StageConfig,
    format_path,
    load_resolved_config,
    stable_hash,
    write_yaml,
)

PipelineStageKind: TypeAlias = Literal["catalog", "sample", "stack"]
StageAction: TypeAlias = Literal["run", "skip"]
StageHandler: TypeAlias = Callable[["StageContext"], None]

MANIFEST_VERSION = 1


class PipelineError(RuntimeError):
    """Base class for pipeline orchestration errors."""


class ManifestMismatchError(PipelineError):
    """Raised when existing outputs do not match the current resolved config."""

    def __init__(self, spec: "StageSpec", reason: str) -> None:
        flag = force_flag_for(spec.kind)
        message = (
            f"{spec.kind} stage '{spec.config.id}' already has outputs in "
            f"{spec.output_dir}, but {reason}. Pass {flag} or --force-all to "
            "recompute this stage and its dependents."
        )
        super().__init__(message)


class StageHandlerError(PipelineError):
    """Raised when a stage needs to run but no domain handler is available."""


class StageOutputError(PipelineError):
    """Raised when a stage handler finishes without writing expected outputs."""


@dataclass(frozen=True)
class ForceOptions:
    """Command-line force flags with dependency-aware behavior."""

    catalog: bool = False
    sample: bool = False
    stack: bool = False
    all: bool = False

    def effective(self, kind: PipelineStageKind) -> bool:
        if self.all:
            return True
        if kind == "catalog":
            return self.catalog
        if kind == "sample":
            return self.catalog or self.sample
        if kind == "stack":
            return self.catalog or self.sample or self.stack
        raise PipelineError(f"Unknown stage kind: {kind}")


@dataclass(frozen=True)
class StageHandlers:
    """Optional handler injection for tests or alternate stage implementations."""

    catalog: StageHandler | None = None
    sample: StageHandler | None = None
    stack: StageHandler | None = None


@dataclass(frozen=True)
class StageSpec:
    """Resolved execution metadata for one pipeline stage."""

    kind: PipelineStageKind
    config: StageConfig
    output_dir: Path
    expected_outputs: tuple[Path, ...]
    input_hashes: dict[str, str]
    stage_hash: str


@dataclass(frozen=True)
class StageContext:
    """Context passed to imported domain stage functions."""

    kind: PipelineStageKind
    config: StageConfig
    resolved: ResolvedConfig
    output_dir: Path
    expected_outputs: tuple[Path, ...]
    input_dirs: Mapping[str, Path]
    force: bool = False


@dataclass(frozen=True)
class ManifestCheck:
    """Result of checking stage outputs and manifest consistency."""

    action: Literal["run", "skip", "mismatch"]
    reason: str
    manifest: dict[str, Any] | None = None


@dataclass(frozen=True)
class StageRunResult:
    """Outcome for one stage in a pipeline run."""

    kind: PipelineStageKind
    id: str
    action: StageAction
    reason: str
    output_dir: Path
    stage_hash: str


@dataclass(frozen=True)
class PipelineResult:
    """Outcome for a full analysis graph run."""

    resolved: ResolvedConfig
    stages: tuple[StageRunResult, ...] = field(default_factory=tuple)


def run_pipeline(
    config_path: str | Path,
    *,
    root: str | Path | None = None,
    force: ForceOptions | None = None,
    handlers: StageHandlers | None = None,
) -> PipelineResult:
    """Run catalog, sample, and stack stages for an analysis YAML."""

    root_path = Path.cwd() if root is None else Path(root)
    root_path = root_path.resolve()
    resolved = load_resolved_config(config_path, root=root_path)
    force_options = force or ForceOptions()
    specs = build_stage_specs(resolved, root=root_path)
    results: list[StageRunResult] = []

    for kind in ("catalog", "sample", "stack"):
        spec = specs[kind]
        check = check_manifest(
            spec.output_dir,
            spec.expected_outputs,
            expected_manifest(spec, resolved),
        )
        force_stage = force_options.effective(kind)

        if check.action == "skip" and not force_stage:
            if kind == "stack":
                write_resolved_config(spec.output_dir, resolved)
            results.append(_stage_result(spec, "skip", check.reason))
            continue

        if check.action == "mismatch" and not force_stage:
            raise ManifestMismatchError(spec, check.reason)

        run_stage(spec, resolved, force=force_stage, handlers=handlers)
        reason = "forced" if force_stage else check.reason
        results.append(_stage_result(spec, "run", reason))

    return PipelineResult(resolved=resolved, stages=tuple(results))


def build_stage_specs(
    resolved: ResolvedConfig,
    *,
    root: str | Path | None = None,
) -> dict[PipelineStageKind, StageSpec]:
    """Build executable stage specs with dependency-aware hashes."""

    root_path = Path(root).resolve() if root is not None else resolved.root

    catalog_dir = root_path / "results" / "catalogs" / resolved.catalog.id
    sample_dir = root_path / "results" / "samples" / resolved.sample.id
    stack_dir = root_path / "results" / "stacks" / resolved.analysis.id

    catalog_hash = stage_hash(resolved.catalog.config_hash, {})
    sample_input_hashes = {"catalog": catalog_hash}
    sample_hash = stage_hash(resolved.sample.config_hash, sample_input_hashes)
    stack_input_hashes = {"sample": sample_hash}
    stack_hash = stage_hash(resolved.analysis.config_hash, stack_input_hashes)

    return {
        "catalog": StageSpec(
            kind="catalog",
            config=resolved.catalog,
            output_dir=catalog_dir,
            expected_outputs=(
                catalog_dir / "catalog.parquet",
                catalog_dir / "footprint.parquet",
            ),
            input_hashes={},
            stage_hash=catalog_hash,
        ),
        "sample": StageSpec(
            kind="sample",
            config=resolved.sample,
            output_dir=sample_dir,
            expected_outputs=(
                sample_dir / "foreground.parquet",
                sample_dir / "background.parquet",
            ),
            input_hashes=sample_input_hashes,
            stage_hash=sample_hash,
        ),
        "stack": StageSpec(
            kind="stack",
            config=resolved.analysis,
            output_dir=stack_dir,
            expected_outputs=tuple(
                stack_dir / f"stack_{mode}.npz"
                for mode in stack_modes(resolved.analysis)
            ),
            input_hashes=stack_input_hashes,
            stage_hash=stack_hash,
        ),
    }


def stage_hash(config_hash: str, input_hashes: Mapping[str, str]) -> str:
    """Hash a stage's config hash together with relevant upstream stage hashes."""

    return stable_hash({"config_hash": config_hash, "input_hashes": dict(input_hashes)})


def stack_modes(analysis: StageConfig) -> tuple[str, ...]:
    """Return configured stack output modes, defaulting to both color modes."""

    stack = analysis.data.get("stack", {})
    if not isinstance(stack, Mapping):
        return ("fcolors", "mcolors")
    modes = stack.get("modes", ("fcolors", "mcolors"))
    if not isinstance(modes, (list, tuple)):
        raise PipelineError("stack.modes must be a list when provided")
    return tuple(str(mode) for mode in modes)


def run_stage(
    spec: StageSpec,
    resolved: ResolvedConfig,
    *,
    force: bool = False,
    handlers: StageHandlers | None = None,
) -> None:
    """Run one stage through its imported or injected domain handler."""

    handler = stage_handler(spec.kind, handlers)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    context = StageContext(
        kind=spec.kind,
        config=spec.config,
        resolved=resolved,
        output_dir=spec.output_dir,
        expected_outputs=spec.expected_outputs,
        input_dirs=input_dirs_for(spec.kind, resolved),
        force=force,
    )
    handler(context)

    missing = [path for path in spec.expected_outputs if not path.exists()]
    if missing:
        missing_list = ", ".join(path.name for path in missing)
        raise StageOutputError(
            f"{spec.kind} stage '{spec.config.id}' did not write expected outputs: "
            f"{missing_list}"
        )

    if spec.kind == "stack":
        write_resolved_config(spec.output_dir, resolved)
    write_manifest(spec.output_dir, expected_manifest(spec, resolved))


def input_dirs_for(
    kind: PipelineStageKind,
    resolved: ResolvedConfig,
) -> dict[str, Path]:
    root = resolved.root
    catalog_dir = root / "results" / "catalogs" / resolved.catalog.id
    sample_dir = root / "results" / "samples" / resolved.sample.id
    if kind == "catalog":
        return {}
    if kind == "sample":
        return {"catalog": catalog_dir}
    return {"catalog": catalog_dir, "sample": sample_dir}


def stage_handler(
    kind: PipelineStageKind,
    handlers: StageHandlers | None = None,
) -> StageHandler:
    """Return an injected handler or import the default domain stage function."""

    if handlers is not None:
        injected = getattr(handlers, kind)
        if injected is not None:
            return injected

    module_name, candidates = _handler_candidates(kind)
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        raise StageHandlerError(
            f"{kind} stage needs to run, but {module_name} could not be imported: "
            f"{exc}"
        ) from exc

    for name in candidates:
        handler = getattr(module, name, None)
        if callable(handler):
            if name.endswith("_stage"):
                return handler
            return _wrap_domain_handler(kind, handler)

    candidate_list = ", ".join(candidates)
    raise StageHandlerError(
        f"{kind} stage needs to run, but {module_name} does not define one of: "
        f"{candidate_list}"
    )


def expected_manifest(spec: StageSpec, resolved: ResolvedConfig) -> dict[str, Any]:
    """Build the manifest content expected for a completed stage."""

    return {
        "manifest_version": MANIFEST_VERSION,
        "stage": spec.kind,
        "id": spec.config.id,
        "source": format_path(spec.config.path, resolved.root),
        "config_hash": spec.config.config_hash,
        "input_hashes": dict(spec.input_hashes),
        "stage_hash": spec.stage_hash,
        "outputs": [
            format_path(path, spec.output_dir) for path in spec.expected_outputs
        ],
        "created_at": _utc_now(),
    }


def manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "manifest.yaml"


def read_manifest(output_dir: str | Path) -> dict[str, Any] | None:
    path = manifest_path(output_dir)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise PipelineError(f"Manifest {path} must contain a mapping")
    return dict(data)


def write_manifest(output_dir: str | Path, manifest: Mapping[str, Any]) -> None:
    write_yaml(manifest_path(output_dir), manifest)


def check_manifest(
    output_dir: str | Path,
    expected_outputs: tuple[Path, ...],
    manifest: Mapping[str, Any],
) -> ManifestCheck:
    """Check whether a stage should run, skip, or fail for mismatched outputs."""

    missing_outputs = [path for path in expected_outputs if not path.exists()]
    if missing_outputs:
        missing_list = ", ".join(path.name for path in missing_outputs)
        return ManifestCheck("run", f"missing outputs: {missing_list}")

    existing = read_manifest(output_dir)
    if existing is None:
        return ManifestCheck("mismatch", "its manifest is missing")

    mismatches = [
        key
        for key in ("manifest_version", "stage", "id", "config_hash", "stage_hash")
        if existing.get(key) != manifest.get(key)
    ]
    if existing.get("input_hashes") != manifest.get("input_hashes"):
        mismatches.append("input_hashes")

    if mismatches:
        return ManifestCheck(
            "mismatch",
            f"manifest fields differ: {', '.join(mismatches)}",
            existing,
        )

    return ManifestCheck("skip", "outputs and manifest match", existing)


def write_resolved_config(output_dir: str | Path, resolved: ResolvedConfig) -> None:
    write_yaml(Path(output_dir) / "config_resolved.yaml", resolved.to_dict())


def force_flag_for(kind: PipelineStageKind) -> str:
    if kind == "catalog":
        return "--force-catalog"
    if kind == "sample":
        return "--force-sample"
    if kind == "stack":
        return "--force-stack"
    raise PipelineError(f"Unknown stage kind: {kind}")


def _handler_candidates(kind: PipelineStageKind) -> tuple[str, tuple[str, ...]]:
    if kind == "catalog":
        return "dusty_colors.catalogs", ("prepare_catalog",)
    if kind == "sample":
        return "dusty_colors.selection", ("prepare_sample",)
    if kind == "stack":
        return "dusty_colors.treecorr_stacker", ("run_treecorr_stack",)
    raise PipelineError(f"Unknown stage kind: {kind}")


def _wrap_domain_handler(
    kind: PipelineStageKind,
    handler: Callable[..., Any],
) -> StageHandler:
    if kind == "catalog":
        return lambda context: handler(context.config.data, context.output_dir)

    if kind == "sample":
        def run_sample(context: StageContext) -> None:
            catalog_config = context.resolved.catalog.data
            handler(
                context.input_dirs["catalog"] / "catalog.parquet",
                context.config.data,
                context.output_dir,
                bands=catalog_config.get("bands"),
                photometry=catalog_config.get("photometry"),
            )

        return run_sample

    if kind == "stack":
        def run_stack(context: StageContext) -> None:
            handler(
                context.input_dirs["sample"],
                context.output_dir,
                context.config.data.get("stack", {}),
                footprint_path=context.input_dirs["catalog"] / "footprint.parquet",
                force=context.force,
            )

        return run_stack

    raise PipelineError(f"Unknown stage kind: {kind}")


def _stage_result(
    spec: StageSpec,
    action: StageAction,
    reason: str,
) -> StageRunResult:
    return StageRunResult(
        kind=spec.kind,
        id=spec.config.id,
        action=action,
        reason=reason,
        output_dir=spec.output_dir,
        stage_hash=spec.stage_hash,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
