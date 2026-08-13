"""YAML loading and resolution for the three-stage pipeline."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class StageConfig:
    """One resolved stage config."""

    kind: str
    id: str
    path: Path
    data: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ResolvedConfig:
    """Resolved analysis graph: analysis -> sample -> catalog."""

    root: Path
    analysis: StageConfig
    sample: StageConfig
    catalog: StageConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": _stage_to_dict(self.analysis, self.root),
            "sample": _stage_to_dict(self.sample, self.root),
            "catalog": _stage_to_dict(self.catalog, self.root),
        }


def load_resolved_config(
    analysis_path: str | Path,
    *,
    root: str | Path | None = None,
) -> ResolvedConfig:
    """Load and resolve an analysis YAML plus referenced sample/catalog YAML."""
    root_path = Path.cwd().resolve() if root is None else Path(root).resolve()
    analysis_path = _resolve_path(analysis_path, root_path)
    analysis_data = _load_stage_yaml(analysis_path)
    if "sample" not in analysis_data:
        raise ValueError(f"Analysis config missing 'sample': {analysis_path}")

    sample_path = _resolve_ref(analysis_data["sample"], analysis_path.parent, root_path)
    sample_data = _load_stage_yaml(sample_path)
    if "catalog" not in sample_data:
        raise ValueError(f"Sample config missing 'catalog': {sample_path}")

    catalog_path = _resolve_ref(sample_data["catalog"], sample_path.parent, root_path)
    catalog_data = _load_stage_yaml(catalog_path)

    catalog_id = _required_id(catalog_data, catalog_path)
    sample_id = _required_id(sample_data, sample_path)
    analysis_id = _required_id(analysis_data, analysis_path)

    sample_data = dict(sample_data)
    sample_data["catalog"] = catalog_id
    # Expand declarative array specs (geomspace/linspace/logspace) into lists.
    analysis_data = dict(analysis_data)
    stack = dict(analysis_data.get("stack", {}))
    if "r_bin_edges" in stack:
        stack["r_bin_edges"] = parse_array_spec(stack["r_bin_edges"])
    analysis_data["stack"] = stack
    analysis_data["sample"] = sample_id

    catalog = StageConfig(
        kind="catalog",
        id=catalog_id,
        path=catalog_path,
        data=catalog_data,
        config_hash=stable_hash(catalog_data),
    )
    sample = StageConfig(
        kind="sample",
        id=sample_id,
        path=sample_path,
        data=sample_data,
        config_hash=stable_hash(sample_data),
    )
    analysis = StageConfig(
        kind="analysis",
        id=analysis_id,
        path=analysis_path,
        data=analysis_data,
        config_hash=stable_hash(hashable_analysis_data(analysis_data)),
    )
    return ResolvedConfig(
        root=root_path,
        analysis=analysis,
        sample=sample,
        catalog=catalog,
    )


def hashable_analysis_data(analysis_data: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the part of an analysis config that identifies its stack outputs.

    The ``postrun`` block configures analyses that run *after* a stack and read
    it back off disk, so it cannot change the stack itself. Hashing it would
    mean that retuning a fit parameter invalidates the manifest and demands a
    full re-stack to regenerate a text report, so it is excluded here.

    The early return is not an optimisation. It is the guarantee that adding
    this exclusion cannot invalidate a stack already on disk: configs without a
    ``postrun`` key hash through the identical code path they always did.
    """

    if "postrun" not in analysis_data:
        return analysis_data
    return {key: value for key, value in analysis_data.items() if key != "postrun"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping from disk."""
    return _load_stage_yaml(Path(path))


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write a YAML mapping to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(_normalize(data), handle, sort_keys=False)


def parse_array_spec(value: Any) -> list[float]:
    """Parse explicit lists or small NumPy-style array specs."""
    if isinstance(value, dict) and len(value) == 1:
        name, kwargs = next(iter(value.items()))
        if not isinstance(kwargs, dict):
            raise ValueError(f"Array spec '{name}' must contain keyword arguments")
        if name == "geomspace":
            arr = np.geomspace(kwargs["start"], kwargs["stop"], int(kwargs["num"]))
        elif name == "linspace":
            arr = np.linspace(kwargs["start"], kwargs["stop"], int(kwargs["num"]))
        elif name == "logspace":
            arr = np.logspace(kwargs["start"], kwargs["stop"], int(kwargs["num"]))
        else:
            raise ValueError(f"Unsupported array spec: {name}")
        return [float(x) for x in arr]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    raise ValueError(f"Expected an array list or supported array spec, got {value!r}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_path(path: str | Path, root: str | Path) -> str:
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _load_stage_yaml(path: Path, _chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load one stage YAML, resolving `extends` into a merged mapping."""
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")

    base_ref = data.pop("extends", None)
    if base_ref is None:
        return data

    # A variant that inherits its parent's id would silently overwrite the
    # parent's outputs, so make declaring a new one mandatory.
    if "id" not in data:
        raise ValueError(f"Config using 'extends' must declare its own 'id': {path}")

    base_path = _resolve_path(base_ref, path.parent)
    resolved = path.resolve()
    if base_path in _chain or base_path == resolved:
        cycle = " -> ".join(str(item) for item in (*_chain, resolved, base_path))
        raise ValueError(f"Circular 'extends' chain: {cycle}")
    if not base_path.exists():
        raise FileNotFoundError(
            f"'extends' target not found: {base_path} (from {path})"
        )

    base = _load_stage_yaml(base_path, (*_chain, resolved))
    # `extends` is consumed here rather than kept, so a merged config hashes
    # exactly like the equivalent standalone file and existing results stay
    # valid.
    return _deep_merge(base, data)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge `override` onto `base`, recursing into nested mappings.

    Only mappings merge. Scalars, lists, and strings are replaced wholesale,
    so an override cannot append to a list or edit part of a query string.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _resolve_path(path: str | Path, base: Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_ref(path: str | Path, local_base: Path, root: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    local = (local_base / path).resolve()
    if local.exists():
        return local
    return (root / path).resolve()


def _required_id(data: dict[str, Any], path: Path) -> str:
    if "id" not in data:
        raise ValueError(f"Config missing required 'id': {path}")
    return str(data["id"])


def _stage_to_dict(stage: StageConfig, root: Path) -> dict[str, Any]:
    return {
        "id": stage.id,
        "kind": stage.kind,
        "path": format_path(stage.path, root),
        "config_hash": stage.config_hash,
        "data": stage.data,
    }


def _normalize(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _normalize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
