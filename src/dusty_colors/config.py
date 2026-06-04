"""YAML loading and resolution for the three-stage pipeline."""

from __future__ import annotations

import hashlib
import json
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
    analysis_data = _parse_analysis_arrays(dict(analysis_data))
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
        config_hash=stable_hash(analysis_data),
    )
    return ResolvedConfig(
        root=root_path,
        analysis=analysis,
        sample=sample,
        catalog=catalog,
    )


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


def _load_stage_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping: {path}")
    return data


def _parse_analysis_arrays(data: dict[str, Any]) -> dict[str, Any]:
    stack = dict(data.get("stack", {}))
    if "r_bin_edges" in stack:
        stack["r_bin_edges"] = parse_array_spec(stack["r_bin_edges"])
    data["stack"] = stack
    return data


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
