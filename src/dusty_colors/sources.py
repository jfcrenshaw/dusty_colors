"""YAML-driven source table assembly for catalog preparation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[2]


def assemble_sources(config: Mapping[str, Any]) -> pd.DataFrame:
    """Load and join raw source tables described by catalog YAML."""
    sources = config.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("Catalog config must define a non-empty 'sources' mapping")

    primary_name = str(config.get("primary_source", next(iter(sources))))
    if primary_name not in sources:
        raise ValueError(f"primary_source '{primary_name}' is not in sources")

    assembled = load_source_table(sources[primary_name])
    for name, source_config in sources.items():
        if name == primary_name:
            continue
        if not isinstance(source_config, Mapping):
            raise ValueError(f"Source '{name}' must contain a YAML mapping")
        join_config = source_config.get("join")
        if not isinstance(join_config, Mapping):
            raise ValueError(f"Non-primary source '{name}' must define a join")
        assembled = join_source(
            assembled,
            load_source_table(
                source_config,
                ensure_columns=_join_right_columns(join_config),
            ),
            join_config,
            source_name=str(name),
        )
    return assembled


def load_source_table(
    config: Mapping[str, Any],
    *,
    ensure_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load one source table and apply local filters/renames/column pruning."""
    if not isinstance(config, Mapping):
        raise ValueError("Source config must contain a YAML mapping")

    file_configs = _source_file_configs(config)
    project_columns = "columns" in config or "optional_columns" in config
    required_columns = list(config.get("columns", []))
    optional_columns = list(config.get("optional_columns", []))
    for col in ensure_columns or []:
        if col not in required_columns:
            required_columns.append(col)

    table = _read_source_files(
        file_configs,
        required_columns=required_columns if project_columns else [],
        optional_columns=optional_columns if project_columns else [],
    )

    rename = dict(config.get("rename", {}))
    if rename:
        table = table.rename(columns=rename)

    if config.get("query"):
        table = table.query(str(config["query"])).reset_index(drop=True)

    finite = list(config.get("finite", []))
    if finite:
        mask = np.ones(len(table), dtype=bool)
        for col in finite:
            if col not in table:
                raise ValueError(f"Finite filter requested missing column: {col}")
            mask &= np.isfinite(table[col].to_numpy(float))
        table = table.loc[mask].reset_index(drop=True)

    drop_duplicates = config.get("drop_duplicates")
    if drop_duplicates:
        subset = None if drop_duplicates is True else list(drop_duplicates)
        table = table.drop_duplicates(subset=subset).reset_index(drop=True)

    if project_columns:
        missing = sorted(set(required_columns) - set(table.columns))
        if missing:
            raise ValueError(f"Source columns are missing: {missing}")
        output_columns = [
            col
            for col in [*required_columns, *optional_columns]
            if col in table.columns
        ]
        table = table[_unique(output_columns)].copy()
    elif ensure_columns:
        missing = sorted(set(ensure_columns) - set(table.columns))
        if missing:
            raise ValueError(f"Source columns are missing: {missing}")

    return table


def join_source(
    left: pd.DataFrame,
    right: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    source_name: str,
) -> pd.DataFrame:
    """Join one secondary source table onto the assembled primary table."""
    kwargs = _join_kwargs(config, source_name)
    right_key = kwargs.pop("_right_key", None)
    drop_right_key = kwargs.pop("_drop_right_key")
    joined = left.merge(right, **kwargs)
    if drop_right_key and right_key in joined and right_key not in left:
        joined = joined.drop(columns=[right_key])
    return joined


def read_table(
    path: str | Path,
    *,
    columns: list[str] | None = None,
    optional_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read a supported table format into pandas."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    required = list(columns or [])
    optional = list(optional_columns or [])
    projected = _unique([*required, *optional])
    if suffix == ".parquet":
        return _read_parquet(path, required, optional)
    if suffix in {".fits", ".fit", ".fz"}:
        return _read_fits(path, required, optional)
    if suffix == ".csv":
        return pd.read_csv(path, usecols=projected or None)
    if suffix in {".ecsv", ".ascii"}:
        table = Table.read(path)
        if projected:
            missing = sorted(set(required) - set(table.colnames))
            if missing:
                raise ValueError(f"Source columns are missing: {missing}")
            projected = [col for col in projected if col in table.colnames]
            table = table[projected]
        return table.to_pandas()
    raise ValueError(f"Unsupported source table format: {path}")


def _source_file_configs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "files" in config:
        files = config["files"]
    elif "path" in config:
        files = config["path"]
    else:
        raise ValueError("Source config missing required 'path' or 'files'")

    if not isinstance(files, (list, tuple)):
        files = [files]

    out = []
    for item in files:
        if isinstance(item, Mapping):
            if "path" not in item:
                raise ValueError("Source file mapping missing required 'path'")
            out.append(dict(item))
        else:
            out.append({"path": item})
    return out


def _read_source_files(
    file_configs: list[dict[str, Any]],
    *,
    required_columns: list[str],
    optional_columns: list[str],
) -> pd.DataFrame:
    injected_columns = {
        str(key)
        for file_config in file_configs
        for key in file_config
        if key != "path"
    }
    required_to_read = [
        col for col in required_columns if col not in injected_columns
    ]
    optional_to_read = [
        col for col in optional_columns if col not in injected_columns
    ]

    tables = []
    for file_config in file_configs:
        table = read_table(
            file_config["path"],
            columns=required_to_read,
            optional_columns=optional_to_read,
        )
        for key, value in file_config.items():
            if key == "path":
                continue
            table[str(key)] = value
        tables.append(table)

    if len(tables) == 1:
        return tables[0].reset_index(drop=True)
    return pd.concat(tables, ignore_index=True, sort=False)


def _read_parquet(
    path: Path,
    required_columns: list[str],
    optional_columns: list[str],
) -> pd.DataFrame:
    columns = _unique([*required_columns, *optional_columns])
    if not columns:
        return pd.read_parquet(path)
    try:
        return pd.read_parquet(path, columns=columns)
    except (KeyError, ValueError):
        if optional_columns:
            return pd.read_parquet(path, columns=required_columns)
        raise


def _read_fits(
    path: Path,
    required_columns: list[str],
    optional_columns: list[str],
) -> pd.DataFrame:
    columns = _available_fits_columns(path, required_columns, optional_columns)
    if not columns:
        return Table.read(path).to_pandas()
    with fits.open(path, memmap=True) as hdus:
        data = hdus[1].data
        table = Table({col: data[col] for col in columns}, copy=True)
    return table.to_pandas()


def _available_fits_columns(
    path: Path,
    required_columns: list[str],
    optional_columns: list[str],
) -> list[str]:
    requested = _unique([*required_columns, *optional_columns])
    if not requested:
        return []
    with fits.open(path, memmap=True) as hdus:
        names = list(hdus[1].columns.names)
    missing = sorted(set(required_columns) - set(names))
    if missing:
        raise ValueError(f"Source columns are missing in {path}: {missing}")
    available = set(names)
    return [col for col in requested if col in available]


def _unique(values: list[str]) -> list[str]:
    out = []
    seen = set()
    for value in values:
        value = str(value)
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out


def _join_kwargs(config: Mapping[str, Any], source_name: str) -> dict[str, Any]:
    how = str(config.get("how", "left"))
    suffixes = tuple(config.get("suffixes", ["", f"_{source_name}"]))
    kwargs: dict[str, Any] = {"how": how, "suffixes": suffixes}

    right_key: str | None = None
    if "on" in config:
        kwargs["on"] = config["on"]
        right_key = str(config["on"])
    else:
        left_key = config.get("left_key")
        right_key = config.get("right_key", left_key)
        if left_key is None or right_key is None:
            raise ValueError("Join config must define either 'on' or left/right keys")
        kwargs["left_on"] = left_key
        kwargs["right_on"] = right_key

    if "validate" in config:
        kwargs["validate"] = config["validate"]
    kwargs["_right_key"] = str(right_key)
    kwargs["_drop_right_key"] = bool(config.get("drop_right_key", True))
    return kwargs


def _join_right_columns(config: Mapping[str, Any]) -> list[str]:
    if "on" in config:
        return [str(config["on"])]
    right_key = config.get("right_key")
    return [] if right_key is None else [str(right_key)]
