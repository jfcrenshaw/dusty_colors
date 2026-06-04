"""YAML-driven source table assembly for catalog preparation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
    if "path" not in config:
        raise ValueError("Source config missing required 'path'")

    table = read_table(config["path"])
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

    columns = list(config.get("columns", []))
    if columns:
        for col in ensure_columns or []:
            if col not in columns:
                columns.append(col)
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise ValueError(f"Source columns are missing: {missing}")
        table = table[columns].copy()

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


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a supported table format into pandas."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".fits", ".fit", ".fz"}:
        return Table.read(path).to_pandas()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".ecsv", ".ascii"}:
        return Table.read(path).to_pandas()
    raise ValueError(f"Unsupported source table format: {path}")


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
