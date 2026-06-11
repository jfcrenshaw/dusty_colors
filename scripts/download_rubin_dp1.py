"""Download Rubin DP1 PzServer catalog products."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

PRODUCTS: dict[str, str] = {
    "objects": "80_dp1_gold_all_sitcomtn154",
    "training": "81_training_v1_match_prelim_sitcomtn154",
    "test": "82_test_v1_match_prelim_sitcomtn154",
    "photoz": "128_pz_table_dp1_all_gold_baseline_sitcomtn154",
}
DEFAULT_OUTPUT_DIR = "data/rubin_dp1"
DEFAULT_TOKEN_FILE = "pzserver_token.txt"
DEFAULT_TOKEN_ENV = "PZSERVER_TOKEN"


@dataclass(frozen=True)
class DownloadResult:
    product: str
    product_id: str
    filename: str
    path: str
    status: str
    bytes: int
    sha256: str
    rows: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for downloaded Rubin DP1 parquet files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload products even when local parquet files exist.",
    )
    parser.add_argument(
        "--product",
        action="append",
        choices=sorted(PRODUCTS),
        help="Download only one named product; may be repeated.",
    )
    parser.add_argument(
        "--token-file",
        default=DEFAULT_TOKEN_FILE,
        help="Path to a file containing a PzServer token.",
    )
    parser.add_argument(
        "--token-env",
        default=DEFAULT_TOKEN_ENV,
        help="Environment variable containing a PzServer token.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = args.product or list(PRODUCTS)
    output_dir = Path(args.output_dir)
    client = None
    if needs_client(keys, output_dir, force=args.force):
        token = load_token(Path(args.token_file), token_env=args.token_env)
        client = pzserver_client(token)
    results = download_all(
        keys,
        output_dir,
        client=client,
        force=args.force,
    )
    manifest = write_manifest(results, output_dir)
    print(f"Wrote {manifest}")
    for result in results:
        print(f"{result.status}: {result.filename} ({result.bytes} bytes)")


def download_all(
    keys: list[str],
    output_dir: Path,
    *,
    client: Any | None = None,
    force: bool = False,
) -> list[DownloadResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = read_previous_manifest(output_dir)
    results = []
    for key in keys:
        if key not in PRODUCTS:
            raise ValueError(f"Unknown Rubin DP1 product: {key}")
        product_id = PRODUCTS[key]
        filename = f"{product_id}.parquet"
        results.append(
            download_product(
                key,
                product_id,
                output_dir / filename,
                client=client,
                force=force,
                previous=previous.get(filename),
            )
        )
    return results


def download_product(
    key: str,
    product_id: str,
    path: Path,
    *,
    client: Any | None = None,
    force: bool = False,
    previous: Mapping[str, Any] | None = None,
) -> DownloadResult:
    if path.exists() and not force:
        return result_from_path(
            key,
            product_id,
            path,
            status="reused",
            previous=previous,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    if client is None:
        raise RuntimeError("Rubin DP1 download requires a PzServer client")
    tmp_path = path.with_name(f"{path.name}.part")
    try:
        table = client.get_product(product_id)
        frame = product_to_dataframe(table)
        frame.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return result_from_path(
        key,
        product_id,
        path,
        status="downloaded",
        rows=len(frame),
    )


def product_to_dataframe(product: Any) -> pd.DataFrame:
    if isinstance(product, pd.DataFrame):
        return product
    if hasattr(product, "to_pandas"):
        frame = product.to_pandas()
        if isinstance(frame, pd.DataFrame):
            return frame
    raise TypeError("PzServer product must be a pandas DataFrame or expose to_pandas()")


def needs_client(keys: list[str], output_dir: Path, *, force: bool = False) -> bool:
    if force:
        return True
    for key in keys:
        if key not in PRODUCTS:
            raise ValueError(f"Unknown Rubin DP1 product: {key}")
        if not (output_dir / f"{PRODUCTS[key]}.parquet").exists():
            return True
    return False


def load_token(token_file: Path, *, token_env: str = DEFAULT_TOKEN_ENV) -> str:
    env_token = os.environ.get(token_env)
    if env_token:
        return env_token.strip()

    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token

    raise RuntimeError(
        f"PzServer token not found. Set ${token_env} or create {token_file}."
    )


def pzserver_client(token: str) -> Any:
    try:
        from pzserver import PzServer
    except ImportError as exc:
        raise ImportError("Rubin DP1 downloads require the 'pzserver' package") from exc
    return PzServer(token=token)


def write_manifest(results: list[DownloadResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "products": PRODUCTS,
        "files": [asdict(result) for result in results],
    }
    path = output_dir / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def read_previous_manifest(output_dir: Path) -> dict[str, Mapping[str, Any]]:
    path = output_dir / "download_manifest.json"
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    files = manifest.get("files", [])
    if not isinstance(files, list):
        return {}
    previous = {}
    for item in files:
        if isinstance(item, Mapping) and isinstance(item.get("filename"), str):
            previous[str(item["filename"])] = item
    return previous


def result_from_path(
    key: str,
    product_id: str,
    path: Path,
    *,
    status: str,
    rows: int | None = None,
    previous: Mapping[str, Any] | None = None,
) -> DownloadResult:
    if rows is None and previous is not None:
        previous_rows = previous.get("rows")
        if previous_rows is None or isinstance(previous_rows, int):
            rows = previous_rows
    if rows is None:
        rows = parquet_row_count(path)
    return DownloadResult(
        product=key,
        product_id=product_id,
        filename=path.name,
        path=str(path),
        status=status,
        bytes=path.stat().st_size,
        sha256=sha256(path),
        rows=rows,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_row_count(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    return pq.ParquetFile(path).metadata.num_rows


if __name__ == "__main__":
    main()
