"""Download Rubin/LSST bandpasses and convert them for kcorrect."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.request import urlopen

BASE_URL = "https://raw.githubusercontent.com/lsst/throughputs"
DEFAULT_TAG = "v1.9.1"
BANDS = ("u", "g", "r", "i", "z", "y")
KCORRECT_HEADER = "|   lambda |        pass |"


@dataclass(frozen=True)
class BandpassMetadata:
    upstream_version: str
    upstream_sha1: str
    cutoff_blue_nm: float
    cutoff_red_nm: float
    rows: int


@dataclass(frozen=True)
class ParsedBandpass:
    metadata: BandpassMetadata
    samples: list[tuple[float, float]]


@dataclass(frozen=True)
class DownloadResult:
    band: str
    filename: str
    tag: str
    url: str
    path: str
    status: str
    bytes: int
    sha256: str
    upstream_version: str | None
    upstream_sha1: str | None
    cutoff_blue_nm: float | None
    cutoff_red_nm: float | None
    rows: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/bandpasses",
        help="Directory for generated Rubin bandpass files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when a local copy exists.",
    )
    parser.add_argument(
        "--band",
        action="append",
        choices=BANDS,
        help="Download only one band; may be repeated.",
    )
    parser.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help="lsst/throughputs Git tag to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bands = args.band or list(BANDS)
    results = download_all(
        bands,
        Path(args.output_dir),
        tag=args.tag,
        force=args.force,
    )
    manifest = write_manifest(results, Path(args.output_dir), tag=args.tag)
    print(f"Wrote {manifest}")
    for result in results:
        print(f"{result.status}: {result.filename} ({result.bytes} bytes)")


def download_all(
    bands: list[str],
    output_dir: Path,
    *,
    tag: str = DEFAULT_TAG,
    force: bool = False,
    opener: Callable[..., object] = urlopen,
) -> list[DownloadResult]:
    validate_tag(tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = read_previous_manifest(output_dir)
    results = []
    for band in bands:
        if band not in BANDS:
            raise ValueError(f"Unknown Rubin band: {band}")
        results.append(
            download_bandpass(
                band,
                output_dir,
                tag=tag,
                force=force,
                opener=opener,
                previous=previous.get(output_filename(band, tag)),
            )
        )
    return results


def download_bandpass(
    band: str,
    output_dir: Path,
    *,
    tag: str = DEFAULT_TAG,
    force: bool = False,
    opener: Callable[..., object] = urlopen,
    previous: Mapping[str, Any] | None = None,
) -> DownloadResult:
    filename = output_filename(band, tag)
    path = output_dir / filename
    url = source_url(band, tag)
    if path.exists() and not force:
        return result_from_path(
            band,
            filename,
            tag,
            url,
            path,
            status="reused",
            previous=previous,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.part")
    try:
        with opener(url) as response:  # type: ignore[attr-defined]
            raw = response.read()
        parsed = parse_bandpass(raw)
        write_kcorrect_bandpass(tmp_path, parsed.samples)
        tmp_path.replace(path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return result_from_path(
        band,
        filename,
        tag,
        url,
        path,
        status="downloaded",
        metadata=parsed.metadata,
    )


def parse_bandpass(raw: bytes) -> ParsedBandpass:
    text = raw.decode("utf-8")
    version = None
    upstream_sha1 = None
    cutoff_blue = None
    cutoff_red = None
    rows: list[tuple[float, float]] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            parts = stripped.split()
            if len(parts) >= 3 and parts[1] == "Version":
                version = parts[2]
            elif len(parts) >= 3 and parts[1] == "sha1":
                upstream_sha1 = parts[2]
            elif len(parts) >= 3 and parts[1] == "Wavelen_cutoff_BLUE":
                cutoff_blue = float(parts[2])
            elif len(parts) >= 3 and parts[1] == "Wavelen_cutoff_RED":
                cutoff_red = float(parts[2])
            continue

        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            wavelength_nm = float(parts[0])
            throughput = float(parts[1])
        except ValueError:
            continue
        rows.append((wavelength_nm, throughput))

    missing = []
    if version is None:
        missing.append("Version")
    if upstream_sha1 is None:
        missing.append("sha1")
    if cutoff_blue is None:
        missing.append("Wavelen_cutoff_BLUE")
    if cutoff_red is None:
        missing.append("Wavelen_cutoff_RED")
    if missing:
        raise ValueError(f"Rubin bandpass metadata missing: {missing}")

    samples = [
        (wavelength_nm, throughput)
        for wavelength_nm, throughput in rows
        if cutoff_blue <= wavelength_nm < cutoff_red
    ]
    if not samples:
        raise ValueError("Rubin bandpass has no samples within cutoff range")

    metadata = BandpassMetadata(
        upstream_version=version,
        upstream_sha1=upstream_sha1,
        cutoff_blue_nm=cutoff_blue,
        cutoff_red_nm=cutoff_red,
        rows=len(samples),
    )
    return ParsedBandpass(metadata=metadata, samples=samples)


def write_kcorrect_bandpass(path: Path, samples: list[tuple[float, float]]) -> None:
    lines = [KCORRECT_HEADER]
    for wavelength_nm, throughput in samples:
        lines.append(f"| {wavelength_nm * 10.0:8.2f} | {throughput:.9f} |")
    path.write_text("\n".join(lines) + "\n")


def write_manifest(
    results: list[DownloadResult],
    output_dir: Path,
    *,
    tag: str = DEFAULT_TAG,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_url": BASE_URL,
        "tag": tag,
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
    band: str,
    filename: str,
    tag: str,
    url: str,
    path: Path,
    *,
    status: str,
    metadata: BandpassMetadata | None = None,
    previous: Mapping[str, Any] | None = None,
) -> DownloadResult:
    return DownloadResult(
        band=band,
        filename=filename,
        tag=tag,
        url=url,
        path=str(path),
        status=status,
        bytes=path.stat().st_size,
        sha256=sha256(path),
        upstream_version=metadata_value("upstream_version", metadata, previous),
        upstream_sha1=metadata_value("upstream_sha1", metadata, previous),
        cutoff_blue_nm=metadata_value("cutoff_blue_nm", metadata, previous),
        cutoff_red_nm=metadata_value("cutoff_red_nm", metadata, previous),
        rows=metadata_value("rows", metadata, previous),
    )


def metadata_value(
    key: str,
    metadata: BandpassMetadata | None,
    previous: Mapping[str, Any] | None,
) -> Any:
    if metadata is not None:
        return getattr(metadata, key)
    if previous is not None:
        return previous.get(key)
    return None


def source_url(band: str, tag: str) -> str:
    return f"{BASE_URL}/{tag}/baseline/total_{band}.dat"


def output_filename(band: str, tag: str) -> str:
    return f"rubin_bandpass_{band}_{tag}.dat"


def validate_tag(tag: str) -> None:
    if "/" in tag or "\\" in tag:
        raise ValueError("Rubin throughputs tag must not contain path separators")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
