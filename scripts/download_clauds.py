"""Download CLAUDS-HSC SourceExtractor catalog products."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Callable
from urllib.error import HTTPError
from urllib.request import urlopen

BASE_URL = (
    "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/files/vault/clauds/"
    "desprez/PublicRelease"
)

SOURCE_EXTRACTOR_FILES: dict[str, str] = {
    "header": "header_sextractor_Picouet.txt",
    "e_cosmos": "COSMOS_v10_v210913_withGALEX.fits",
    "xmm_lss": "XMM_LSS_v10_v210913_withGALEX.fits",
    "elais_n1": "ELAIS-N1_6bands-SExtractor-Lephare.fits",
    "deep2_3": "DEEP2-3_6bands-SExtractor-Lephare.fits",
}

CADC_AUTH_MESSAGE = (
    "CADC returned 401 Unauthorized. Log in to CADC/CANFAR or configure CADC "
    "credentials, then rerun this script. Already-downloaded local files are "
    "accepted and will not be fetched again unless --force is used."
)


@dataclass(frozen=True)
class DownloadResult:
    key: str
    filename: str
    url: str
    path: str
    status: str
    bytes: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data/clauds",
        help="Directory for downloaded CLAUDS files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when a local copy exists.",
    )
    parser.add_argument(
        "--file",
        action="append",
        choices=sorted(SOURCE_EXTRACTOR_FILES),
        help="Download only one named product; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = args.file or list(SOURCE_EXTRACTOR_FILES)
    results = download_all(keys, Path(args.output_dir), force=args.force)
    manifest = write_manifest(results, Path(args.output_dir))
    print(f"Wrote {manifest}")
    for result in results:
        print(f"{result.status}: {result.filename} ({result.bytes} bytes)")


def download_all(
    keys: list[str],
    output_dir: Path,
    *,
    force: bool = False,
    opener: Callable[..., object] = urlopen,
) -> list[DownloadResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for key in keys:
        if key not in SOURCE_EXTRACTOR_FILES:
            raise ValueError(f"Unknown CLAUDS file key: {key}")
        filename = SOURCE_EXTRACTOR_FILES[key]
        results.append(
            download_file(
                key,
                filename,
                output_dir / filename,
                force=force,
                opener=opener,
            )
        )
    return results


def download_file(
    key: str,
    filename: str,
    path: Path,
    *,
    force: bool = False,
    opener: Callable[..., object] = urlopen,
) -> DownloadResult:
    url = f"{BASE_URL}/{filename}"
    if path.exists() and not force:
        return _result(key, filename, url, path, status="reused")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.part")
    try:
        with opener(url) as response, tmp_path.open("wb") as handle:  # type: ignore[attr-defined]
            shutil.copyfileobj(response, handle)
    except HTTPError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        if exc.code == 401:
            raise RuntimeError(CADC_AUTH_MESSAGE) from exc
        raise
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    tmp_path.replace(path)
    return _result(key, filename, url, path, status="downloaded")


def write_manifest(results: list[DownloadResult], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "base_url": BASE_URL,
        "files": [asdict(result) for result in results],
    }
    path = output_dir / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def _result(
    key: str,
    filename: str,
    url: str,
    path: Path,
    *,
    status: str,
) -> DownloadResult:
    return DownloadResult(
        key=key,
        filename=filename,
        url=url,
        path=str(path),
        status=status,
        bytes=path.stat().st_size,
        sha256=_sha256(path),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
