from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_clauds.py"

spec = importlib.util.spec_from_file_location("download_clauds", SCRIPT)
download_clauds = importlib.util.module_from_spec(spec)
sys.modules["download_clauds"] = download_clauds
assert spec.loader is not None
spec.loader.exec_module(download_clauds)


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class DownloadClaudsTest(unittest.TestCase):
    def test_url_map_and_reuse(self) -> None:
        calls: list[str] = []

        def opener(url: str) -> _Response:
            calls.append(url)
            return _Response(b"header")

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            results = download_clauds.download_all(
                ["header"],
                out_dir,
                opener=opener,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].status, "downloaded")
            self.assertEqual(
                calls,
                [
                    (
                        f"{download_clauds.BASE_URL}/"
                        "header_sextractor_Picouet.txt"
                    )
                ],
            )

            reused = download_clauds.download_all(
                ["header"],
                out_dir,
                opener=opener,
            )
            self.assertEqual(reused[0].status, "reused")
            self.assertEqual(len(calls), 1)

            manifest = download_clauds.write_manifest(reused, out_dir)
            self.assertTrue(manifest.exists())
            self.assertIn("download_manifest.json", str(manifest))

    def test_401_fails_with_cadc_auth_message(self) -> None:
        def opener(url: str) -> _Response:
            raise HTTPError(url, 401, "Unauthorized", {}, None)

        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "CADC returned 401"):
                download_clauds.download_all(
                    ["e_cosmos"],
                    Path(tmp),
                    force=True,
                    opener=opener,
                )


if __name__ == "__main__":
    unittest.main()
