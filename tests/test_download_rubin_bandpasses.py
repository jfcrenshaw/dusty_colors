from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_rubin_bandpasses.py"

spec = importlib.util.spec_from_file_location("download_rubin_bandpasses", SCRIPT)
download_rubin_bandpasses = importlib.util.module_from_spec(spec)
sys.modules["download_rubin_bandpasses"] = download_rubin_bandpasses
assert spec.loader is not None
spec.loader.exec_module(download_rubin_bandpasses)


RAW_BANDPASS = b"""# LSST Throughputs files created from syseng_throughputs repo
# Version 1.9
# sha1 abc123
# Aerosols added to atmosphere
# Wavelen_cutoff_BLUE 300.10
# Wavelen_cutoff_RED 300.40
# Wavelength(nm)  Throughput(0-1)
300.0 0.0
300.1 0.1
300.2 0.2
300.3 0.3
300.4 0.4
"""


class _Response(io.BytesIO):
    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class DownloadRubinBandpassesTest(unittest.TestCase):
    def test_download_convert_manifest_reuse_and_force(self) -> None:
        calls: list[str] = []

        def opener(url: str) -> _Response:
            calls.append(url)
            return _Response(RAW_BANDPASS)

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            results = download_rubin_bandpasses.download_all(
                ["u"],
                out_dir,
                opener=opener,
            )

            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.filename, "rubin_bandpass_u_v1.9.1.dat")
            self.assertEqual(result.upstream_version, "1.9")
            self.assertEqual(result.upstream_sha1, "abc123")
            self.assertEqual(result.cutoff_blue_nm, 300.10)
            self.assertEqual(result.cutoff_red_nm, 300.40)
            self.assertEqual(result.rows, 3)
            self.assertEqual(
                calls,
                [
                    (
                        f"{download_rubin_bandpasses.BASE_URL}/v1.9.1/"
                        "baseline/total_u.dat"
                    )
                ],
            )

            self.assertEqual(
                (out_dir / result.filename).read_text(),
                (
                    "|   lambda |        pass |\n"
                    "|  3001.00 | 0.100000000 |\n"
                    "|  3002.00 | 0.200000000 |\n"
                    "|  3003.00 | 0.300000000 |\n"
                ),
            )

            manifest = download_rubin_bandpasses.write_manifest(results, out_dir)
            manifest_data = json.loads(manifest.read_text())
            self.assertEqual(manifest_data["tag"], "v1.9.1")
            self.assertEqual(manifest_data["files"][0]["filename"], result.filename)
            self.assertEqual(manifest_data["files"][0]["rows"], 3)

            reused = download_rubin_bandpasses.download_all(
                ["u"],
                out_dir,
                opener=opener,
            )
            self.assertEqual(reused[0].status, "reused")
            self.assertEqual(reused[0].upstream_sha1, "abc123")
            self.assertEqual(len(calls), 1)

            forced = download_rubin_bandpasses.download_all(
                ["u"],
                out_dir,
                force=True,
                opener=opener,
            )
            self.assertEqual(forced[0].status, "downloaded")
            self.assertEqual(len(calls), 2)

    def test_custom_tag_and_band_filter_control_filename_and_url(self) -> None:
        calls: list[str] = []

        def opener(url: str) -> _Response:
            calls.append(url)
            return _Response(RAW_BANDPASS)

        with TemporaryDirectory() as tmp:
            results = download_rubin_bandpasses.download_all(
                ["g"],
                Path(tmp),
                tag="v1.2.3",
                opener=opener,
            )

            self.assertEqual(results[0].filename, "rubin_bandpass_g_v1.2.3.dat")
            self.assertEqual(
                calls,
                [
                    (
                        f"{download_rubin_bandpasses.BASE_URL}/v1.2.3/"
                        "baseline/total_g.dat"
                    )
                ],
            )

    def test_unknown_band_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown Rubin band"):
                download_rubin_bandpasses.download_all(["q"], Path(tmp))

    def test_missing_metadata_fails_and_cleans_partial_file(self) -> None:
        bad_raw = RAW_BANDPASS.replace(b"# Wavelen_cutoff_RED 300.40\n", b"")

        def opener(url: str) -> _Response:
            return _Response(bad_raw)

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Wavelen_cutoff_RED"):
                download_rubin_bandpasses.download_all(
                    ["u"],
                    out_dir,
                    opener=opener,
                )
            self.assertEqual(list(out_dir.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
