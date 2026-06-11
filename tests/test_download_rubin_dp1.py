from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "download_rubin_dp1.py"

spec = importlib.util.spec_from_file_location("download_rubin_dp1", SCRIPT)
download_rubin_dp1 = importlib.util.module_from_spec(spec)
sys.modules["download_rubin_dp1"] = download_rubin_dp1
assert spec.loader is not None
spec.loader.exec_module(download_rubin_dp1)


class _Table:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self.frame


class _Client:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_product(self, product_id: str) -> _Table:
        self.calls.append(product_id)
        return _Table(pd.DataFrame({"objectId": [1, 2], "product": product_id}))


class DownloadRubinDP1Test(unittest.TestCase):
    def test_download_manifest_reuse_and_force(self) -> None:
        client = _Client()

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            results = download_rubin_dp1.download_all(
                ["objects"],
                out_dir,
                client=client,
            )

            self.assertEqual(len(results), 1)
            result = results[0]
            self.assertEqual(result.status, "downloaded")
            self.assertEqual(result.product, "objects")
            self.assertEqual(result.product_id, "80_dp1_gold_all_sitcomtn154")
            self.assertEqual(result.filename, "80_dp1_gold_all_sitcomtn154.parquet")
            self.assertEqual(result.rows, 2)
            self.assertEqual(client.calls, ["80_dp1_gold_all_sitcomtn154"])
            self.assertTrue((out_dir / result.filename).exists())
            self.assertEqual(len(pd.read_parquet(out_dir / result.filename)), 2)

            manifest = download_rubin_dp1.write_manifest(results, out_dir)
            manifest_data = json.loads(manifest.read_text())
            self.assertEqual(
                manifest_data["products"]["objects"],
                "80_dp1_gold_all_sitcomtn154",
            )
            self.assertEqual(manifest_data["files"][0]["rows"], 2)

            reused = download_rubin_dp1.download_all(
                ["objects"],
                out_dir,
            )
            self.assertEqual(reused[0].status, "reused")
            self.assertEqual(reused[0].rows, 2)
            self.assertEqual(client.calls, ["80_dp1_gold_all_sitcomtn154"])

            forced = download_rubin_dp1.download_all(
                ["objects"],
                out_dir,
                client=client,
                force=True,
            )
            self.assertEqual(forced[0].status, "downloaded")
            self.assertEqual(
                client.calls,
                [
                    "80_dp1_gold_all_sitcomtn154",
                    "80_dp1_gold_all_sitcomtn154",
                ],
            )

    def test_product_selection_uses_expected_product_id(self) -> None:
        client = _Client()

        with TemporaryDirectory() as tmp:
            results = download_rubin_dp1.download_all(
                ["photoz"],
                Path(tmp),
                client=client,
            )

            self.assertEqual(
                results[0].filename,
                "128_pz_table_dp1_all_gold_baseline_sitcomtn154.parquet",
            )
            self.assertEqual(
                client.calls,
                ["128_pz_table_dp1_all_gold_baseline_sitcomtn154"],
            )

    def test_unknown_product_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown Rubin DP1 product"):
                download_rubin_dp1.download_all(
                    ["missing"],
                    Path(tmp),
                    client=_Client(),
                )

    def test_missing_token_reports_actionable_message(self) -> None:
        with TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "missing-token.txt"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "PzServer token not found"):
                    download_rubin_dp1.load_token(token_path)

    def test_token_env_takes_precedence_over_token_file(self) -> None:
        with TemporaryDirectory() as tmp:
            token_path = Path(tmp) / "token.txt"
            token_path.write_text("file-token\n")
            with patch.dict(os.environ, {"PZSERVER_TOKEN": "env-token"}):
                token = download_rubin_dp1.load_token(token_path)
            self.assertEqual(token, "env-token")

    def test_failure_cleans_partial_file(self) -> None:
        class FailingClient:
            def get_product(self, product_id: str) -> object:
                raise RuntimeError(f"failed {product_id}")

        with TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                download_rubin_dp1.download_all(
                    ["objects"],
                    out_dir,
                    client=FailingClient(),
                )
            self.assertEqual(list(out_dir.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
