from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.analysis_stats import save_analysis_catalog_stats, stats_for_analysis


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_graph(root: Path, *, jackknife: bool = True) -> Path:
    _write_yaml(
        root / "configs/catalogs/test.yaml",
        {
            "id": "test_catalog",
            "adapter": "rubin_dp1",
            "primary_source": "objects",
            "sources": {"objects": {"path": "data/raw.parquet"}},
            "footprint": {"nside": 1},
        },
    )
    _write_yaml(
        root / "configs/samples/test.yaml",
        {
            "id": "test_sample",
            "catalog": "configs/catalogs/test.yaml",
            "selection": {
                "foreground_z": [0.2, 0.5],
                "background_z": [0.7, 1.4],
            },
        },
    )
    analysis_path = root / "configs/analyses/test.yaml"
    _write_yaml(
        analysis_path,
        {
            "id": "test_analysis",
            "sample": "configs/samples/test.yaml",
            "stack": {"jackknife": jackknife},
        },
    )
    return analysis_path


def _write_samples(root: Path) -> None:
    sample_dir = root / "results/samples/test_sample"
    sample_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4],
            "ra": [10.0, 20.0, np.nan, 40.0],
            "dec": [0.0, 0.0, 0.0, 0.0],
            "z_phot": [0.3, 0.4, 0.35, 0.45],
            "pixel": [0, 1, 2, 3],
            "jackknife_region": [0, 1, 2, 3],
        }
    ).to_parquet(sample_dir / "foreground.parquet", index=False)
    pd.DataFrame(
        {
            "object_id": [5, 6, 7, 8],
            "ra": [11.0, 21.0, 51.0, 31.0],
            "dec": [0.0, 0.0, 0.0, np.nan],
            "pixel": [0, 1, 4, 3],
            "jackknife_region": [0, 1, 4, 3],
        }
    ).to_parquet(sample_dir / "background.parquet", index=False)


class AnalysisCatalogStatsTest(unittest.TestCase):
    def test_stats_use_stack_final_position_and_jackknife_filters(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root)
            _write_samples(root)

            stats = stats_for_analysis(
                analysis_path,
                root=root,
                require_current=False,
            )

            pixel_area = 4.0 * np.pi * (180.0 / np.pi) ** 2 / 12.0
            self.assertEqual(stats.foreground_galaxies, 2)
            self.assertEqual(stats.background_galaxies, 2)
            self.assertEqual(stats.occupied_pixels, 2)
            self.assertAlmostEqual(stats.area_deg2, 2 * pixel_area)
            self.assertAlmostEqual(stats.foreground_density_deg2, 2 / stats.area_deg2)
            self.assertAlmostEqual(stats.background_density_deg2, 2 / stats.area_deg2)
            self.assertEqual(stats.dropped_foreground_bad_position, 1)
            self.assertEqual(stats.dropped_background_bad_position, 1)
            self.assertEqual(stats.dropped_foreground_jackknife, 1)
            self.assertEqual(stats.dropped_background_jackknife, 1)

    def test_stats_can_skip_jackknife_filtering_when_analysis_does(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root, jackknife=False)
            _write_samples(root)

            stats = stats_for_analysis(
                analysis_path,
                root=root,
                require_current=False,
            )

            self.assertEqual(stats.foreground_galaxies, 3)
            self.assertEqual(stats.background_galaxies, 3)
            self.assertEqual(stats.occupied_pixels, 4)
            self.assertEqual(stats.dropped_foreground_jackknife, 0)
            self.assertEqual(stats.dropped_background_jackknife, 0)

    def test_save_analysis_catalog_stats_writes_text_and_json_reports(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root)
            _write_samples(root)

            sample_dir = root / "results/samples/test_sample"
            text_path, json_path = save_analysis_catalog_stats(
                analysis_path,
                sample_dir,
                root=root,
                require_current=False,
            )

            self.assertTrue(text_path.exists())
            self.assertTrue(json_path.exists())
            self.assertEqual(text_path.parent, sample_dir)
            self.assertEqual(json_path.parent, sample_dir)
            self.assertIn("foreground_galaxies: 2", text_path.read_text())
            data = yaml.safe_load(json_path.read_text())
            self.assertEqual(data["analysis_id"], "test_analysis")
            self.assertEqual(data["foreground_galaxies"], 2)
            self.assertEqual(data["background_galaxies"], 2)


if __name__ == "__main__":
    unittest.main()
