from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.config import load_resolved_config, parse_array_spec, stable_hash
from dusty_colors.pipeline import (
    ForceOptions,
    ManifestMismatchError,
    StageHandlers,
    run_pipeline,
)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_graph(root: Path, *, foreground_z: list[float] | None = None) -> Path:
    _write_yaml(
        root / "configs/catalogs/dp1.yaml",
        {
            "id": "dp1_processed",
            "adapter": "dp1",
            "primary_source": "objects",
            "sources": {
                "objects": {"path": "data/dp1.parquet"},
            },
            "bands": ["g", "r", "i"],
            "photometry": "flux",
        },
    )
    _write_yaml(
        root / "configs/samples/dp1_default.yaml",
        {
            "id": "dp1_default",
            "catalog": "configs/catalogs/dp1.yaml",
            "selection": {
                "foreground_z": foreground_z or [0.2, 0.5],
                "background_z": [0.7, 1.4],
            },
        },
    )
    analysis_path = root / "configs/analyses/default.yaml"
    _write_yaml(
        analysis_path,
        {
            "id": "analysis_default",
            "sample": "configs/samples/dp1_default.yaml",
            "stack": {
                "modes": ["fcolors"],
                "r_bin_edges": {"linspace": {"start": 0.0, "stop": 1.0, "num": 3}},
                "reference_annulus": [2.0, 4.0],
            },
        },
    )
    return analysis_path


def _handlers() -> StageHandlers:
    def catalog(context) -> None:
        (context.output_dir / "catalog.parquet").write_bytes(b"catalog")
        (context.output_dir / "footprint.parquet").write_bytes(b"footprint")

    def sample(context) -> None:
        (context.output_dir / "foreground.parquet").write_bytes(b"foreground")
        (context.output_dir / "background.parquet").write_bytes(b"background")

    def stack(context) -> None:
        for path in context.expected_outputs:
            path.write_bytes(b"stack")

    return StageHandlers(catalog=catalog, sample=sample, stack=stack)


class ConfigPipelineTest(unittest.TestCase):
    def test_array_specs(self) -> None:
        self.assertEqual(parse_array_spec([1, 2, 3]), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            parse_array_spec({"linspace": {"start": 1, "stop": 3, "num": 3}}),
            [1.0, 2.0, 3.0],
        )
        np.testing.assert_allclose(
            parse_array_spec({"geomspace": {"start": 1, "stop": 100, "num": 3}}),
            [1.0, 10.0, 100.0],
        )
        np.testing.assert_allclose(
            parse_array_spec({"logspace": {"start": 0, "stop": 2, "num": 3}}),
            [1.0, 10.0, 100.0],
        )

    def test_resolved_config_hash_is_stable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root)

            resolved = load_resolved_config(analysis_path, root=root)

            self.assertEqual(resolved.analysis.data["sample"], "dp1_default")
            self.assertEqual(resolved.sample.data["catalog"], "dp1_processed")
            self.assertEqual(
                resolved.analysis.config_hash,
                stable_hash(resolved.analysis.data),
            )
            np.testing.assert_allclose(
                resolved.analysis.data["stack"]["r_bin_edges"],
                [0.0, 0.5, 1.0],
            )

    def test_manifest_mismatch_requires_force(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root)
            run_pipeline(analysis_path, root=root, handlers=_handlers())

            run_again = run_pipeline(analysis_path, root=root, handlers=StageHandlers())
            self.assertEqual([stage.action for stage in run_again.stages], ["skip"] * 3)

            _write_graph(root, foreground_z=[0.25, 0.55])
            with self.assertRaises(ManifestMismatchError):
                run_pipeline(analysis_path, root=root, handlers=_handlers())

            forced = run_pipeline(
                analysis_path,
                root=root,
                force=ForceOptions(sample=True),
                handlers=_handlers(),
            )
            self.assertEqual(
                [stage.action for stage in forced.stages],
                ["skip", "run", "run"],
            )


if __name__ == "__main__":
    unittest.main()
