from __future__ import annotations

from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dusty_colors_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/dusty_colors_cache")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.config import (  # noqa: E402
    load_resolved_config,
    parse_array_spec,
    stable_hash,
)
from dusty_colors.pipeline import (  # noqa: E402
    ForceOptions,
    ManifestMismatchError,
    StageHandlers,
    _wrap_domain_handler,
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
                "colors": ["g-i", "g-r"],
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
            np.savez_compressed(path, **_stack_arrays())

    return StageHandlers(catalog=catalog, sample=sample, stack=stack)


def _stack_arrays() -> dict[str, np.ndarray]:
    radius = np.geomspace(5.0, 100.0, 4)
    arrays = {}
    for index, color in enumerate(("g-i", "g-r")):
        signal = np.array([0.08, 0.03, 0.012, 0.006]) * (index + 1)
        err = signal * 0.2
        samples = np.vstack((signal * 0.9, signal, signal * 1.1))
        arrays.update(
            {
                f"{color}_bin_centers": radius,
                f"{color}_avg": signal,
                f"{color}_err": err,
                f"{color}_jackknife_avg": samples.mean(axis=0),
                f"{color}_jackknife_err": err,
                f"{color}_jackknife_samples": samples,
            }
        )
    return arrays


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
            stack_dir = root / "results/stacks/analysis_default"
            jackknife_plot = stack_dir / "analysis_default_fcolors_g_i_jackknife.pdf"
            all_colors_plot = stack_dir / "analysis_default_fcolors_all_colors.pdf"
            self.assertTrue(jackknife_plot.exists())
            self.assertTrue(all_colors_plot.exists())

            jackknife_plot.unlink()
            all_colors_plot.unlink()

            run_again = run_pipeline(analysis_path, root=root, handlers=StageHandlers())
            self.assertEqual([stage.action for stage in run_again.stages], ["skip"] * 3)
            self.assertTrue(jackknife_plot.exists())
            self.assertTrue(all_colors_plot.exists())

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

    def test_stack_wrapper_prefers_sample_footprint_when_present(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample_dir = root / "sample"
            catalog_dir = root / "catalog"
            output_dir = root / "stack"
            sample_dir.mkdir()
            catalog_dir.mkdir()
            (sample_dir / "footprint.parquet").write_bytes(b"sample footprint")
            (catalog_dir / "footprint.parquet").write_bytes(b"catalog footprint")
            called = {}

            def stack_handler(
                sample_path,
                output_path,
                stack_config,
                *,
                footprint_path,
                force,
            ) -> None:
                called["sample_path"] = sample_path
                called["output_path"] = output_path
                called["stack_config"] = stack_config
                called["footprint_path"] = footprint_path
                called["force"] = force

            context = SimpleNamespace(
                input_dirs={"sample": sample_dir, "catalog": catalog_dir},
                output_dir=output_dir,
                config=SimpleNamespace(data={"stack": {"random_seed": 11}}),
                force=True,
            )

            _wrap_domain_handler("stack", stack_handler)(context)

            self.assertEqual(called["footprint_path"], sample_dir / "footprint.parquet")
            self.assertEqual(called["sample_path"], sample_dir)
            self.assertEqual(called["output_path"], output_dir)
            self.assertEqual(called["stack_config"], {"random_seed": 11})
            self.assertTrue(called["force"])


if __name__ == "__main__":
    unittest.main()
