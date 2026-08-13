from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dusty_colors_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/dusty_colors_cache")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.config import (  # noqa: E402
    load_resolved_config,
    load_yaml,
    parse_array_spec,
    stable_hash,
)
from dusty_colors.pipeline import (  # noqa: E402
    ForceOptions,
    ManifestMismatchError,
    StageHandlers,
    StageOutputError,
    _wrap_domain_handler,
    run_pipeline,
    run_post_run_only,
)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_graph(root: Path, *, foreground_z: list[float] | None = None) -> Path:
    _write_yaml(
        root / "configs/catalogs/dp1.yaml",
        {
            "id": "dp1_processed",
            "adapter": "rubin_dp1",
            "primary_source": "objects",
            "sources": {
                "objects": {"path": "data/dp1.parquet"},
            },
            "bands": ["g", "r", "i"],
            "photometry": "flux",
            "footprint": {"nside": 1},
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
        pd.DataFrame(
            {
                "ra": [10.0, 20.0],
                "dec": [0.0, 1.0],
                "z_phot": [0.3, 0.4],
                "pixel": [0, 1],
                "jackknife_region": [0, 1],
            }
        ).to_parquet(context.output_dir / "foreground.parquet", index=False)
        pd.DataFrame(
            {
                "ra": [11.0, 21.0],
                "dec": [0.0, 1.0],
                "pixel": [0, 1],
                "jackknife_region": [0, 1],
            }
        ).to_parquet(context.output_dir / "background.parquet", index=False)
        (context.output_dir / "sample_report.md").write_text("# sample\n")
        (context.output_dir / "sample_report.json").write_text("{}\n")

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
            self.assertTrue((stack_dir / "stack_fcolors.npz").exists())
            self.assertTrue((stack_dir / "stack_fcolors_provenance.npz").exists())
            self.assertTrue((stack_dir / "stack_fcolors_diagnostics.npz").exists())
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

    def test_only_postrun_regenerates_products_without_running_stages(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root)
            run_pipeline(analysis_path, root=root, handlers=_handlers())
            stack_dir = root / "results/stacks/analysis_default"
            all_colors_plot = stack_dir / "analysis_default_fcolors_all_colors.pdf"
            all_colors_plot.unlink()

            # No handlers at all: if any stage tried to run, this would fail.
            outputs = run_post_run_only(analysis_path, root=root)

            self.assertTrue(all_colors_plot.exists())
            self.assertIn(all_colors_plot.resolve(), {p.resolve() for p in outputs})

    def test_only_postrun_accepts_an_edited_postrun_block(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root)
            run_pipeline(analysis_path, root=root, handlers=_handlers())

            # Editing postrun must not invalidate the stack it reads.
            analysis = load_yaml(analysis_path)
            analysis["postrun"] = {"color_power_law_fit": {"radial_pivot_kpc": 30.0}}
            _write_yaml(analysis_path, analysis)

            outputs = run_post_run_only(analysis_path, root=root)
            self.assertTrue(outputs)

    def test_only_postrun_rejects_a_stale_stack(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root)
            run_pipeline(analysis_path, root=root, handlers=_handlers())

            # A stack setting, unlike a postrun setting, does invalidate.
            analysis = load_yaml(analysis_path)
            analysis["stack"]["reference_annulus"] = [3.0, 5.0]
            _write_yaml(analysis_path, analysis)

            with self.assertRaises(ManifestMismatchError):
                run_post_run_only(analysis_path, root=root)

    def test_only_postrun_reports_missing_stack_outputs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root)
            with self.assertRaises(StageOutputError):
                run_post_run_only(analysis_path, root=root)

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

    def test_checked_in_analysis_graphs_resolve(self) -> None:
        analysis_paths = sorted((ROOT / "configs/analyses").glob("*.yaml"))
        self.assertGreater(len(analysis_paths), 0)

        for analysis_path in analysis_paths:
            with self.subTest(path=analysis_path.relative_to(ROOT)):
                resolved = load_resolved_config(analysis_path, root=ROOT)

                self.assertTrue(resolved.analysis.id)
                self.assertTrue(resolved.sample.id)
                self.assertTrue(resolved.catalog.id)
                self.assertEqual(resolved.analysis.data["sample"], resolved.sample.id)
                self.assertEqual(resolved.sample.data["catalog"], resolved.catalog.id)

                self.assertIsInstance(resolved.catalog.data.get("sources"), dict)
                self.assertIsInstance(resolved.catalog.data.get("adapter"), str)
                self.assertIsInstance(resolved.sample.data.get("selection"), dict)

                stack = resolved.analysis.data.get("stack")
                self.assertIsInstance(stack, dict)
                self.assertIsInstance(stack.get("colors"), list)
                self.assertGreater(len(stack["colors"]), 0)
                if "r_bin_edges" in stack:
                    edges = np.asarray(stack["r_bin_edges"], dtype=float)
                    self.assertGreaterEqual(len(edges), 2)
                    self.assertTrue(np.isfinite(edges).all())
                    self.assertTrue(np.all(np.diff(edges) > 0))


class ConfigExtendsTest(unittest.TestCase):
    """Tests for the `extends` merge used by the sample config variants."""

    def _base(self, directory: Path) -> Path:
        path = directory / "base.yaml"
        _write_yaml(
            path,
            {
                "id": "base",
                "catalog": "cat.yaml",
                "selection": {"foreground_z": [0.2, 0.5], "query": "a > 1"},
                "cleaning": {"background": {"trend": {"enabled": True, "degree": 2}}},
            },
        )
        return path

    def test_merged_config_matches_an_equivalent_standalone_file(self) -> None:
        """The whole point: merging must not change the hash, so existing
        results stay valid when a config is rewritten to use `extends`."""
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._base(directory)

            variant = directory / "variant.yaml"
            _write_yaml(
                variant,
                {
                    "id": "variant",
                    "extends": "base.yaml",
                    "cleaning": {"background": {"trend": {"enabled": False}}},
                },
            )

            standalone = directory / "standalone.yaml"
            _write_yaml(
                standalone,
                {
                    "id": "variant",
                    "catalog": "cat.yaml",
                    "selection": {"foreground_z": [0.2, 0.5], "query": "a > 1"},
                    "cleaning": {
                        "background": {"trend": {"enabled": False, "degree": 2}}
                    },
                },
            )

            merged = load_yaml(variant)
            self.assertEqual(merged, load_yaml(standalone))
            self.assertEqual(stable_hash(merged), stable_hash(load_yaml(standalone)))
            self.assertNotIn("extends", merged)

    def test_nested_mappings_merge_but_lists_are_replaced(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._base(directory)
            variant = directory / "variant.yaml"
            _write_yaml(
                variant,
                {
                    "id": "variant",
                    "extends": "base.yaml",
                    "selection": {"foreground_z": [0.3, 0.6]},
                },
            )
            merged = load_yaml(variant)
            # The sibling key survives the merge...
            self.assertEqual(merged["selection"]["query"], "a > 1")
            # ...but the list is replaced wholesale, not merged element-wise.
            self.assertEqual(merged["selection"]["foreground_z"], [0.3, 0.6])

    def test_chained_extends_resolves(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._base(directory)
            middle = directory / "middle.yaml"
            _write_yaml(
                middle,
                {"id": "middle", "extends": "base.yaml", "selection": {"query": "b>2"}},
            )
            leaf = directory / "leaf.yaml"
            _write_yaml(
                leaf,
                {"id": "leaf", "extends": "middle.yaml", "catalog": "other.yaml"},
            )
            merged = load_yaml(leaf)
            self.assertEqual(merged["id"], "leaf")
            self.assertEqual(merged["catalog"], "other.yaml")
            self.assertEqual(merged["selection"]["query"], "b>2")
            self.assertEqual(merged["selection"]["foreground_z"], [0.2, 0.5])

    def test_missing_id_is_rejected(self) -> None:
        """Inheriting the parent's id would silently overwrite its outputs."""
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._base(directory)
            variant = directory / "variant.yaml"
            _write_yaml(variant, {"extends": "base.yaml"})
            with self.assertRaises(ValueError) as caught:
                load_yaml(variant)
            self.assertIn("must declare its own 'id'", str(caught.exception))

    def test_circular_extends_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            first = directory / "first.yaml"
            second = directory / "second.yaml"
            _write_yaml(first, {"id": "first", "extends": "second.yaml"})
            _write_yaml(second, {"id": "second", "extends": "first.yaml"})
            with self.assertRaises(ValueError) as caught:
                load_yaml(first)
            self.assertIn("Circular", str(caught.exception))

    def test_missing_target_is_reported_with_both_paths(self) -> None:
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            variant = directory / "variant.yaml"
            _write_yaml(variant, {"id": "variant", "extends": "nope.yaml"})
            with self.assertRaises(FileNotFoundError) as caught:
                load_yaml(variant)
            self.assertIn("nope.yaml", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
