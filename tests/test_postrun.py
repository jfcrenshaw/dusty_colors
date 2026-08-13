"""Tests for the post-run analysis registry and its shared context."""

from __future__ import annotations

import os
import sys
import unittest
import unittest.mock
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dusty_colors_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/dusty_colors_cache")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.config import (  # noqa: E402
    hashable_analysis_data,
    load_resolved_config,
    stable_hash,
)
from dusty_colors.postrun import registered_analyses  # noqa: E402
from dusty_colors.postrun.base import (  # noqa: E402
    PostRunAnalysis,
    PostRunContext,
    run_post_run_analyses,
)

COLORS = ("g-r", "r-i", "i-z")


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_graph(root: Path, analysis_extra: dict | None = None) -> Path:
    _write_yaml(root / "configs/catalogs/dp1.yaml", {"id": "dp1"})
    _write_yaml(
        root / "configs/samples/dp1_default.yaml",
        {
            "id": "dp1_default",
            "catalog": "configs/catalogs/dp1.yaml",
            "selection": {"foreground_z": [0.2, 0.4]},
        },
    )
    analysis: dict = {
        "id": "dp1_default",
        "sample": "configs/samples/dp1_default.yaml",
        "stack": {
            "colors": list(COLORS),
            "modes": ["fcolors"],
            "r_bin_edges": [10.0, 15.0, 40.0, 120.0, 1000.0],
        },
    }
    analysis.update(analysis_extra or {})
    analysis_path = root / "configs/analyses/dp1_default.yaml"
    _write_yaml(analysis_path, analysis)
    return analysis_path


def _stack_arrays() -> dict:
    """Build a stack whose colors follow a declining power law.

    The amplitudes decrease from g-r to i-z so the band-relative excesses have
    the sign structure a real reddening signal would.
    """

    radius = np.array([12.0, 25.0, 70.0, 300.0])
    arrays: dict = {}
    for index, color in enumerate(COLORS):
        signal = 0.04 * (0.7**index) * (radius / 20.0) ** -1.0
        err = signal * 0.15
        samples = np.vstack([signal * 0.92, signal, signal * 1.08])
        arrays.update(
            {
                f"{color}_bin_centers": radius,
                f"{color}_avg": signal,
                f"{color}_err": err,
                f"{color}_analytic_err": err,
                f"{color}_cov": np.diag(err**2),
                f"{color}_jackknife_avg": samples.mean(axis=0),
                f"{color}_jackknife_err": err,
                f"{color}_jackknife_samples": samples,
            }
        )
    return arrays


def _build_context(root: Path, analysis_extra: dict | None = None) -> PostRunContext:
    analysis_path = _write_graph(root, analysis_extra)
    stack_dir = root / "results/stacks/dp1_default"
    stack_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stack_dir / "stack_fcolors.npz", **_stack_arrays())

    resolved = load_resolved_config(analysis_path, root=root)
    return PostRunContext(
        resolved=resolved,
        stack_dir=stack_dir,
        sample_dir=root / "results/samples/dp1_default",
        catalog_dir=root / "results/catalogs/dp1",
        modes=("fcolors",),
    )


class RegistryTest(unittest.TestCase):
    def test_expected_analyses_are_registered_in_order(self) -> None:
        names = [analysis.name for analysis in registered_analyses()]
        self.assertEqual(
            names,
            [
                "analysis_catalog_stats",
                "stack_figures",
                "diagnostic_figures",
                "dust_extinction_fit",
                "color_power_law_fit",
                "chromaticity",
            ],
        )

    def test_only_catalog_stats_is_sample_level(self) -> None:
        per_mode = {a.name: a.per_mode for a in registered_analyses()}
        self.assertFalse(per_mode["analysis_catalog_stats"])
        self.assertTrue(
            all(v for k, v in per_mode.items() if k != "analysis_catalog_stats")
        )


class ContextOptionsTest(unittest.TestCase):
    def test_postrun_block_takes_precedence_over_stack(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(
                root,
                {
                    "postrun": {"dust_extinction_fit": {"fixed_rv": 2.5}},
                    "stack": {
                        "colors": list(COLORS),
                        "modes": ["fcolors"],
                        "dust_extinction_fit": {"fixed_rv": 3.1},
                    },
                },
            )
            self.assertEqual(context.options("dust_extinction_fit"), {"fixed_rv": 2.5})

    def test_legacy_stack_block_is_still_read(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(
                root,
                {
                    "stack": {
                        "colors": list(COLORS),
                        "modes": ["fcolors"],
                        "dust_extinction_fit": {"fixed_rv": 3.1},
                    }
                },
            )
            self.assertEqual(context.options("dust_extinction_fit"), {"fixed_rv": 3.1})

    def test_boolean_and_absent_values_normalise_to_empty_mapping(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(
                root,
                {"postrun": {"stack_figures": True, "chromaticity": None}},
            )
            self.assertEqual(context.options("stack_figures"), {})
            self.assertEqual(context.options("chromaticity"), {})
            self.assertEqual(context.options("never_configured"), {})

    def test_enabled_honours_both_disable_spellings(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(
                root,
                {
                    "postrun": {
                        "stack_figures": False,
                        "chromaticity": {"enabled": False},
                        "color_power_law_fit": {"radial_pivot_kpc": 20.0},
                    }
                },
            )
            self.assertFalse(context.enabled("stack_figures"))
            self.assertFalse(context.enabled("chromaticity"))
            self.assertTrue(context.enabled("color_power_law_fit"))
            self.assertTrue(context.enabled("unmentioned"))
            self.assertFalse(context.enabled("unmentioned", default=False))

    def test_non_mapping_option_block_is_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(root, {"postrun": {"stack_figures": ["oops"]}})
            with self.assertRaises(ValueError):
                context.options("stack_figures")


class ContextCacheTest(unittest.TestCase):
    def test_stack_results_are_loaded_once_per_mode(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(root)
            first = context.results("fcolors")
            second = context.results("fcolors")
            # Identity, not equality: repeated stages must not re-read the npz.
            self.assertIs(first, second)

    def test_foreground_redshift_falls_back_to_selection_midpoint(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(root)
            redshift, source = context.foreground_redshift
            self.assertAlmostEqual(redshift, 0.3)
            self.assertEqual(source, "selection_foreground_z_midpoint")
            self.assertIs(context.foreground_redshift, context.foreground_redshift)


class RunnerTest(unittest.TestCase):
    def test_a_failing_stage_warns_and_the_rest_still_run(self) -> None:
        calls: list[str] = []

        def boom(context: PostRunContext, mode: str) -> tuple[Path, ...]:
            raise RuntimeError("deliberate failure")

        def ok(context: PostRunContext, mode: str) -> tuple[Path, ...]:
            calls.append(mode)
            path = context.stack_dir / f"ok_{mode}.txt"
            path.write_text("written", encoding="utf-8")
            return (path,)

        analyses = (
            PostRunAnalysis(name="boom", func=boom),
            PostRunAnalysis(name="ok", func=ok),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(root)
            with unittest.mock.patch(
                "dusty_colors.postrun.base.registered_analyses",
                return_value=analyses,
            ):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    outputs = run_post_run_analyses(context)

            self.assertEqual(calls, ["fcolors"])
            self.assertEqual(len(outputs), 1)
            self.assertTrue(outputs[0].exists())
            messages = [str(w.message) for w in caught]
            self.assertTrue(any("deliberate failure" in m for m in messages))
            self.assertTrue(any("boom fcolors" in m for m in messages))

    def test_disabled_stages_are_skipped(self) -> None:
        calls: list[str] = []

        def ok(context: PostRunContext, mode: str) -> tuple[Path, ...]:
            calls.append(mode)
            return ()

        analyses = (PostRunAnalysis(name="stack_figures", func=ok),)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(root, {"postrun": {"stack_figures": False}})
            with unittest.mock.patch(
                "dusty_colors.postrun.base.registered_analyses",
                return_value=analyses,
            ):
                run_post_run_analyses(context)
        self.assertEqual(calls, [])

    def test_end_to_end_writes_fits_and_figures(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = _build_context(
                root,
                {
                    "postrun": {
                        # The sample parquet files do not exist here, so the
                        # catalog-stats stage is expected to be unavailable.
                        "analysis_catalog_stats": False,
                        "dust_extinction_fit": {"covariance": "diagonal_errors"},
                    }
                },
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = run_post_run_analyses(context)

            names = {path.name for path in outputs}
            self.assertIn("dust_extinction_fit_fcolors.txt", names)
            self.assertIn("color_power_law_fits_fcolors.txt", names)
            self.assertIn("dp1_default_fcolors_extinction_curve_comparison.pdf", names)
            self.assertTrue(
                any(name.endswith("_all_colors.pdf") for name in names),
                f"expected a stack figure among {sorted(names)}",
            )
            for path in outputs:
                self.assertTrue(path.exists(), f"{path} was reported but not written")


class AnalysisHashTest(unittest.TestCase):
    """The postrun block must not participate in the stack config hash."""

    def test_adding_and_editing_postrun_leaves_the_hash_unchanged(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = load_resolved_config(_write_graph(root), root=root)

            with_block = load_resolved_config(
                _write_graph(
                    root, {"postrun": {"dust_extinction_fit": {"fixed_rv": 3.1}}}
                ),
                root=root,
            )
            edited = load_resolved_config(
                _write_graph(
                    root, {"postrun": {"dust_extinction_fit": {"fixed_rv": 2.2}}}
                ),
                root=root,
            )

            self.assertEqual(
                baseline.analysis.config_hash, with_block.analysis.config_hash
            )
            self.assertEqual(baseline.analysis.config_hash, edited.analysis.config_hash)

    def test_editing_the_stack_block_still_changes_the_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = load_resolved_config(_write_graph(root), root=root)
            changed = load_resolved_config(
                _write_graph(
                    root,
                    {
                        "stack": {
                            "colors": list(COLORS),
                            "modes": ["fcolors", "mcolors"],
                        }
                    },
                ),
                root=root,
            )
            self.assertNotEqual(
                baseline.analysis.config_hash, changed.analysis.config_hash
            )

    def test_configs_without_postrun_hash_through_the_untouched_path(self) -> None:
        """Guarantees the exclusion cannot invalidate a stack already on disk."""

        data = {"id": "a", "sample": "b", "stack": {"colors": ["g-r"]}}
        self.assertIs(hashable_analysis_data(data), data)
        self.assertEqual(stable_hash(hashable_analysis_data(data)), stable_hash(data))


if __name__ == "__main__":
    unittest.main()
