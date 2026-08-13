"""Tests for the pair-weighted diagnostic figures."""

from __future__ import annotations

import os
import sys
import unittest
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

from test_plot_stacks import stack_arrays  # noqa: E402

from dusty_colors.postrun.plot_diagnostics import (  # noqa: E402
    plot_radial_distributions,
    save_stack_diagnostic_figures,
)
from dusty_colors.results import StackResults, load_stack_results  # noqa: E402


def diagnostic_arrays(colors: tuple[str, ...] = ("g-i", "g-r", "r-i")) -> dict:
    arrays = {
        "diagnostic_radial_bin_edges": np.geomspace(5.0, 160.0, 5),
        "diagnostic_radial_bin_centers": np.geomspace(5.0, 100.0, 4),
        "diagnostic_photoz_bin_edges": np.array([0.6, 0.9, 1.2, 1.5]),
        "diagnostic_photoz_counts": np.array(
            [
                [10.0, 3.0, 1.0],
                [8.0, 4.0, 2.0],
                [6.0, 5.0, 3.0],
                [4.0, 6.0, 4.0],
            ]
        ),
    }
    for color in colors:
        arrays[f"{color}_diagnostic_color_bin_edges"] = np.array([-0.5, 0.0, 0.5, 1.0])
        arrays[f"{color}_diagnostic_color_counts"] = np.array(
            [
                [2.0, 6.0, 2.0],
                [1.0, 7.0, 3.0],
                [1.0, 5.0, 4.0],
                [0.0, 4.0, 6.0],
            ]
        )
    return arrays


def results_for(colors: tuple[str, ...], **kwargs) -> StackResults:
    return StackResults(
        stack_dir=Path("dp1_default"),
        mode="fcolors",
        colors=colors,
        arrays=stack_arrays(colors),
        diagnostics=kwargs.get("diagnostics", diagnostic_arrays(colors)),
    )


class PlotRadialDistributionsTest(unittest.TestCase):
    def test_photoz_distribution_is_labelled_and_drawn(self) -> None:
        fig = plot_radial_distributions(results_for(("g-i", "g-r")), "photoz")
        ax = fig.axes[0]

        self.assertEqual(ax.get_xlabel(), "Photometric redshift")
        self.assertEqual(ax.get_ylabel(), "Pair density")
        self.assertEqual(ax.get_box_aspect(), 1)
        self.assertGreater(len(ax.patches), 0)

    def test_color_distribution_uses_a_math_label(self) -> None:
        fig = plot_radial_distributions(results_for(("g-i", "g-r")), "g-r")
        self.assertEqual(fig.axes[0].get_xlabel(), r"$g-r$ [mag]")
        self.assertGreater(len(fig.axes[0].patches), 0)

    def test_one_curve_per_radial_bin_labelled_by_range(self) -> None:
        fig = plot_radial_distributions(results_for(("g-i",)), "photoz")
        labels = [line.get_label() for line in fig.axes[0].patches]
        self.assertEqual(len(labels), 4)
        self.assertTrue(all(label.endswith(" kpc") for label in labels))

    def test_each_radial_bin_is_normalised_to_unit_area(self) -> None:
        """Bins with very different pair counts must be comparable by shape."""
        results = results_for(("g-i",))
        fig = plot_radial_distributions(results, "photoz")
        edges = results.diagnostics["diagnostic_photoz_bin_edges"]
        widths = np.diff(edges)
        for patch in fig.axes[0].patches:
            densities = np.asarray(patch.get_data()[0], dtype=float)
            self.assertAlmostEqual(float(np.sum(densities * widths)), 1.0, places=10)

    def test_mismatched_edges_are_rejected(self) -> None:
        diagnostics = diagnostic_arrays(("g-i",))
        diagnostics["diagnostic_photoz_bin_edges"] = np.array([0.6, 0.9])
        results = results_for(("g-i",), diagnostics=diagnostics)
        with self.assertRaises(ValueError) as caught:
            plot_radial_distributions(results, "photoz")
        self.assertIn("bin_edges", str(caught.exception))

    def test_empty_histograms_raise(self) -> None:
        diagnostics = diagnostic_arrays(("g-i",))
        diagnostics["diagnostic_photoz_counts"] = np.zeros((4, 3))
        results = results_for(("g-i",), diagnostics=diagnostics)
        with self.assertRaises(ValueError):
            plot_radial_distributions(results, "photoz")


class SaveStackDiagnosticFiguresTest(unittest.TestCase):
    def test_writes_photoz_plus_one_figure_per_color(self) -> None:
        results = results_for(("g-i", "g-r"))
        with TemporaryDirectory() as tmp:
            paths = save_stack_diagnostic_figures(results, tmp)

            self.assertEqual(
                [path.name for path in paths],
                [
                    "dp1_default_fcolors_photoz_distribution.pdf",
                    "dp1_default_fcolors_g_i_color_distribution.pdf",
                    "dp1_default_fcolors_g_r_color_distribution.pdf",
                ],
            )
            self.assertTrue(all(path.exists() for path in paths))

    def test_returns_nothing_when_the_stack_has_no_diagnostics(self) -> None:
        results = StackResults(
            stack_dir=Path("dp1_default"),
            mode="fcolors",
            colors=("g-i",),
            arrays=stack_arrays(("g-i",)),
        )
        with TemporaryDirectory() as tmp:
            self.assertEqual(save_stack_diagnostic_figures(results, tmp), ())

    def test_skips_colors_without_diagnostic_arrays(self) -> None:
        diagnostics = diagnostic_arrays(("g-i",))
        results = results_for(("g-i", "g-r"), diagnostics=diagnostics)
        with TemporaryDirectory() as tmp:
            paths = save_stack_diagnostic_figures(results, tmp)
            names = [path.name for path in paths]
        self.assertIn("dp1_default_fcolors_g_i_color_distribution.pdf", names)
        self.assertNotIn("dp1_default_fcolors_g_r_color_distribution.pdf", names)


class LoadFromDiskTest(unittest.TestCase):
    def test_diagnostics_load_separately_from_the_science_arrays(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, data in (
                ("configs/catalogs/dp1.yaml", {"id": "dp1"}),
                (
                    "configs/samples/dp1_default.yaml",
                    {"id": "dp1_default", "catalog": "configs/catalogs/dp1.yaml"},
                ),
                (
                    "configs/analyses/dp1_default.yaml",
                    {
                        "id": "dp1_default",
                        "sample": "configs/samples/dp1_default.yaml",
                        "stack": {
                            "colors": ["g-i", "g-r", "r-i"],
                            "modes": ["fcolors"],
                        },
                    },
                ),
            ):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

            stack_dir = root / "results/stacks/dp1_default"
            stack_dir.mkdir(parents=True)
            np.savez_compressed(stack_dir / "stack_fcolors.npz", **stack_arrays())
            np.savez_compressed(
                stack_dir / "stack_fcolors_diagnostics.npz", **diagnostic_arrays()
            )

            results = load_stack_results(
                root / "configs/analyses/dp1_default.yaml", root=root
            )

            self.assertEqual(results.colors, ("g-i", "g-r", "r-i"))
            self.assertIn("diagnostic_photoz_counts", results.diagnostics)
            self.assertNotIn("diagnostic_photoz_counts", results.arrays)


if __name__ == "__main__":
    unittest.main()
