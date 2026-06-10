from __future__ import annotations

from pathlib import Path
import os
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dusty_colors_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/dusty_colors_cache")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.plotting import (
    StackResults,
    default_style_path,
    load_stack_results,
    plot_all_color_signals,
    plot_color_radial_distributions,
    plot_first_color_jackknife,
    plot_photoz_radial_distributions,
    save_stack_diagnostic_figures,
)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _write_graph(root: Path) -> Path:
    _write_yaml(root / "configs/catalogs/dp1.yaml", {"id": "dp1"})
    _write_yaml(
        root / "configs/samples/dp1_default.yaml",
        {
            "id": "dp1_default",
            "catalog": "configs/catalogs/dp1.yaml",
        },
    )
    analysis_path = root / "configs/analyses/dp1_default.yaml"
    _write_yaml(
        analysis_path,
        {
            "id": "dp1_default",
            "sample": "configs/samples/dp1_default.yaml",
            "stack": {
                "colors": ["g-i", "g-r", "r-i"],
                "modes": ["fcolors"],
            },
        },
    )
    return analysis_path


def _stack_arrays(colors: tuple[str, ...] = ("g-i", "g-r", "r-i")) -> dict:
    radius = np.geomspace(5.0, 100.0, 4)
    arrays = {}
    for index, color in enumerate(colors):
        signal = np.array([0.08, 0.03, 0.012, 0.006]) * (index + 1)
        err = signal * 0.2
        samples = np.vstack(
            [
                signal * 0.9,
                signal * 1.0,
                signal * 1.1,
            ]
        )
        arrays.update(
            {
                f"{color}_bin_centers": radius,
                f"{color}_avg": signal,
                f"{color}_err": err,
                f"{color}_cov": np.diag(err**2),
                f"{color}_jackknife_avg": samples.mean(axis=0),
                f"{color}_jackknife_err": err,
                f"{color}_jackknife_samples": samples,
            }
        )
    return arrays


def _diagnostic_arrays(colors: tuple[str, ...] = ("g-i", "g-r", "r-i")) -> dict:
    radius = np.geomspace(5.0, 100.0, 4)
    radial_edges = np.geomspace(5.0, 160.0, 5)
    arrays = {
        "diagnostic_radial_bin_edges": radial_edges,
        "diagnostic_radial_bin_centers": radius,
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
        arrays.update(
            {
                f"{color}_diagnostic_color_bin_edges": np.array([-0.5, 0.0, 0.5, 1.0]),
                f"{color}_diagnostic_color_counts": np.array(
                    [
                        [2.0, 6.0, 2.0],
                        [1.0, 7.0, 3.0],
                        [1.0, 5.0, 4.0],
                        [0.0, 4.0, 6.0],
                    ]
                ),
            }
        )
    return arrays


class PlottingTest(unittest.TestCase):
    def test_default_style_path_exists(self) -> None:
        self.assertTrue(default_style_path().exists())

    def test_load_stack_results_uses_config_color_order(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            analysis_path = _write_graph(root)
            stack_dir = root / "results/stacks/dp1_default"
            stack_dir.mkdir(parents=True)
            np.savez_compressed(stack_dir / "stack_fcolors.npz", **_stack_arrays())
            np.savez_compressed(
                stack_dir / "stack_fcolors_diagnostics.npz",
                **_diagnostic_arrays(),
            )

            results = load_stack_results(analysis_path, root=root)

            self.assertEqual(results.mode, "fcolors")
            self.assertEqual(results.colors, ("g-i", "g-r", "r-i"))
            self.assertEqual(results.first_color, "g-i")
            self.assertIn("diagnostic_photoz_counts", results.diagnostics)
            self.assertNotIn("diagnostic_photoz_counts", results.arrays)

    def test_plot_first_color_jackknife_is_square_and_log_scaled(self) -> None:
        results = StackResults(
            stack_dir=Path("unused"),
            mode="fcolors",
            colors=("g-i", "g-r"),
            arrays=_stack_arrays(("g-i", "g-r")),
            diagnostics=_diagnostic_arrays(("g-i", "g-r")),
        )

        fig = plot_first_color_jackknife(results)
        ax = fig.axes[0]

        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(fig.get_size_inches().tolist(), [3.0, 3.0])
        self.assertGreaterEqual(len(ax.lines), 3)
        self.assertEqual(len(ax.containers), 1)

    def test_plot_all_color_signals_uses_full_signal_and_jackknife_errors(self) -> None:
        results = StackResults(
            stack_dir=Path("unused"),
            mode="fcolors",
            colors=("g-i", "g-r", "r-i"),
            arrays=_stack_arrays(("g-i", "g-r", "r-i")),
        )

        fig = plot_all_color_signals(results)
        ax = fig.axes[0]

        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(fig.get_size_inches().tolist(), [3.0, 3.0])
        self.assertEqual(len(ax.containers), 3)

    def test_plot_radial_distribution_diagnostics(self) -> None:
        results = StackResults(
            stack_dir=Path("unused"),
            mode="fcolors",
            colors=("g-i", "g-r"),
            arrays=_stack_arrays(("g-i", "g-r")),
            diagnostics=_diagnostic_arrays(("g-i", "g-r")),
        )

        fig = plot_photoz_radial_distributions(results)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Photometric redshift")
        self.assertGreater(len(ax.patches), 0)

        fig = plot_color_radial_distributions(results, "g-r")
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), r"$g-r$ [mag]")
        self.assertGreater(len(ax.patches), 0)

    def test_save_stack_diagnostic_figures_writes_available_diagnostics(self) -> None:
        results = StackResults(
            stack_dir=Path("dp1_default"),
            mode="fcolors",
            colors=("g-i", "g-r"),
            arrays=_stack_arrays(("g-i", "g-r")),
            diagnostics=_diagnostic_arrays(("g-i", "g-r")),
        )

        with TemporaryDirectory() as tmp:
            paths = save_stack_diagnostic_figures(results, tmp)

            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
