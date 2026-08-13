"""Tests for the science figures."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/dusty_colors_mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/dusty_colors_cache")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.postrun.plot_stacks import (  # noqa: E402
    COLOR_STYLES,
    plot_color_signals,
    plot_jackknife_samples,
    save_stack_figures,
)
from dusty_colors.results import StackResults  # noqa: E402


def stack_arrays(colors: tuple[str, ...] = ("g-i", "g-r", "r-i")) -> dict:
    radius = np.geomspace(5.0, 100.0, 4)
    arrays: dict = {}
    for index, color in enumerate(colors):
        signal = np.array([0.08, 0.03, 0.012, 0.006]) * (index + 1)
        err = signal * 0.2
        samples = np.vstack([signal * 0.9, signal, signal * 1.1])
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


def results_for(
    colors: tuple[str, ...], stack_dir: str = "dp1_default"
) -> StackResults:
    return StackResults(
        stack_dir=Path(stack_dir),
        mode="fcolors",
        colors=colors,
        arrays=stack_arrays(colors),
    )


class PlotJackknifeSamplesTest(unittest.TestCase):
    def test_is_square_log_log_and_draws_every_sample(self) -> None:
        results = results_for(("g-i", "g-r"))
        fig = plot_jackknife_samples(results)
        ax = fig.axes[0]

        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(fig.get_size_inches().tolist(), [3.0, 3.0])
        self.assertEqual(ax.get_box_aspect(), 1)
        # One faint line per jackknife patch. Identified by alpha because
        # errorbar also contributes lines for its caps.
        faint = [line for line in ax.lines if line.get_alpha() == 0.18]
        self.assertEqual(len(faint), 3)
        # Plus exactly one errorbar container for the jackknife mean.
        self.assertEqual(len(ax.containers), 1)

    def test_labels_name_the_first_color_only(self) -> None:
        fig = plot_jackknife_samples(results_for(("g-i", "g-r")))
        ax = fig.axes[0]
        self.assertEqual(ax.get_ylabel(), r"$E(g-i)$ [mag]")
        self.assertEqual(ax.get_xlabel(), r"$r_\perp$ [kpc]")

    def test_rejects_samples_that_do_not_match_the_radial_bins(self) -> None:
        arrays = stack_arrays(("g-i",))
        arrays["g-i_jackknife_samples"] = np.zeros((3, 2))
        results = StackResults(
            stack_dir=Path("unused"), mode="fcolors", colors=("g-i",), arrays=arrays
        )
        with self.assertRaises(ValueError) as caught:
            plot_jackknife_samples(results)
        self.assertIn("jackknife_samples", str(caught.exception))

    def test_raises_when_nothing_is_loggable(self) -> None:
        arrays = stack_arrays(("g-i",))
        arrays["g-i_jackknife_avg"] = np.full(4, -1.0)
        results = StackResults(
            stack_dir=Path("unused"), mode="fcolors", colors=("g-i",), arrays=arrays
        )
        with self.assertRaises(ValueError):
            plot_jackknife_samples(results)


class PlotColorSignalsTest(unittest.TestCase):
    def test_draws_one_series_per_color(self) -> None:
        fig = plot_color_signals(results_for(("g-i", "g-r", "r-i")))
        ax = fig.axes[0]

        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(fig.get_size_inches().tolist(), [3.0, 3.0])
        self.assertEqual(len(ax.containers), 3)
        self.assertEqual(ax.get_ylabel(), "Color excess [mag]")

    def test_uses_the_shared_color_convention(self) -> None:
        """A color must look the same here as in every other figure."""
        fig = plot_color_signals(results_for(("g-r", "r-i")))
        labels = [c.get_label() for c in fig.axes[0].containers]
        self.assertEqual(labels, ["$g-r$", "$r-i$"])
        self.assertEqual(COLOR_STYLES["g-r"], "C0")
        self.assertEqual(COLOR_STYLES["r-i"], "C2")

    def test_requires_jackknife_errors(self) -> None:
        arrays = stack_arrays(("g-i",))
        del arrays["g-i_jackknife_err"]
        results = StackResults(
            stack_dir=Path("unused"), mode="fcolors", colors=("g-i",), arrays=arrays
        )
        with self.assertRaises(KeyError) as caught:
            plot_color_signals(results)
        self.assertIn("jackknife_err", str(caught.exception))


class SaveStackFiguresTest(unittest.TestCase):
    def test_writes_both_figures_with_expected_names(self) -> None:
        results = results_for(("g-i", "g-r"))
        with TemporaryDirectory() as tmp:
            paths = save_stack_figures(results, tmp)

            self.assertEqual(len(paths), 2)
            self.assertEqual(
                [path.name for path in paths],
                [
                    "dp1_default_fcolors_g_i_jackknife.pdf",
                    "dp1_default_fcolors_all_colors.pdf",
                ],
            )
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_extension_is_configurable(self) -> None:
        results = results_for(("g-i",))
        with TemporaryDirectory() as tmp:
            paths = save_stack_figures(results, tmp, extension="png")
            self.assertTrue(all(path.suffix == ".png" for path in paths))


if __name__ == "__main__":
    unittest.main()
