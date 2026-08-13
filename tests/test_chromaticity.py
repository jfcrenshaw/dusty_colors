"""Tests for the chromaticity post-run analysis."""

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

from dusty_colors.postrun.chromaticity import (  # noqa: E402
    ChromaticityConfig,
    _curve_coefficients,
    band_relative_chain,
    band_relative_points,
    fit_radial_power_law,
    parse_chromaticity_options,
    save_stack_chromaticity_figure,
)
from dusty_colors.postrun.dust_extinction_fit import _dust_law  # noqa: E402
from dusty_colors.results import StackResults  # noqa: E402

BANDS = ("g", "r", "i", "z")
COLORS = ("g-r", "r-i", "i-z")


class BandRelativeChainTest(unittest.TestCase):
    def test_bluewards_target_adds_the_measured_colors(self) -> None:
        self.assertEqual(band_relative_chain(BANDS, "r", "g"), (("g-r", 1.0),))

    def test_redwards_target_subtracts_the_measured_colors(self) -> None:
        self.assertEqual(band_relative_chain(BANDS, "r", "i"), (("r-i", -1.0),))
        self.assertEqual(
            band_relative_chain(BANDS, "r", "z"),
            (("r-i", -1.0), ("i-z", -1.0)),
        )

    def test_reference_band_itself_is_an_empty_walk(self) -> None:
        self.assertEqual(band_relative_chain(BANDS, "r", "r"), ())

    def test_unknown_band_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            band_relative_chain(BANDS, "r", "y")


def _results(stack_dir: Path, *, amplitude: float, alpha: float) -> StackResults:
    """Build a stack whose colors are exactly an F99 reddening profile.

    Using the same law the fitter will assume means a correct implementation
    must recover ``amplitude`` and ``alpha`` to numerical precision.
    """

    radius = np.array([12.0, 25.0, 70.0, 300.0])
    law = _dust_law("F99", 3.1)
    wavelengths = ChromaticityConfig().wavelengths_um

    av = amplitude * (radius / 20.0) ** alpha
    arrays: dict[str, np.ndarray] = {}
    for blue, red in (("g", "r"), ("r", "i"), ("i", "z")):
        coefficient = float(
            _curve_coefficients(
                law, [wavelengths[blue]], lambda_ref=wavelengths[red], redshift=0.0
            )[0]
        )
        signal = av * coefficient
        arrays[f"{blue}-{red}_bin_centers"] = radius
        arrays[f"{blue}-{red}_avg"] = signal
        arrays[f"{blue}-{red}_err"] = np.full_like(signal, 1.0e-3)
        # Three patches spread symmetrically, so the jackknife errors are finite
        # and equal across bins without biasing the mean.
        arrays[f"{blue}-{red}_jackknife_samples"] = np.vstack(
            [signal * 0.99, signal, signal * 1.01]
        )
    return StackResults(
        stack_dir=stack_dir,
        mode="fcolors",
        colors=COLORS,
        arrays=arrays,
    )


class BandRelativePointsTest(unittest.TestCase):
    def test_points_have_the_expected_shape_and_signs(self) -> None:
        with TemporaryDirectory() as tmp:
            results = _results(Path(tmp), amplitude=0.1, alpha=-1.0)
            values, errors = band_relative_points(results, ChromaticityConfig())

            self.assertEqual(values.shape, (4, 3))
            self.assertEqual(errors.shape, (4, 3))
            self.assertTrue(np.all(errors > 0))
            # g is bluer than r so it is reddened more; i and z are redder so
            # their excesses relative to r are negative.
            self.assertTrue(np.all(values[:, 0] > 0))
            self.assertTrue(np.all(values[:, 1] < 0))
            self.assertTrue(np.all(values[:, 2] < values[:, 1]))

    def test_errors_use_the_correlated_jackknife_not_quadrature(self) -> None:
        with TemporaryDirectory() as tmp:
            results = _results(Path(tmp), amplitude=0.1, alpha=-1.0)
            _, errors = band_relative_points(results, ChromaticityConfig())

            samples = results.arrays["r-i_jackknife_samples"]
            n = samples.shape[0]
            centered = samples - samples.mean(axis=0)
            expected = np.sqrt((1.0 - 1.0 / n) * np.sum(centered**2, axis=0))
            np.testing.assert_allclose(errors[:, 1], expected, rtol=1e-12)


class RadialFitTest(unittest.TestCase):
    def test_recovers_the_injected_amplitude_and_slope(self) -> None:
        with TemporaryDirectory() as tmp:
            config = ChromaticityConfig()
            results = _results(Path(tmp), amplitude=0.08, alpha=-1.3)
            values, errors = band_relative_points(results, config)

            law = _dust_law("F99", 3.1)
            coefficients = _curve_coefficients(
                law,
                [config.wavelengths_um[b] for b in config.point_bands],
                lambda_ref=config.wavelengths_um[config.reference_band],
                redshift=0.0,
            )
            fit = fit_radial_power_law(
                results.arrays["g-r_bin_centers"],
                values,
                errors,
                coefficients,
                pivot_kpc=20.0,
            )

            self.assertTrue(fit.optimizer_success)
            self.assertAlmostEqual(fit.amplitude, 0.08, places=6)
            self.assertAlmostEqual(fit.alpha, -1.3, places=6)
            self.assertLess(fit.chi2, 1.0e-6)

    def test_fit_bin_indices_restrict_the_radial_range(self) -> None:
        with TemporaryDirectory() as tmp:
            config = ChromaticityConfig()
            results = _results(Path(tmp), amplitude=0.08, alpha=-1.3)
            values, errors = band_relative_points(results, config)
            coefficients = _curve_coefficients(
                law := _dust_law("F99", 3.1),
                [config.wavelengths_um[b] for b in config.point_bands],
                lambda_ref=config.wavelengths_um[config.reference_band],
                redshift=0.0,
            )
            del law
            fit = fit_radial_power_law(
                results.arrays["g-r_bin_centers"],
                values,
                errors,
                coefficients,
                pivot_kpc=20.0,
                fit_bin_indices=(0, 1, 2),
            )
            # Three radial bins times three point bands, minus two parameters.
            self.assertEqual(fit.dof, 7)


class ConfigParsingTest(unittest.TestCase):
    def test_false_disables_the_stage(self) -> None:
        self.assertFalse(parse_chromaticity_options(False).enabled)

    def test_true_and_none_give_defaults(self) -> None:
        for raw in (True, None):
            config = parse_chromaticity_options(raw)
            self.assertTrue(config.enabled)
            self.assertEqual(config.bands, BANDS)
            self.assertEqual(config.radial_pivot_kpc, 20.0)

    def test_options_override_defaults(self) -> None:
        config = parse_chromaticity_options(
            {
                "bands": ["g", "r", "i"],
                "point_bands": ["g", "i"],
                "reference_band": "r",
                "radial_pivot_kpc": 50.0,
                "fit_bin_indices": [0, 1],
                "extension": "pdf",
                "wavelengths_um": {"g": 0.5},
            }
        )
        self.assertEqual(config.bands, ("g", "r", "i"))
        self.assertEqual(config.point_bands, ("g", "i"))
        self.assertEqual(config.radial_pivot_kpc, 50.0)
        self.assertEqual(config.fit_bin_indices, (0, 1))
        self.assertEqual(config.extension, ("pdf",))
        self.assertEqual(config.wavelengths_um["g"], 0.5)
        # Unlisted bands keep their default wavelengths.
        self.assertIn("z", config.wavelengths_um)

    def test_non_mapping_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_chromaticity_options(["nope"])


class SaveFigureTest(unittest.TestCase):
    def test_writes_both_extensions(self) -> None:
        with TemporaryDirectory() as tmp:
            stack_dir = Path(tmp) / "results/stacks/dp1_default"
            stack_dir.mkdir(parents=True)
            results = _results(stack_dir, amplitude=0.1, alpha=-1.0)

            paths = save_stack_chromaticity_figure(
                results, stack_dir, foreground_redshift=0.36
            )
            self.assertEqual(len(paths), 2)
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)
            self.assertEqual({path.suffix for path in paths}, {".png", ".pdf"})

    def test_declines_when_a_required_color_is_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            stack_dir = Path(tmp)
            results = _results(stack_dir, amplitude=0.1, alpha=-1.0)
            stripped = StackResults(
                stack_dir=stack_dir,
                mode="fcolors",
                colors=COLORS,
                arrays={
                    k: v for k, v in results.arrays.items() if not k.startswith("i-z")
                },
            )
            self.assertEqual(save_stack_chromaticity_figure(stripped, stack_dir), ())

    def test_disabled_config_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            stack_dir = Path(tmp)
            results = _results(stack_dir, amplitude=0.1, alpha=-1.0)
            paths = save_stack_chromaticity_figure(
                results, stack_dir, config=ChromaticityConfig(enabled=False)
            )
            self.assertEqual(paths, ())


if __name__ == "__main__":
    unittest.main()
