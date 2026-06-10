from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.dust_extinction_fit import (  # noqa: E402
    DEFAULT_FILTER_WAVELENGTHS_UM,
    DustExtinctionFitConfig,
    color_excess_per_av,
    color_excess_to_av,
    dust_color_excess_model,
    fit_dust_extinction_law,
    save_stack_dust_extinction_fit,
)
from dusty_colors.plotting import StackResults  # noqa: E402


HAS_DUST_EXTINCTION = find_spec("dust_extinction") is not None


def _synthetic_stack_arrays(
    *,
    amplitude: float = 0.025,
    alpha: float = -0.75,
    rv: float = 3.4,
    foreground_redshift: float = 0.35,
) -> dict[str, np.ndarray]:
    radius = np.geomspace(10.0, 900.0, 7)
    arrays: dict[str, np.ndarray] = {}
    for color in ("g-r", "r-i", "i-z"):
        row_colors = (color,) * len(radius)
        signal = dust_color_excess_model(
            row_colors,
            radius,
            amplitude_av_mag=amplitude,
            alpha=alpha,
            rv=rv,
            foreground_redshift=foreground_redshift,
        )
        err = np.full_like(signal, 5.0e-4)
        arrays.update(
            {
                f"{color}_bin_centers": radius,
                f"{color}_avg": signal,
                f"{color}_err": err,
                f"{color}_cov": np.diag(err**2),
            }
        )
    return arrays


@unittest.skipUnless(HAS_DUST_EXTINCTION, "dust_extinction is not installed")
class DustExtinctionFitTest(unittest.TestCase):
    def test_color_excess_to_av_inverts_all_rubin_filter_pairs(self) -> None:
        expected_av = np.array([0.003, 0.01, 0.04])

        for band1 in DEFAULT_FILTER_WAVELENGTHS_UM:
            for band2 in DEFAULT_FILTER_WAVELENGTHS_UM:
                if band1 == band2:
                    continue
                color = f"{band1}-{band2}"
                coefficient = color_excess_per_av(
                    color,
                    rv=3.1,
                    foreground_redshift=0.35,
                )
                excess = expected_av * coefficient

                av = color_excess_to_av(
                    excess,
                    color,
                    rv=3.1,
                    foreground_redshift=0.35,
                )

                np.testing.assert_allclose(av, expected_av, rtol=1.0e-12)

    def test_color_excess_to_av_accepts_scalar(self) -> None:
        coefficient = color_excess_per_av("g-i", foreground_redshift=0.35)

        av = color_excess_to_av(0.02 * coefficient, "g-i", foreground_redshift=0.35)

        self.assertIsInstance(av, float)
        self.assertAlmostEqual(av, 0.02)

    def test_color_excess_to_av_rejects_zero_contrast_color(self) -> None:
        with self.assertRaisesRegex(ValueError, "zero extinction contrast"):
            color_excess_to_av(0.01, "g-g")

    def test_fit_recovers_synthetic_parameters(self) -> None:
        foreground_redshift = 0.35
        results = StackResults(
            stack_dir=Path("unused"),
            mode="fcolors",
            colors=("g-r", "r-i", "i-z"),
            arrays=_synthetic_stack_arrays(foreground_redshift=foreground_redshift),
        )

        fit = fit_dust_extinction_law(
            results,
            config=DustExtinctionFitConfig(initial_rv=3.1),
            foreground_redshift=foreground_redshift,
        )

        self.assertAlmostEqual(fit.amplitude_av_mag, 0.025, delta=1.0e-4)
        self.assertAlmostEqual(fit.alpha, -0.75, delta=1.0e-3)
        self.assertAlmostEqual(fit.rv, 3.4, delta=1.0e-2)
        self.assertLess(fit.chi2, 1.0e-6)

    def test_fit_can_fix_rv(self) -> None:
        foreground_redshift = 0.35
        results = StackResults(
            stack_dir=Path("unused"),
            mode="fcolors",
            colors=("g-r", "r-i", "i-z"),
            arrays=_synthetic_stack_arrays(
                rv=3.1,
                foreground_redshift=foreground_redshift,
            ),
        )

        fit = fit_dust_extinction_law(
            results,
            config=DustExtinctionFitConfig(fixed_rv=3.1),
            foreground_redshift=foreground_redshift,
        )

        self.assertTrue(fit.rv_fixed)
        self.assertEqual(len(fit.parameter_errors), 2)
        self.assertAlmostEqual(fit.rv, 3.1)
        self.assertAlmostEqual(fit.amplitude_av_mag, 0.025, delta=1.0e-4)
        self.assertAlmostEqual(fit.alpha, -0.75, delta=1.0e-3)

    def test_save_writes_text_report_when_required_colors_exist(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            stack_dir = root / "results/stacks/test"
            stack_dir.mkdir(parents=True)
            np.savez_compressed(
                stack_dir / "stack_fcolors.npz",
                **_synthetic_stack_arrays(),
            )

            path = save_stack_dust_extinction_fit(
                stack_dir,
                stack_dir,
                mode="fcolors",
                foreground_redshift=0.35,
            )

            self.assertIsNotNone(path)
            assert path is not None
            report = path.read_text(encoding="utf-8")
            self.assertIn("amplitude_Av_at_pivot_mag", report)
            self.assertIn("radial_power_law_index_alpha", report)
            self.assertIn("R_V", report)

    def test_save_skips_stack_without_required_colors(self) -> None:
        with TemporaryDirectory() as tmp:
            stack_dir = Path(tmp)
            np.savez_compressed(
                stack_dir / "stack_fcolors.npz",
                **{
                    "g-i_bin_centers": np.array([10.0, 20.0]),
                    "g-i_avg": np.array([0.01, 0.005]),
                    "g-i_err": np.array([0.001, 0.001]),
                },
            )

            path = save_stack_dust_extinction_fit(
                stack_dir,
                stack_dir,
                mode="fcolors",
                foreground_redshift=0.35,
            )

            self.assertIsNone(path)


if __name__ == "__main__":
    unittest.main()
