from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dusty_colors.color_power_law_fit import (  # noqa: E402
    fit_color_power_law,
    save_stack_color_power_law_fits,
)
from dusty_colors.plotting import StackResults  # noqa: E402


def _stack_arrays(
    colors: tuple[str, ...] = ("g-i", "g-r"),
    *,
    amplitude: float = 0.012,
    alpha: float = -0.8,
) -> dict[str, np.ndarray]:
    radius = np.geomspace(10.0, 800.0, 6)
    arrays: dict[str, np.ndarray] = {}
    for index, color in enumerate(colors):
        signal = amplitude * (index + 1) * (radius / 100.0) ** alpha
        error = np.full_like(signal, 2.0e-4)
        arrays.update(
            {
                f"{color}_bin_centers": radius,
                f"{color}_avg": signal,
                f"{color}_jackknife_err": error,
            }
        )
    return arrays


class ColorPowerLawFitTest(unittest.TestCase):
    def test_fit_recovers_synthetic_power_law(self) -> None:
        results = StackResults(
            stack_dir=Path("unused"),
            mode="fcolors",
            colors=("g-i",),
            arrays=_stack_arrays(("g-i",)),
        )

        fit = fit_color_power_law(results, "g-i")

        self.assertIsNotNone(fit)
        assert fit is not None
        self.assertAlmostEqual(fit.amplitude_mag, 0.012, delta=1.0e-5)
        self.assertAlmostEqual(fit.alpha, -0.8, delta=1.0e-4)
        self.assertLess(fit.chi2, 1.0e-8)

    def test_save_writes_one_report_for_available_colors(self) -> None:
        with TemporaryDirectory() as tmp:
            stack_dir = Path(tmp)
            np.savez_compressed(
                stack_dir / "stack_fcolors.npz",
                **_stack_arrays(("g-i", "g-r")),
            )

            path = save_stack_color_power_law_fits(
                stack_dir,
                stack_dir,
                mode="fcolors",
            )

            self.assertIsNotNone(path)
            assert path is not None
            report = path.read_text(encoding="utf-8")
            self.assertIn("Direct color power-law fits", report)
            self.assertIn("g-i:", report)
            self.assertIn("g-r:", report)
            self.assertIn("amplitude_at_pivot_mag", report)


if __name__ == "__main__":
    unittest.main()
