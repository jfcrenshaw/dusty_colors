"""End-to-end characterization test for the TreeCorr stacker.

This test does not check that the estimator is scientifically correct.
It checks that the estimator's output does not *change*, which is what makes it
safe to restructure `treecorr_stacker.py` without re-deriving the science.

It runs the full stack on a small deterministic synthetic sample with every
correction path enabled (flipped stack, random correction, reference annulus,
jackknife, diagnostics, both color modes) and compares every output array
against a stored reference.

If a change to the stacker is *intended* to change the numbers, regenerate the
reference deliberately and review the diff:

    python -m tests.test_stacker_characterization --regenerate
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd

from dusty_colors.treecorr_stacker import TreeCorrStacker

REFERENCE = Path(__file__).parent / "data" / "stack_characterization.npz"

NSIDE = 256
BANDS = ("g", "r", "i", "z")
# Deterministic everywhere: a fixed seed, a single thread, and bin_slop=0, so
# TreeCorr's pair summation order cannot vary between runs.
SEED = 20260812
NUM_THREADS = 1


def build_samples() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build a small synthetic foreground, background, and footprint.

    Returns
    -------
    tuple of pandas.DataFrame
        Foreground, background, and footprint tables.
    """
    rng = np.random.default_rng(SEED)

    def sky(n: int) -> tuple[np.ndarray, np.ndarray]:
        ra = 53.0 + rng.uniform(-0.5, 0.5, n)
        dec = -28.0 + rng.uniform(-0.5, 0.5, n)
        return ra, dec

    def photometry(n: int, frame: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for band in BANDS:
            flux = rng.lognormal(mean=1.0, sigma=0.3, size=n)
            frame[f"flux_{band}"] = flux
            frame[f"fluxerr_{band}"] = 0.05 * flux
        return frame

    n_fg, n_bg = 240, 1600

    fg_ra, fg_dec = sky(n_fg)
    foreground = photometry(
        n_fg,
        {
            "ra": fg_ra,
            "dec": fg_dec,
            "z_phot": rng.uniform(0.2, 0.5, n_fg),
        },
    )

    bg_ra, bg_dec = sky(n_bg)
    background = photometry(
        n_bg,
        {
            "ra": bg_ra,
            "dec": bg_dec,
            "z_phot": rng.uniform(0.7, 1.4, n_bg),
        },
    )

    fg = pd.DataFrame(foreground)
    bg = pd.DataFrame(background)

    # Four jackknife regions, split on the sky so patches are contiguous.
    for frame in (fg, bg):
        frame["jackknife_region"] = (frame["ra"] > 53.0).to_numpy(int) * 2 + (
            frame["dec"] > -28.0
        ).to_numpy(int)
        frame["pixel"] = hp.ang2pix(
            NSIDE, frame["ra"].to_numpy(), frame["dec"].to_numpy(), lonlat=True
        )

    footprint = (
        pd.concat(
            [fg[["pixel", "jackknife_region"]], bg[["pixel", "jackknife_region"]]]
        )
        .drop_duplicates(subset="pixel")
        .sort_values("pixel")
        .reset_index(drop=True)
    )
    return fg, bg, footprint


def run_stack(out_dir: Path) -> dict[str, np.ndarray]:
    """Run the full stacker and return every array it wrote.

    Parameters
    ----------
    out_dir : Path
        Directory the stacker writes into.

    Returns
    -------
    dict of str to numpy.ndarray
        Arrays from every output file, keyed as ``<file stem>/<array name>``.
    """
    foreground, background, footprint = build_samples()

    stacker = TreeCorrStacker(
        foreground=foreground,
        background=background,
        out_dir=out_dir,
        footprint=footprint,
        colors=("g-r", "i-z"),
        modes=("fcolors", "mcolors"),
        r_bin_edges=[20.0, 60.0, 200.0, 600.0],
        reference_annulus=(2000.0, 4000.0),
        bin_slop=0.0,
        num_threads=NUM_THREADS,
        jackknife=True,
        random_correction=True,
        random_multiplier=3,
        random_seed=42,
        random_nside=NSIDE,
        flipped_correction=True,
        diagnostic_plots=True,
    )
    stacker.run(force=True)

    arrays: dict[str, np.ndarray] = {}
    for path in sorted(out_dir.glob("*.npz")):
        with np.load(path) as handle:
            for key in sorted(handle.files):
                arrays[f"{path.stem}/{key}"] = handle[key]
    return arrays


class StackerCharacterizationTest(unittest.TestCase):
    def test_stack_output_matches_reference(self) -> None:
        if not REFERENCE.exists():  # pragma: no cover - guard for a missing fixture
            self.fail(
                f"Missing reference {REFERENCE}. "
                "Regenerate with: python -m tests.test_stacker_characterization "
                "--regenerate"
            )

        with tempfile.TemporaryDirectory() as tmp:
            produced = run_stack(Path(tmp))

        with np.load(REFERENCE) as handle:
            expected = {key: handle[key] for key in handle.files}

        self.assertEqual(
            sorted(produced), sorted(expected), "stack output array names changed"
        )

        for key in sorted(expected):
            with self.subTest(array=key):
                np.testing.assert_allclose(
                    produced[key],
                    expected[key],
                    rtol=0,
                    atol=0,
                    err_msg=f"{key} changed",
                )


def _regenerate() -> None:
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        arrays = run_stack(Path(tmp))
    np.savez_compressed(REFERENCE, **arrays)
    print(f"wrote {REFERENCE} with {len(arrays)} arrays")


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        unittest.main()
