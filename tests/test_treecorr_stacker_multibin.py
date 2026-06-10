from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dusty_colors.treecorr_stacker import TreeCorrStacker


class TreeCorrStackerMultibinTest(unittest.TestCase):
    def test_stack_bins_batches_moment_passes_for_log_bins(self) -> None:
        stacker = TreeCorrStacker(
            foreground=pd.DataFrame(),
            background=pd.DataFrame(),
            out_dir="unused",
            r_bin_edges=[10.0, 100.0, 1000.0],
        )
        bins = stacker._radial_bins()
        value = np.array([2.0, 4.0, 8.0])
        err = np.array([1.0, 2.0, 4.0])
        good = np.array([True, False, True])
        moment_calls = []
        final_calls = []

        def pair_means(
            value_arg,
            good_arg,
            radial_bins,
            direction,
            weight=None,
            random=False,
        ):
            moment_calls.append(np.asarray(value_arg).copy())
            self.assertIsNone(weight)
            self.assertEqual(radial_bins, bins)
            self.assertEqual(direction, "forward")
            self.assertFalse(random)
            np.testing.assert_array_equal(good_arg, good)
            if len(moment_calls) == 1:
                return (
                    np.array([2.0, 5.0]),
                    np.array([3.0, 4.0]),
                    np.array([30.0, 40.0]),
                    (),
                )
            return (
                np.array([6.0, 34.0]),
                np.array([3.0, 4.0]),
                np.array([30.0, 40.0]),
                (),
            )

        def pair_mean(
            value_arg,
            good_arg,
            radial_bin,
            direction,
            weight=None,
            random=False,
        ):
            final_calls.append((radial_bin, np.asarray(weight).copy()))
            call_number = len(final_calls)
            return (
                float(10 * call_number),
                float(100 * call_number),
                float(1000 * call_number),
                (f"corr{call_number}",),
            )

        stacker._pair_means = pair_means
        stacker._pair_mean = pair_mean

        results = stacker._stack_bins(value, err, good, bins, "forward")

        self.assertEqual(len(moment_calls), 2)
        self.assertEqual(len(final_calls), 2)
        np.testing.assert_allclose(moment_calls[1], value**2)
        np.testing.assert_allclose(final_calls[0][1], [1.0 / 3.0, 0.0, 1.0 / 18.0])
        np.testing.assert_allclose(final_calls[1][1], [1.0 / 10.0, 0.0, 1.0 / 25.0])
        np.testing.assert_allclose([result.raw for result in results], [10.0, 20.0])
        np.testing.assert_allclose(
            [result.raw_err for result in results],
            [0.1, np.sqrt(1.0 / 200.0)],
        )
        np.testing.assert_allclose(
            [result.weight for result in results], [100.0, 200.0]
        )
        np.testing.assert_allclose(
            [result.npairs for result in results], [1000.0, 2000.0]
        )
        self.assertEqual([result.corrs for result in results], [("corr1",), ("corr2",)])

    def test_stack_bins_falls_back_for_irregular_bins(self) -> None:
        stacker = TreeCorrStacker(
            foreground=pd.DataFrame(),
            background=pd.DataFrame(),
            out_dir="unused",
            r_bin_edges=[10.0, 15.0, 40.0],
        )
        bins = stacker._radial_bins()
        calls = []

        def pair_means(*args, **kwargs):
            raise AssertionError("irregular bins should not use batched moments")

        def stack_bin(value, err, good, radial_bin, direction, random=False):
            calls.append(radial_bin)
            return len(calls)

        stacker._pair_means = pair_means
        stacker._stack_bin = stack_bin

        results = stacker._stack_bins(
            np.array([1.0]),
            np.array([1.0]),
            np.array([True]),
            bins,
            "forward",
        )

        self.assertEqual(results, [1, 2])
        self.assertEqual(calls, bins)


if __name__ == "__main__":
    unittest.main()
