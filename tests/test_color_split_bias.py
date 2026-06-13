from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from dusty_colors.color_split_bias import (
    BG_REDDER_GROUP,
    FG_REDDER_GROUP,
    HIGH_GROUP,
    LOW_GROUP,
    analyze_color_split_bin,
    analyze_proxy_split_bin,
    attach_inner_bin_exclusion_flags,
    attach_independent_proxy_columns,
    attach_proxy_groups,
    attach_proxy_signal_normalization,
    attach_color_split_columns,
    contamination_summary_text,
    fcolor_stack,
    proxy_contamination_summary_table,
    summarize_inner_bin_exclusion_rerun,
    summarize_inner_bin_pair_influence,
    summarize_pair_influence_table,
)


class ColorSplitBiasTest(unittest.TestCase):
    def test_attach_color_split_columns_splits_by_g_minus_z_color(self) -> None:
        pairs = pd.DataFrame(
            {
                "pair_id": ["a", "b"],
                "foreground_index": [0, 1],
                "background_index": [0, 1],
                "theta_arcsec": [2.0, 3.0],
                "r_perp_kpc": [12.0, 20.0],
            }
        )
        foreground = pd.DataFrame(
            {
                "flux_g": [1.0, 10.0],
                "flux_z": [10.0, 10.0],
                "fluxerr_g": [0.1, 0.1],
                "fluxerr_z": [0.1, 0.1],
            }
        )
        background = pd.DataFrame(
            {
                "flux_g": [10.0, 1.0],
                "flux_z": [10.0, 10.0],
                "fluxerr_g": [0.1, 0.1],
                "fluxerr_z": [0.1, 0.1],
            }
        )

        split = attach_color_split_columns(pairs, foreground, background)

        self.assertEqual(list(split["color_group"]), [FG_REDDER_GROUP, BG_REDDER_GROUP])
        np.testing.assert_allclose(
            split["foreground_minus_background_g_z"],
            [2.5, -2.5],
        )

    def test_fcolor_stack_matches_weighted_flux_ratio_to_color_conversion(self) -> None:
        result = fcolor_stack([0.2, 0.4], [0.1, 0.1])

        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["raw_ratio"], 0.3)
        self.assertAlmostEqual(result["gz_color"], -2.5 * np.log10(0.3))

    def test_analyze_color_split_bin_reports_delta_and_summary_text(self) -> None:
        pairs = pd.DataFrame(
            {
                "pair_id": ["a", "b", "c", "d"],
                "foreground_index": [0, 1, 2, 3],
                "background_index": [0, 1, 2, 3],
                "theta_arcsec": [2.0, 3.0, 4.0, 5.0],
                "r_perp_kpc": [12.0, 13.0, 16.0, 17.0],
            }
        )
        foreground = pd.DataFrame(
            {
                "flux_g": [1.0, 1.0, 10.0, 10.0],
                "flux_z": [10.0, 10.0, 10.0, 10.0],
                "fluxerr_g": [0.1] * 4,
                "fluxerr_z": [0.1] * 4,
            }
        )
        background = pd.DataFrame(
            {
                "flux_g": [10.0, 10.0, 1.0, 1.0],
                "flux_z": [10.0, 10.0, 10.0, 10.0],
                "fluxerr_g": [0.1] * 4,
                "fluxerr_z": [0.1] * 4,
            }
        )

        result = analyze_color_split_bin(
            pairs,
            foreground,
            background,
            stack_id="test",
            bin_index=0,
            r_min_kpc=10.0,
            r_max_kpc=15.0,
            bootstrap_samples=20,
            permutation_samples=20,
            rng=np.random.default_rng(1),
        )

        self.assertEqual(result["summary"]["n_fg_redder"], 2)
        self.assertEqual(result["summary"]["n_bg_redder"], 2)
        self.assertLess(result["summary"]["delta_gz_mag"], 0)

        comparison = pd.DataFrame(
            [
                {**result["summary"], "bin_number": 1},
                {**result["summary"], "bin_number": 2},
            ]
        )
        self.assertIn("opposite sign", contamination_summary_text(comparison))

    def test_attach_independent_proxy_columns_includes_local_and_metadata_proxies(
        self,
    ) -> None:
        pairs = pd.DataFrame(
            {
                "pair_id": ["a"],
                "foreground_index": [0],
                "background_index": [0],
                "foreground_object_id": [10],
                "background_object_id": [20],
                "theta_arcsec": [2.0],
                "r_perp_kpc": [12.0],
            }
        )
        foreground = pd.DataFrame(
            {
                "flux_g": [1.0],
                "flux_z": [10.0],
                "fluxerr_g": [0.1],
                "fluxerr_z": [0.1],
            }
        )
        background = pd.DataFrame(
            {
                "flux_g": [10.0],
                "flux_z": [5.0],
                "fluxerr_g": [0.1],
                "fluxerr_z": [0.1],
            }
        )
        metadata = pd.DataFrame(
            {
                "objectId": [10, 20],
                "r_deblend_fluxOverlapFraction": [0.1, 0.7],
            }
        )

        out = attach_independent_proxy_columns(
            pairs,
            foreground,
            background,
            object_metadata=metadata,
            overlap_bands=("r",),
        )

        self.assertAlmostEqual(out.loc[0, "foreground_background_z_flux_ratio"], 2.0)
        self.assertAlmostEqual(out.loc[0, "max_deblend_fluxOverlapFraction"], 0.7)
        self.assertTrue(bool(out.loc[0, "valid_background_stack_color"]))

    def test_analyze_proxy_split_bin_reports_high_minus_low_summary(self) -> None:
        pairs = pd.DataFrame(
            {
                "pair_id": ["a", "b", "c", "d"],
                "foreground_index": [0, 1, 2, 3],
                "background_index": [0, 1, 2, 3],
                "theta_arcsec": [2.0, 3.0, 4.0, 5.0],
                "r_perp_kpc": [12.0, 13.0, 16.0, 17.0],
            }
        )
        foreground = pd.DataFrame(
            {
                "flux_g": [1.0, 1.0, 10.0, 10.0],
                "flux_z": [10.0, 10.0, 10.0, 10.0],
                "fluxerr_g": [0.1] * 4,
                "fluxerr_z": [0.1] * 4,
            }
        )
        background = pd.DataFrame(
            {
                "flux_g": [1.0, 1.0, 10.0, 10.0],
                "flux_z": [10.0, 10.0, 10.0, 10.0],
                "fluxerr_g": [0.1] * 4,
                "fluxerr_z": [0.1] * 4,
            }
        )
        pair_inputs = attach_independent_proxy_columns(pairs, foreground, background)
        pair_inputs["proxy_metric"] = [10.0, 9.0, 1.0, 0.0]

        grouped, threshold = attach_proxy_groups(
            pair_inputs,
            metric_col="proxy_metric",
            group_col="proxy_group",
        )
        self.assertEqual(threshold, 5.0)
        self.assertEqual(
            list(grouped["proxy_group"]), [HIGH_GROUP, HIGH_GROUP, LOW_GROUP, LOW_GROUP]
        )

        result = analyze_proxy_split_bin(
            pair_inputs,
            stack_id="test",
            bin_index=0,
            r_min_kpc=10.0,
            r_max_kpc=15.0,
            proxy="proxy_metric",
            metric_col="proxy_metric",
            high_label=HIGH_GROUP,
            low_label=LOW_GROUP,
            bootstrap_samples=20,
            permutation_samples=20,
            rng=np.random.default_rng(2),
        )

        self.assertEqual(result["summary"]["n_high"], 2)
        self.assertEqual(result["summary"]["n_low"], 2)
        self.assertGreater(result["summary"]["delta_high_minus_low_gz_mag"], 0)

        comparison = pd.DataFrame(
            [
                {**result["summary"], "bin_number": 1},
                {**result["summary"], "bin_number": 2},
            ]
        )
        interp = proxy_contamination_summary_table(comparison)
        self.assertIn("Both bins show", interp.loc[0, "interpretation"])

    def test_proxy_signal_normalization_uses_corrected_stack_signal(self) -> None:
        summary = pd.DataFrame(
            {
                "delta_high_minus_low_gz_mag": [0.2, -0.1, 0.5],
                "saved_corrected_gz": [0.05, 0.0, np.nan],
                "saved_corrected_gz_err": [0.1, 0.1, 0.0],
            }
        )

        out = attach_proxy_signal_normalization(summary)

        self.assertAlmostEqual(
            out.loc[0, "delta_over_regular_corrected_gz_signal"], 4.0
        )
        self.assertTrue(np.isnan(out.loc[1, "delta_over_regular_corrected_gz_signal"]))
        self.assertTrue(np.isnan(out.loc[2, "delta_over_regular_corrected_gz_signal"]))
        self.assertAlmostEqual(out.loc[0, "delta_over_regular_corrected_gz_err"], 2.0)
        self.assertAlmostEqual(out.loc[1, "delta_over_regular_corrected_gz_err"], -1.0)
        self.assertTrue(np.isnan(out.loc[2, "delta_over_regular_corrected_gz_err"]))

    def test_proxy_summary_table_reports_normalized_delta(self) -> None:
        comparison = pd.DataFrame(
            [
                {
                    "proxy": "proxy_metric",
                    "bin_number": 1,
                    "n_proxy_valid": 2,
                    "delta_high_minus_low_gz_mag": 0.2,
                    "regular_corrected_gz_signal_mag": 0.05,
                    "regular_corrected_gz_signal_err_mag": 0.1,
                    "delta_over_regular_corrected_gz_signal": 4.0,
                    "delta_over_regular_corrected_gz_err": 2.0,
                    "bootstrap_z": 1.0,
                    "permutation_p_two_sided": 0.2,
                    "expected_contamination_sign_detected": True,
                    "significant_at_2sigma": False,
                    "significant_by_permutation_0p05": False,
                },
                {
                    "proxy": "proxy_metric",
                    "bin_number": 2,
                    "n_proxy_valid": 2,
                    "delta_high_minus_low_gz_mag": 0.1,
                    "regular_corrected_gz_signal_mag": 0.02,
                    "regular_corrected_gz_signal_err_mag": 0.1,
                    "delta_over_regular_corrected_gz_signal": 5.0,
                    "delta_over_regular_corrected_gz_err": 1.0,
                    "bootstrap_z": 1.0,
                    "permutation_p_two_sided": 0.2,
                    "expected_contamination_sign_detected": True,
                    "significant_at_2sigma": False,
                    "significant_by_permutation_0p05": False,
                },
            ]
        )

        interp = proxy_contamination_summary_table(comparison)

        self.assertAlmostEqual(
            interp.loc[0, "bin1_delta_over_regular_corrected_gz_signal"], 4.0
        )
        self.assertAlmostEqual(
            interp.loc[0, "bin1_delta_over_regular_corrected_gz_err"], 2.0
        )
        self.assertIn("delta/sigma_jk=2", interp.loc[0, "interpretation"])
        self.assertIn("delta/E(g-z)=4", interp.loc[0, "interpretation"])

    def test_inner_bin_exclusion_flags_include_overlap_and_manual_labels(
        self,
    ) -> None:
        pairs = pd.DataFrame(
            {
                "pair_id": [
                    "dp1_default_bin1_0001",
                    "dp1_default_bin1_0002",
                    "dp1_default_bin1_0003",
                    "dp1_default_bin1_0004",
                ],
                "valid_background_stack_color": [True, True, True, True],
                "max_deblend_fluxOverlapFraction": [0.0, 1.0, 2.0, 3.0],
                "foreground_parentObjectId": [10, 20, 30, 40],
                "background_parentObjectId": [10, 200, 300, 400],
                "foreground_deblend_failed": [False, False, False, False],
                "background_deblend_failed": [False, True, False, False],
            }
        )
        review_labels = pd.DataFrame(
            {
                "pair_id": ["dp1_default_inner_0004"],
                "blend_review": ["reject"],
            }
        )

        flagged = attach_inner_bin_exclusion_flags(
            pairs,
            review_labels=review_labels,
        )

        self.assertEqual(int(flagged["exclude_overlap_top10pct"].sum()), 1)
        self.assertEqual(int(flagged["exclude_overlap_top20pct"].sum()), 1)
        self.assertTrue(bool(flagged.loc[0, "exclude_same_parent"]))
        self.assertTrue(bool(flagged.loc[1, "exclude_deblend_failed"]))
        self.assertTrue(bool(flagged.loc[3, "exclude_manual_reject"]))
        self.assertEqual(int(flagged["exclude_practical_blend_risk"].sum()), 2)

    def test_summarize_inner_bin_exclusion_rerun_reports_signal_change(self) -> None:
        pairs = pd.DataFrame(
            {
                "valid_background_stack_color": [True, True, True, True],
                "background_flux_ratio_g_z": [0.1, 0.2, 0.4, 0.8],
                "background_flux_ratio_g_z_err": [0.1, 0.1, 0.1, 0.1],
                "exclude_one": [False, False, False, True],
            }
        )
        context = {
            "saved_forward_gz": fcolor_stack(
                pairs["background_flux_ratio_g_z"],
                pairs["background_flux_ratio_g_z_err"],
            )["gz_color"],
            "saved_corrected_gz": 0.5,
            "saved_corrected_gz_err": 0.05,
        }
        scenarios = [
            {
                "scenario": "remove_one",
                "mask_col": "exclude_one",
                "description": "remove one pair",
            }
        ]

        summary = summarize_inner_bin_exclusion_rerun(
            pairs,
            saved_context=context,
            scenarios=scenarios,
        )

        self.assertEqual(int(summary.loc[0, "n_excluded_valid"]), 1)
        self.assertEqual(int(summary.loc[0, "n_kept_valid"]), 3)
        self.assertGreater(summary.loc[0, "approx_corrected_gz_after_exclusion"], 0.5)
        self.assertGreater(
            summary.loc[0, "approx_delta_over_regular_corrected_gz_signal"],
            0.0,
        )
        self.assertGreater(
            summary.loc[0, "approx_delta_over_regular_corrected_gz_err"],
            0.0,
        )

    def test_inner_bin_pair_influence_reports_leave_one_out_shift(self) -> None:
        pairs = pd.DataFrame(
            {
                "pair_id": ["a", "b", "c", "d"],
                "valid_background_stack_color": [True, True, True, True],
                "background_flux_ratio_g_z": [0.1, 0.2, 0.4, 0.8],
                "background_flux_ratio_g_z_err": [0.1, 0.1, 0.1, 0.1],
                "theta_arcsec": [1.0, 2.0, 3.0, 4.0],
                "r_perp_kpc": [10.0, 11.0, 12.0, 13.0],
            }
        )
        context = {
            "saved_corrected_gz": 0.5,
            "saved_corrected_gz_err": 0.05,
        }

        influence = summarize_inner_bin_pair_influence(pairs, saved_context=context)
        summary = summarize_pair_influence_table(influence)

        self.assertEqual(len(influence), 4)
        self.assertEqual(int(summary.loc[0, "n_valid_pairs"]), 4)
        self.assertGreater(summary.loc[0, "max_abs_delta_jackknife_sigma"], 0.0)
        self.assertEqual(
            influence.iloc[0]["abs_delta_over_regular_corrected_gz_err"],
            influence["abs_delta_over_regular_corrected_gz_err"].max(),
        )

    def test_analyze_proxy_split_bin_marks_unavailable_metric_not_significant(
        self,
    ) -> None:
        pairs = pd.DataFrame(
            {
                "pair_id": ["a", "b"],
                "foreground_index": [0, 1],
                "background_index": [0, 1],
                "theta_arcsec": [2.0, 3.0],
                "r_perp_kpc": [12.0, 13.0],
            }
        )
        foreground = pd.DataFrame(
            {
                "flux_g": [1.0, 1.0],
                "flux_z": [10.0, 10.0],
                "fluxerr_g": [0.1, 0.1],
                "fluxerr_z": [0.1, 0.1],
            }
        )
        background = pd.DataFrame(
            {
                "flux_g": [1.0, 1.0],
                "flux_z": [10.0, 10.0],
                "fluxerr_g": [0.1, 0.1],
                "fluxerr_z": [0.1, 0.1],
            }
        )
        pair_inputs = attach_independent_proxy_columns(pairs, foreground, background)

        result = analyze_proxy_split_bin(
            pair_inputs,
            stack_id="test",
            bin_index=0,
            r_min_kpc=10.0,
            r_max_kpc=15.0,
            proxy="overlap",
            metric_col="max_deblend_fluxOverlapFraction",
            high_label=HIGH_GROUP,
            low_label=LOW_GROUP,
            bootstrap_samples=20,
            permutation_samples=20,
            rng=np.random.default_rng(3),
        )

        self.assertEqual(result["summary"]["n_proxy_valid"], 0)
        self.assertTrue(np.isnan(result["summary"]["permutation_p_two_sided"]))
        self.assertFalse(result["summary"]["significant_at_2sigma"])
        self.assertFalse(result["summary"]["significant_by_permutation_0p05"])
        self.assertFalse(result["summary"]["expected_contamination_sign_detected"])

        comparison = pd.DataFrame(
            [
                {**result["summary"], "bin_number": 1},
                {**result["summary"], "bin_number": 2},
            ]
        )
        interp = proxy_contamination_summary_table(comparison)
        self.assertIn("unavailable", interp.loc[0, "interpretation"])


if __name__ == "__main__":
    unittest.main()
