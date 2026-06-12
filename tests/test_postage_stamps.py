from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from astropy.cosmology import Planck18 as cosmo
import numpy as np
import pandas as pd

from dusty_colors.postage_stamps import (
    ARCSEC_PER_RADIAN,
    build_object_metadata_query,
    build_projected_pair_table,
    pair_table_summary,
    requested_blend_columns,
    score_pair_blend_risk,
    select_existing_columns,
    write_contact_sheet_html,
    write_review_labels_template,
)


class PostageStampPairsTest(unittest.TestCase):
    def test_build_projected_pair_table_uses_foreground_angular_distance(self) -> None:
        redshift = 0.3
        da_kpc = cosmo.angular_diameter_distance(redshift).to_value("kpc")
        theta_inside = 12.0 / da_kpc
        theta_outside = 20.0 / da_kpc
        foreground = pd.DataFrame(
            {
                "object_id": [101],
                "ra": [0.0],
                "dec": [0.0],
                "z_phot": [redshift],
                "field": ["test"],
            }
        )
        background = pd.DataFrame(
            {
                "object_id": [201, 202],
                "ra": [np.rad2deg(theta_inside), np.rad2deg(theta_outside)],
                "dec": [0.0, 0.0],
                "z_phot": [0.8, 0.9],
                "field": ["test", "test"],
            }
        )

        pairs = build_projected_pair_table(foreground, background)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.loc[0, "foreground_object_id"], 101)
        self.assertEqual(pairs.loc[0, "background_object_id"], 201)
        self.assertAlmostEqual(pairs.loc[0, "r_perp_kpc"], 12.0, places=8)
        self.assertAlmostEqual(
            pairs.loc[0, "theta_arcsec"],
            theta_inside * ARCSEC_PER_RADIAN,
            places=8,
        )
        self.assertAlmostEqual(
            pairs.loc[0, "midpoint_ra"], 0.5 * np.rad2deg(theta_inside), places=8
        )

    def test_pair_table_summary_reports_quantiles(self) -> None:
        pair_table = pd.DataFrame(
            {"theta_arcsec": [1.0, 2.0, 3.0], "r_perp_kpc": [10.0, 12.0, 15.0]}
        )

        summary = pair_table_summary(pair_table, quantiles=(0.0, 0.5, 1.0))

        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["theta_arcsec"], {"min": 1.0, "q50": 2.0, "max": 3.0})
        self.assertEqual(summary["r_perp_kpc"], {"min": 10.0, "q50": 12.0, "max": 15.0})


class PostageStampTapTest(unittest.TestCase):
    def test_requested_columns_are_filtered_to_available_tap_columns(self) -> None:
        requested = requested_blend_columns(("r",))
        available = {"objectId", "detect_fromBlend", "r_blendedness"}

        selected = select_existing_columns(requested, available)

        self.assertEqual(selected, ["objectId", "detect_fromBlend", "r_blendedness"])

    def test_build_object_metadata_query_rejects_unsafe_identifiers(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsafe ADQL identifier"):
            build_object_metadata_query([1], ["objectId", "bad;column"])

    def test_build_object_metadata_query_formats_sorted_unique_ids(self) -> None:
        query = build_object_metadata_query([3, 1, 3], ["objectId", "r_blendedness"])

        self.assertEqual(
            query,
            "SELECT objectId, r_blendedness FROM dp1.Object WHERE objectId IN (1, 3)",
        )


class PostageStampBlendRiskTest(unittest.TestCase):
    def test_score_pair_blend_risk_prioritizes_same_parent_pairs(self) -> None:
        pair_table = pd.DataFrame(
            {
                "pair_id": ["pair_b", "pair_a"],
                "foreground_object_id": [10, 20],
                "background_object_id": [11, 21],
                "theta_arcsec": [2.0, 1.0],
                "r_perp_kpc": [12.0, 11.0],
            }
        )
        metadata = pd.DataFrame(
            {
                "objectId": [10, 11, 20, 21],
                "parentObjectId": [99, 99, 0, 0],
                "detect_isIsolated": [False, False, True, True],
                "detect_fromBlend": [True, True, False, False],
                "deblend_failed": [False, False, False, False],
                "r_blendedness": [0.2, 0.3, 0.01, 0.02],
                "i_deblend_fluxOverlapFraction": [0.0, 0.0, 0.0, 0.0],
            }
        )

        review = score_pair_blend_risk(pair_table, metadata, bands=("r", "i"))

        self.assertEqual(list(review["pair_id"]), ["pair_b", "pair_a"])
        self.assertTrue(bool(review.loc[0, "same_parent"]))
        self.assertEqual(review.loc[0, "blend_risk"], "high")
        self.assertEqual(review.loc[1, "blend_risk"], "low")

    def test_review_outputs_are_written(self) -> None:
        review = pd.DataFrame(
            {
                "pair_id": ["pair_1"],
                "foreground_object_id": [1],
                "background_object_id": [2],
                "theta_arcsec": [2.5],
                "r_perp_kpc": [12.0],
                "same_parent": [False],
                "blend_risk": ["low"],
                "max_blendedness": [0.01],
                "max_overlap_fraction": [0.0],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = write_review_labels_template(
                review, tmp_path / "review_labels.csv"
            )
            html = write_contact_sheet_html(
                review,
                tmp_path / "report.html",
                image_dir=tmp_path / "stamps",
            )

            self.assertTrue(labels.exists())
            self.assertTrue(html.exists())
            self.assertIn("pair_1", html.read_text())


if __name__ == "__main__":
    unittest.main()
