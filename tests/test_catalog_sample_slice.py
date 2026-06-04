from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from dusty_colors.catalogs import (
    adapt_clauds_sextractor_catalog,
    adapt_dp1_catalog,
    prepare_catalog,
    validate_canonical_schema,
)
from dusty_colors.cleaning import add_diagnostic_columns, apply_minimal_cleaning
from dusty_colors.enrichments import apply_enrichments
from dusty_colors.observables import (
    add_observable_columns,
    flux_ratio_observable,
    magnitude_color_observable,
)
from dusty_colors.selection import select_samples, write_sample_outputs
from dusty_colors.treecorr_stacker import TreeCorrStacker
from astropy.cosmology import Planck18 as cosmo


class CatalogSampleSliceTest(unittest.TestCase):
    def test_dp1_adapter_maps_flux_photometry_and_schema(self) -> None:
        raw = pd.DataFrame(
            {
                "objectID": [10, 11],
                "coord_ra": [53.0, 53.1],
                "coord_dec": [-28.0, -28.1],
                "field": ["ECDFS", "ECDFS"],
                "z_phot": [0.3, 0.8],
                "z_phot_err": [0.03, 0.04],
                "refExtendedness": [0.9, 0.2],
                "g_gaap1p0Flux": [10.0, 20.0],
                "g_gaap1p0FluxErr": [1.0, 2.0],
                "r_gaap1p0Flux": [5.0, 10.0],
                "r_gaap1p0FluxErr": [0.5, 1.0],
                "g5_pixel": [26.1, 26.2],
            }
        )

        catalog = adapt_dp1_catalog(raw, {"bands": ["g", "r"], "photometry": "flux"})

        validate_canonical_schema(catalog, bands=["g", "r"], photometry="flux")
        self.assertEqual(list(catalog["object_id"]), [10, 11])
        self.assertEqual(list(catalog["is_galaxy"]), [True, False])
        self.assertIn("depth5_g", catalog)
        np.testing.assert_allclose(catalog["flux_g"], raw["g_gaap1p0Flux"])

    def test_prepare_catalog_assembles_sources_and_applies_extinction(self) -> None:
        with TemporaryDirectory() as tmp:
            from pathlib import Path

            tmp_path = Path(tmp)
            objects = pd.DataFrame(
                {
                    "objectID": [1, 2],
                    "coord_ra": [53.0, 53.1],
                    "coord_dec": [-28.0, -28.1],
                    "ebv": [0.1, 0.0],
                    "refExtendedness": [0.9, 0.8],
                    "g_gaap1p0Flux": [10.0, 20.0],
                    "g_gaap1p0FluxErr": [1.0, 2.0],
                    "r_gaap1p0Flux": [5.0, 10.0],
                    "r_gaap1p0FluxErr": [0.5, 1.0],
                }
            )
            photoz = pd.DataFrame(
                {
                    "objectId": [1, 2],
                    "fzboost_z_mode": [0.3, 0.8],
                    "fzboost_z_err68_low": [0.26, 0.74],
                    "fzboost_z_err68_high": [0.34, 0.86],
                    "lephare_z_mode": [0.32, 0.82],
                    "lephare_z_err68_low": [0.26, 0.76],
                    "lephare_z_err68_high": [0.38, 0.88],
                }
            )
            specz = pd.DataFrame(
                {
                    "objectId": [1, 2],
                    "redshift": [0.31, np.nan],
                    "confidence": [0.99, 0.99],
                    "type": ["galaxy", "galaxy"],
                    "source": ["ref", "ref"],
                }
            )
            object_path = tmp_path / "objects.parquet"
            photoz_path = tmp_path / "photoz.parquet"
            specz_path = tmp_path / "specz.parquet"
            objects.to_parquet(object_path, index=False)
            photoz.to_parquet(photoz_path, index=False)
            specz.to_parquet(specz_path, index=False)

            config = {
                "id": "dp1_test",
                "adapter": "dp1",
                "primary_source": "objects",
                "sources": {
                    "objects": {"path": object_path},
                    "photoz": {
                        "path": photoz_path,
                        "join": {
                            "left_key": "objectID",
                            "right_key": "objectId",
                            "how": "inner",
                        },
                    },
                    "specz": {
                        "path": specz_path,
                        "query": "confidence >= 0.95",
                        "finite": ["redshift"],
                        "columns": [
                            "objectId",
                            "redshift",
                            "confidence",
                            "type",
                            "source",
                        ],
                        "join": {
                            "left_key": "objectID",
                            "right_key": "objectId",
                            "how": "left",
                        },
                    },
                },
                "bands": ["g", "r"],
                "photometry": "flux",
                "photoz": {
                    "combine": {
                        "estimates": [
                            {
                                "z": "fzboost_z_mode",
                                "err_low": "fzboost_z_err68_low",
                                "err_high": "fzboost_z_err68_high",
                            },
                            {
                                "z": "lephare_z_mode",
                                "err_low": "lephare_z_err68_low",
                                "err_high": "lephare_z_err68_high",
                            },
                        ]
                    }
                },
                "extinction": {
                    "enabled": True,
                    "ebv_column": "ebv",
                    "bands": ["g"],
                    "coefficients": {"g": 1.0},
                },
            }

            prepare_catalog(config, tmp_path / "out")
            catalog = pd.read_parquet(tmp_path / "out/catalog.parquet")

            self.assertEqual(list(catalog["object_id"]), [1, 2])
            np.testing.assert_allclose(
                catalog["flux_g"],
                [10.0 * 10 ** 0.04, 20.0],
            )
            np.testing.assert_allclose(
                catalog["fluxerr_g"],
                [1.0 * 10 ** 0.04, 2.0],
            )
            np.testing.assert_allclose(catalog["spec_z"].iloc[0], 0.31)
            self.assertTrue(np.isnan(catalog["spec_z"].iloc[1]))
            self.assertTrue((tmp_path / "out/footprint.parquet").exists())

    def test_clauds_adapter_maps_source_extractor_magnitudes(self) -> None:
        raw = pd.DataFrame(
            {
                "ID": [1, 2],
                "RA": [150.0, 150.2],
                "DEC": [2.1, 2.2],
                "ZPHOT": [0.4, 1.0],
                "ZPDF_L68": [0.35, 0.9],
                "ZPDF_U68": [0.45, 1.1],
                "OBJ_TYPE": [0, 2],
                "MASK": [0, 1],
                "MASS_MED": [10.1, 9.5],
                "HSC_g_MAG_APER_2s": [24.0, 24.5],
                "HSC_g_MAGERR_APER_2s": [0.03, 0.04],
                "HSC_r_MAG_APER_2s": [23.5, 24.1],
                "HSC_r_MAGERR_APER_2s": [0.02, 0.03],
            }
        )

        catalog = adapt_clauds_sextractor_catalog(
            raw, {"bands": ["g", "r"], "photometry": "mag"}
        )

        validate_canonical_schema(catalog, bands=["g", "r"], photometry="mag")
        np.testing.assert_allclose(catalog["z_phot_err"], [0.05, 0.1])
        self.assertEqual(list(catalog["is_galaxy"]), [True, False])
        self.assertEqual(list(catalog["mask_ok"]), [True, False])
        np.testing.assert_allclose(catalog["stellar_mass_log"], [10.1, 9.5])

    def test_selection_applies_minimal_masks_and_redshift_windows(self) -> None:
        catalog = pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4],
                "ra": [1.0, 2.0, 3.0, 4.0],
                "dec": [0.0, 0.0, 0.0, 0.0],
                "field": ["A", "A", "A", "A"],
                "z_phot": [0.3, 0.9, 0.4, 1.1],
                "z_phot_err": [0.03, 0.04, 0.2, 0.03],
                "is_galaxy": [True, True, True, True],
                "mask_ok": [True, True, True, False],
                "quality_ok": [True, True, True, True],
                "flux_g": [10.0, 20.0, 30.0, 40.0],
                "fluxerr_g": [1.0, 1.0, 1.0, 1.0],
                "flux_r": [5.0, 10.0, 15.0, 20.0],
                "fluxerr_r": [1.0, 1.0, 1.0, 1.0],
            }
        )
        config = {
            "selection": {
                "foreground_z": [0.2, 0.5],
                "background_z": [0.7, 1.4],
                "photoz_max_sigma": 0.1,
                "shared_query": "mask_ok and quality_ok",
            }
        }

        samples = select_samples(catalog, config, bands=["g", "r"], photometry="flux")

        self.assertEqual(list(samples["foreground"]["object_id"]), [1])
        self.assertEqual(list(samples["background"]["object_id"]), [2])

    def test_sample_outputs_write_foreground_and_background(self) -> None:
        with TemporaryDirectory() as tmp:
            samples = {
                "foreground": pd.DataFrame({"object_id": [1]}),
                "background": pd.DataFrame({"object_id": [2]}),
            }

            write_sample_outputs(samples, tmp)

            from pathlib import Path

            self.assertTrue((Path(tmp) / "foreground.parquet").exists())
            self.assertTrue((Path(tmp) / "background.parquet").exists())

    def test_cleaning_and_observables_do_not_overwrite_raw_photometry(self) -> None:
        catalog = pd.DataFrame(
            {
                "object_id": [1],
                "ra": [1.0],
                "dec": [0.0],
                "field": ["A"],
                "z_phot": [0.3],
                "z_phot_err": [0.03],
                "is_galaxy": [True],
                "mask_ok": [True],
                "quality_ok": [True],
                "flux_g": [10.0],
                "fluxerr_g": [1.0],
                "flux_r": [5.0],
                "fluxerr_r": [0.5],
            }
        )

        cleaned = apply_minimal_cleaning(catalog, bands=["g", "r"], photometry="flux")
        diagnostics = add_diagnostic_columns(cleaned, bands=["g", "r"])
        with_observable = add_observable_columns(diagnostics, "g-r", "fcolors")

        np.testing.assert_allclose(with_observable["flux_g"], [10.0])
        np.testing.assert_allclose(with_observable["fluxerr_g"], [1.0])
        self.assertIn("diagnostic_snr_g", with_observable)
        self.assertIn("fcolor_g_r", with_observable)

    def test_observable_error_propagation_from_fluxes(self) -> None:
        catalog = pd.DataFrame(
            {
                "flux_g": [10.0],
                "fluxerr_g": [1.0],
                "flux_r": [5.0],
                "fluxerr_r": [0.5],
            }
        )

        ratio, ratio_err = flux_ratio_observable(catalog, "g-r")
        color, color_err = magnitude_color_observable(catalog, "g-r")

        np.testing.assert_allclose(ratio, [2.0])
        np.testing.assert_allclose(ratio_err, [2.0 * np.sqrt(0.1**2 + 0.1**2)])
        np.testing.assert_allclose(color, [-2.5 * np.log10(2.0)])
        np.testing.assert_allclose(
            color_err, [2.5 / np.log(10) * np.sqrt(0.1**2 + 0.1**2)]
        )

    def test_halo_mass_enrichment_uses_stellar_mass_log(self) -> None:
        catalog = pd.DataFrame(
            {
                "stellar_mass_log": [10.0, np.nan],
                "z_phot": [0.3, 0.3],
            }
        )

        enriched = apply_enrichments(
            catalog,
            {
                "enrichments": {
                    "halo_mass": {
                        "enabled": True,
                        "stellar_mass_col": "stellar_mass_log",
                    }
                }
            },
        )

        self.assertIn("halo_mass_log", enriched)
        self.assertIn("r200_mpc", enriched)
        self.assertTrue(np.isfinite(enriched.loc[0, "halo_mass_log"]))
        self.assertTrue(np.isfinite(enriched.loc[0, "r200_mpc"]))
        self.assertTrue(np.isnan(enriched.loc[1, "halo_mass_log"]))

    def test_treecorr_stacker_defaults_to_kpc_radii(self) -> None:
        foreground = pd.DataFrame(
            {
                "ra": [53.0],
                "dec": [-28.0],
                "z_phot": [0.3],
                "flux_g": [10.0],
                "fluxerr_g": [1.0],
                "flux_r": [5.0],
                "fluxerr_r": [0.5],
            }
        )
        background = foreground.drop(columns=["z_phot"])
        stacker = TreeCorrStacker(
            foreground=foreground,
            background=background,
            out_dir="unused",
            colors=("g-r",),
            modes=("fcolors",),
            jackknife=False,
            random_correction=False,
        )

        np.testing.assert_allclose(
            stacker.r_bin_edges,
            np.geomspace(5.0, 1000.0, 6),
        )
        self.assertEqual(stacker.reference_annulus, (2000.0, 4000.0))
        stacker._load_samples()
        np.testing.assert_allclose(
            stacker._foreground_da,
            cosmo.angular_diameter_distance([0.3]).to_value("kpc"),
        )


if __name__ == "__main__":
    unittest.main()
