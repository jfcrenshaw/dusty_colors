from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from dusty_colors.catalogs import (
    ClaudsSExtractorCatalogAdapter,
    DP1CatalogAdapter,
    prepare_catalog,
    validate_canonical_schema,
)
from dusty_colors.cleaning import (
    add_diagnostic_columns,
    apply_minimal_cleaning,
    clean_sample,
)
from dusty_colors.enrichments import (
    KcorrectEnrichment,
    apply_enrichments,
    apply_kcorrect,
)
from dusty_colors.observables import (
    add_observable_columns,
    flux_ratio_observable,
    magnitude_color_observable,
)
from dusty_colors.selection import select_samples, write_sample_outputs
from dusty_colors.treecorr_stacker import TreeCorrStacker
from astropy.cosmology import Planck18 as cosmo

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None


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
                "i_blendedness": [0.1, 0.2],
                "refExtendedness": [0.9, 0.2],
                "r_cModelFlux": [12.0, 22.0],
                "r_cModelFluxErr": [1.2, 2.2],
                "g_gaap1p0Flux": [10.0, 20.0],
                "g_gaap1p0FluxErr": [1.0, 2.0],
                "r_gaap1p0Flux": [5.0, 10.0],
                "r_gaap1p0FluxErr": [0.5, 1.0],
                "g5_pixel": [26.1, 26.2],
            }
        )

        catalog = DP1CatalogAdapter({"bands": ["g", "r"], "photometry": "flux"}).adapt(
            raw
        )

        validate_canonical_schema(catalog, bands=["g", "r"], photometry="flux")
        self.assertEqual(list(catalog["object_id"]), [10, 11])
        self.assertEqual(list(catalog["is_galaxy"]), [True, False])
        np.testing.assert_allclose(catalog["blendedness_i"], [0.1, 0.2])
        np.testing.assert_allclose(catalog["cmodel_flux_r"], [12.0, 22.0])
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
                [10.0 * 10**0.04, 20.0],
            )
            np.testing.assert_allclose(
                catalog["fluxerr_g"],
                [1.0 * 10**0.04, 2.0],
            )
            np.testing.assert_allclose(catalog["spec_z"].iloc[0], 0.31)
            self.assertTrue(np.isnan(catalog["spec_z"].iloc[1]))
            self.assertTrue((tmp_path / "out/footprint.parquet").exists())
            self.assertIn("photoz_sigma_fzboost", catalog)
            self.assertIn("photoz_sigma_lephare", catalog)
            self.assertIn("z_phot_diff", catalog)

    def test_prepare_catalog_assigns_footprint_without_sample_cuts(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            objects = pd.DataFrame(
                {
                    "objectID": [1, 2],
                    "coord_ra": [53.0, 55.0],
                    "coord_dec": [-28.0, -28.0],
                    "z_phot": [0.3, 0.4],
                    "z_phot_err": [0.03, 0.04],
                    "refExtendedness": [1.0, 1.0],
                    "g_gaap1p0Flux": [10.0, 20.0],
                    "g_gaap1p0FluxErr": [1.0, 2.0],
                    "r_gaap1p0Flux": [5.0, 10.0],
                    "r_gaap1p0FluxErr": [0.5, 1.0],
                    "r_cModelFlux": [5.0, 10.0],
                    "r_cModelFluxErr": [0.1, 100.0],
                }
            )
            object_path = tmp_path / "objects.parquet"
            objects.to_parquet(object_path, index=False)

            prepare_catalog(
                {
                    "id": "dp1_depth_test",
                    "adapter": "dp1",
                    "primary_source": "objects",
                    "sources": {"objects": {"path": object_path}},
                    "bands": ["g", "r"],
                    "photometry": "flux",
                    "footprint": {
                        "nside": 1024,
                        "fields": {
                            "ECDFS": {"ra": 53.13, "dec": -28.10},
                        },
                    },
                },
                tmp_path / "out",
            )
            catalog = pd.read_parquet(tmp_path / "out/catalog.parquet")
            footprint = pd.read_parquet(tmp_path / "out/footprint.parquet")

            self.assertEqual(list(catalog["object_id"]), [1, 2])
            self.assertIn("pixel", catalog)
            self.assertEqual(list(catalog["field"]), ["ECDFS", "ECDFS"])
            self.assertEqual(len(footprint), 2)
            self.assertIn("pixel", footprint)

    def test_clauds_adapter_rejects_non_picouet_photometry(self) -> None:
        raw = pd.DataFrame(
            {
                "ID": [1, 2],
                "RA": [150.0, 150.2],
                "DEC": [2.1, 2.2],
                "field": ["E-COSMOS", "E-COSMOS"],
                "ZPHOT": [0.4, 1.0],
                "Z_BEST68_LOW": [0.35, 0.9],
                "Z_BEST68_HIGH": [0.45, 1.1],
                "OBJ_TYPE": [0, 2],
                "CLEAN": [1, 0],
                "EB_V": [0.01, 0.02],
                "Z_SPEC": [0.41, -99.0],
                "OFFSET_MAG_2s": [0.1, 0.2],
                "MASS_MED": [10.1, 9.5],
                "HSC_g_MAG_APER_2s": [24.0, 24.5],
                "HSC_g_MAGERR_APER_2s": [0.03, 0.04],
            }
        )

        with self.assertRaisesRegex(ValueError, "MAG_APER_2s_g"):
            ClaudsSExtractorCatalogAdapter(
                {"bands": ["g"], "photometry": "flux"}
            ).adapt(raw)

    def test_clauds_adapter_maps_picouet_columns_and_u_fallback(self) -> None:
        raw = pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "RA": [150.0, 150.2, 150.4],
                "DEC": [2.1, 2.2, 2.3],
                "field": ["E-COSMOS"] * 3,
                "ZPHOT": [0.4, 1.0, 0.8],
                "Z_BEST68_LOW": [0.35, 0.9, 0.75],
                "Z_BEST68_HIGH": [0.45, 1.1, 0.85],
                "OBJ_TYPE": [0, 0, 2],
                "CLEAN": [1, 0, 1],
                "EB_V": [0.01, 0.02, 0.03],
                "Z_SPEC": [0.41, -99.0, 0.82],
                "MASS_MED": [10.1, np.nan, 9.8],
                "MASS_MED_6B": [10.0, 9.4, 9.7],
                "OFFSET_MAG_2s": [0.1, 0.2, 0.3],
                "MAG_APER_2s_u": [np.nan, 24.0, np.nan],
                "MAGERR_APER_2s_u": [np.nan, 0.04, np.nan],
                "MAG_APER_2s_uS": [23.5, 23.0, 22.0],
                "MAGERR_APER_2s_uS": [0.03, 0.03, 0.03],
                "MAG_APER_2s_g": [24.0, 24.5, 25.0],
                "MAGERR_APER_2s_g": [0.03, 0.04, 0.05],
                "MAG_APER_2s_r": [23.0, 23.5, 24.0],
                "MAGERR_APER_2s_r": [0.02, 0.02, 0.02],
            }
        )
        config = {
            "bands": ["u", "g", "r"],
            "photometry": "flux",
            "mag_kind": "APER_2s",
        }

        catalog = ClaudsSExtractorCatalogAdapter(config).adapt(raw)

        def flux_from_mag(mag: float) -> float:
            return 10 ** ((31.4 - mag) / 2.5)

        def fluxerr_from_magerr(mag: float, magerr: float) -> float:
            return flux_from_mag(mag) * np.log(10.0) / 2.5 * magerr

        validate_canonical_schema(catalog, bands=["u", "g", "r"], photometry="flux")
        validate_canonical_schema(catalog, bands=["u", "g", "r"], photometry="mag")
        np.testing.assert_allclose(catalog["mag_u"], [23.6, 24.2, 22.3])
        np.testing.assert_allclose(catalog["mag_g"], [24.1, 24.7, 25.3])
        np.testing.assert_allclose(catalog["mag_r"], [23.1, 23.7, 24.3])
        np.testing.assert_allclose(
            catalog["flux_g"],
            [flux_from_mag(mag) for mag in [24.1, 24.7, 25.3]],
        )
        np.testing.assert_allclose(
            catalog["fluxerr_g"],
            [
                fluxerr_from_magerr(mag, magerr)
                for mag, magerr in zip([24.1, 24.7, 25.3], [0.03, 0.04, 0.05])
            ],
        )
        self.assertEqual(list(catalog["mask_ok"]), [True, False, True])
        np.testing.assert_allclose(catalog["ebv"], [0.01, 0.02, 0.03])
        np.testing.assert_allclose(catalog["spec_z"], [0.41, -99.0, 0.82])
        np.testing.assert_allclose(catalog["stellar_mass_log"], [10.1, 9.4, 9.8])

        fallback = ClaudsSExtractorCatalogAdapter(config).adapt(
            raw.drop(columns=["MASS_MED"])
        )
        np.testing.assert_allclose(
            fallback["stellar_mass_log"],
            [10.0, 9.4, 9.7],
        )

    def test_clauds_adapter_handles_6band_missing_optional_columns(self) -> None:
        raw = pd.DataFrame(
            {
                "ID": [1, 2, 3],
                "RA": [242.0, 242.1, 242.2],
                "DEC": [54.0, 54.1, 54.2],
                "field": ["ELAIS-N1"] * 3,
                "ZPHOT": [0.4, 0.8, 1.0],
                "Z_BEST68_LOW": [0.35, 0.75, 0.95],
                "Z_BEST68_HIGH": [0.45, 0.85, 1.05],
                "OBJ_TYPE": [0, 0, 0],
                "MASK": [0, 1, 0],
                "CLEAN": [np.nan, False, True],
                "EB_V": [0.01, 0.02, 0.03],
                "OFFSET_MAG_2s": [0.1, 0.2, 0.3],
                "MAG_APER_2s_g": [24.0, 24.5, 25.0],
                "MAGERR_APER_2s_g": [0.03, 0.04, 0.05],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FutureWarning)
            catalog = ClaudsSExtractorCatalogAdapter(
                {"bands": ["g"], "photometry": "flux", "mag_kind": "APER_2s"}
            ).adapt(raw)

        self.assertEqual(caught, [])

        self.assertEqual(list(catalog["mask_ok"]), [True, False, True])
        self.assertTrue(np.isnan(catalog["spec_z"]).all())
        self.assertTrue(np.isnan(catalog["stellar_mass_log"]).all())

        without_clean = ClaudsSExtractorCatalogAdapter(
            {"bands": ["g"], "photometry": "flux", "mag_kind": "APER_2s"}
        ).adapt(raw.drop(columns=["CLEAN"]))
        self.assertEqual(list(without_clean["mask_ok"]), [True, False, True])

    def test_treecorr_stacker_ignores_analysis_only_stack_options(self) -> None:
        kwargs = TreeCorrStacker._init_kwargs(
            {
                "colors": ["g-r"],
                "random_seed": 11,
                "dust_extinction_fit": {"fixed_rv": 3.1},
            }
        )

        self.assertEqual(kwargs, {"colors": ["g-r"], "random_seed": 11})

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

    def test_selection_applies_paper_style_optional_cuts(self) -> None:
        def flux_from_mag(mag: float) -> float:
            return 10 ** ((31.4 - mag) / 2.5)

        catalog = pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4, 5, 6],
                "ra": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "dec": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "field": ["A"] * 6,
                "z_phot": [0.3, 0.35, 0.4, 0.45, 0.9, 1.0],
                "z_phot_err": [0.03] * 6,
                "photoz_sigma_fzboost": [0.04, 0.11, 0.04, 0.04, 0.04, 0.04],
                "photoz_sigma_lephare": [0.05] * 6,
                "z_phot_diff": [0.03, 0.03, 0.2, 0.03, 0.03, 0.03],
                "blendedness_i": [0.1, 0.1, 0.1, 0.5, 0.1, 0.1],
                "is_galaxy": [True] * 6,
                "mask_ok": [True] * 6,
                "quality_ok": [True] * 6,
                "flux_g": [10.0] * 6,
                "fluxerr_g": [1.0] * 6,
                "flux_r": [5.0] * 6,
                "fluxerr_r": [1.0] * 6,
                "cmodel_flux_r": [
                    flux_from_mag(23.0),
                    flux_from_mag(23.0),
                    flux_from_mag(23.0),
                    flux_from_mag(23.0),
                    flux_from_mag(23.5),
                    flux_from_mag(24.5),
                ],
            }
        )
        config = {
            "selection": {
                "foreground_z": [0.2, 0.5],
                "background_z": [0.7, 1.4],
                "photoz_max_sigma": 0.1,
                "photoz_estimate_max_sigma": 0.1,
                "photoz_max_diff_norm": 0.1,
                "blendedness_max": {
                    "column": "blendedness_i",
                    "value": 0.42,
                },
                "magnitude_limits": [
                    {
                        "band": "r",
                        "max": 24.0,
                        "flux_col": "cmodel_flux_r",
                    }
                ],
            }
        }

        samples = select_samples(catalog, config, bands=["g", "r"], photometry="flux")

        self.assertEqual(list(samples["foreground"]["object_id"]), [1])
        self.assertEqual(list(samples["background"]["object_id"]), [5])

    def test_selection_applies_depth_snr_cuts_and_writes_footprint(
        self,
    ) -> None:
        def flux_from_mag(mag: float) -> float:
            return 10 ** ((31.4 - mag) / 2.5)

        depth25_err = 10 ** ((31.4 - 25.0) / 2.5) / 10.0
        catalog = pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4],
                "ra": [1.0, 2.0, 3.0, 4.0],
                "dec": [0.0, 0.0, 0.0, 0.0],
                "field": ["A"] * 4,
                "pixel": [10, 10, 10, 11],
                "z_phot": [0.3, 0.35, 0.9, 1.0],
                "z_phot_err": [0.03] * 4,
                "is_galaxy": [True] * 4,
                "mask_ok": [True] * 4,
                "quality_ok": [True] * 4,
                "flux_g": [10.0] * 4,
                "fluxerr_g": [1.0] * 4,
                "flux_r": [5.0] * 4,
                "fluxerr_r": [1.0] * 4,
                "cmodel_flux_g": [1000.0, 2.0, 1000.0, 1000.0],
                "cmodel_fluxerr_g": [depth25_err] * 4,
                "cmodel_flux_r": [flux_from_mag(23.0)] * 4,
                "blendedness_i": [0.1] * 4,
            }
        )
        config = {
            "selection": {
                "foreground_z": [0.2, 0.5],
                "background_z": [0.7, 1.4],
                "pixel_depth_cuts": {
                    "fluxerr_template": "cmodel_fluxerr_{band}",
                    "depth_sigma": 10,
                    "aggregate": "median",
                    "valid_range": {"bands": ["g"], "min": 20.0, "max": 30.0},
                    "min_occupancy": 2,
                    "complete_to": {"band": "g", "magnitude": 24.0},
                },
                "snr_min": {
                    "flux_template": "cmodel_flux_{band}",
                    "fluxerr_template": "cmodel_fluxerr_{band}",
                    "bands": {"g": 5},
                },
                "blendedness_max": {"column": "blendedness_i", "value": 0.42},
                "magnitude_limits": [
                    {
                        "band": "r",
                        "min": 18.0,
                        "max": 24.0,
                        "flux_col": "cmodel_flux_r",
                    }
                ],
            },
            "footprint": {
                "enabled": True,
                "columns": ["object_id", "ra", "dec", "pixel"],
            },
        }

        samples = select_samples(catalog, config, bands=["g", "r"], photometry="flux")

        self.assertEqual(list(samples["foreground"]["object_id"]), [1])
        self.assertEqual(list(samples["background"]["object_id"]), [3])
        self.assertEqual(list(samples["footprint"]["object_id"]), [1, 3])

    def test_drop_shallowest_combines_band_masks_before_dropping(self) -> None:
        def fluxerr_from_depth(depth: float) -> float:
            return 10 ** ((31.4 - depth) / 2.5) / 10.0

        depths = np.arange(21.0, 31.0)
        depth_fluxerrs = [fluxerr_from_depth(depth) for depth in depths]
        catalog = pd.DataFrame(
            {
                "object_id": list(range(1, 11)),
                "ra": np.arange(10.0),
                "dec": np.zeros(10),
                "pixel": list(range(100, 110)),
                "z_phot": [0.3] * 5 + [0.9] * 5,
                "is_galaxy": [True] * 10,
                "mask_ok": [True] * 10,
                "quality_ok": [True] * 10,
                "flux_g": [10.0] * 10,
                "fluxerr_g": [1.0] * 10,
                "flux_r": [10.0] * 10,
                "fluxerr_r": [1.0] * 10,
                "cmodel_fluxerr_g": depth_fluxerrs,
                "cmodel_fluxerr_r": depth_fluxerrs,
            }
        )
        config = {
            "selection": {
                "foreground_z": [0.2, 0.5],
                "background_z": [0.7, 1.4],
                "pixel_depth_cuts": {
                    "fluxerr_template": "cmodel_fluxerr_{band}",
                    "depth_sigma": 10,
                    "drop_shallowest": {
                        "bands": ["g", "r"],
                        "fraction": 0.2,
                        "unique_pixels": True,
                    },
                },
            },
        }

        samples = select_samples(catalog, config, bands=["g", "r"], photometry="flux")

        self.assertEqual(list(samples["foreground"]["object_id"]), [3, 4, 5])
        self.assertEqual(
            list(samples["background"]["object_id"]),
            [6, 7, 8, 9, 10],
        )

    def test_selection_can_reassign_jackknife_after_sample_cuts(self) -> None:
        catalog = pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4],
                "ra": [0.0, 1.0, 2.0, 3.0],
                "dec": [-1.0, 1.0, -1.0, 1.0],
                "field": ["A", "A", "A", "A"],
                "pixel": [10, 10, 11, 11],
                "jackknife_region": [99, 99, 99, 99],
                "z_phot": [0.3, 0.4, 0.8, 0.9],
                "z_phot_err": [0.03] * 4,
                "is_galaxy": [True] * 4,
                "mask_ok": [True] * 4,
                "quality_ok": [True] * 4,
                "flux_g": [10.0] * 4,
                "fluxerr_g": [1.0] * 4,
                "flux_r": [5.0] * 4,
                "fluxerr_r": [1.0] * 4,
            }
        )
        config = {
            "selection": {
                "foreground_z": [0.2, 0.5],
                "background_z": [0.7, 1.4],
            },
            "footprint": {
                "enabled": True,
                "columns": ["object_id", "pixel", "jackknife_region"],
            },
            "jackknife": {"regions_per_field": 2},
        }

        samples = select_samples(catalog, config, bands=["g", "r"], photometry="flux")

        self.assertEqual(list(samples["foreground"]["jackknife_region"]), [0, 1])
        self.assertEqual(list(samples["background"]["jackknife_region"]), [0, 1])
        self.assertEqual(list(samples["footprint"]["jackknife_region"]), [0, 1, 0, 1])

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

    def test_disabled_clean_sample_is_no_op(self) -> None:
        catalog = pd.DataFrame(
            {
                "object_id": [1, 2],
                "z_phot": [0.2, np.nan],
                "flux_g": [10.0, 999.0],
                "color_gr": [0.3, 99.0],
            }
        )
        config = {
            "enabled": False,
            "finite_columns": ["missing_when_disabled"],
            "redshift_trend": {
                "redshift_col": "missing_when_disabled",
                "columns": ["color_gr"],
                "degree": 1,
                "output_suffix": "_z_detrended",
            },
            "isolation_forest": {
                "columns": ["missing_when_disabled"],
                "contamination": 0.5,
                "random_state": 11,
            },
        }

        cleaned = clean_sample(catalog, config)

        pd.testing.assert_frame_equal(cleaned, catalog)

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_isolation_forest_cleaning_removes_outlier_deterministically(self) -> None:
        catalog = pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4, 5, 99],
                "feature_x": [0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
                "feature_y": [0.0, 0.0, 0.0, 0.0, 0.0, 12.0],
            }
        )
        config = {
            "enabled": True,
            "isolation_forest": {
                "columns": ["feature_x", "feature_y"],
                "contamination": 1.0 / 6.0,
                "random_state": 7,
                "n_estimators": 64,
                "max_samples": 6,
            },
        }

        cleaned = clean_sample(catalog, config)

        self.assertEqual(list(cleaned["object_id"]), [1, 2, 3, 4, 5])

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is not installed")
    def test_column_isolation_forest_masks_values_without_dropping_rows(self) -> None:
        catalog = pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4, 5, 99],
                "z_phot": [0.8, 0.82, 0.84, 0.86, 0.88, 0.9],
                "flux_g": [10.0, 10.1, 9.9, 10.2, 9.8, 200.0],
                "flux_r": [5.0, 5.1, 4.9, 5.2, 4.8, 5.0],
            }
        )
        config = {
            "enabled": True,
            "column_isolation_forest": {
                "columns": ["flux_g"],
                "redshift_col": "z_phot",
                "contamination": 1.0 / 6.0,
                "random_state": 7,
                "n_estimators": 64,
                "max_samples": 6,
                "scale": "none",
            },
        }

        cleaned = clean_sample(catalog, config)

        self.assertEqual(list(cleaned["object_id"]), [1, 2, 3, 4, 5, 99])
        self.assertTrue(np.isnan(cleaned.loc[5, "flux_g"]))
        self.assertEqual(cleaned.loc[5, "flux_r"], 5.0)

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is not installed")
    def test_column_redshift_trend_overwrites_flux_without_dropping_rows(self) -> None:
        redshift = np.linspace(0.7, 1.35, 80)
        flux_g = 50.0 * (1.0 + redshift)
        catalog = pd.DataFrame(
            {
                "object_id": np.arange(len(redshift)),
                "z_phot": redshift,
                "flux_g": flux_g,
                "flux_r": np.full(len(redshift), 5.0),
            }
        )
        config = {
            "enabled": True,
            "column_redshift_trend": {
                "columns": ["flux_g"],
                "redshift_col": "z_phot",
                "bin_width": 0.04,
                "mode": "flux",
                "overwrite": True,
            },
        }

        cleaned = clean_sample(catalog, config)

        self.assertEqual(list(cleaned["object_id"]), list(catalog["object_id"]))
        self.assertNotIn("flux_g_z_detrended", cleaned)
        np.testing.assert_allclose(cleaned["flux_r"], catalog["flux_r"])
        original_slope = np.polyfit(redshift, flux_g, 1)[0]
        cleaned_slope = np.polyfit(redshift, cleaned["flux_g"], 1)[0]
        self.assertLess(abs(cleaned_slope), 0.05 * abs(original_slope))
        np.testing.assert_allclose(
            np.nanmedian(cleaned["flux_g"]),
            np.nanmedian(flux_g),
            rtol=1.0e-3,
        )

    def test_redshift_trend_adds_detrended_columns_without_overwriting(self) -> None:
        catalog = pd.DataFrame(
            {
                "object_id": [1, 2, 3, 4, 5],
                "z_phot": [0.1, 0.2, 0.3, np.nan, 0.4],
                "color_gr": [1.2, 1.4, 1.6, 9.0, 1.8],
                "flux_g": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        original = catalog.copy(deep=True)
        config = {
            "enabled": True,
            "redshift_trend": {
                "redshift_col": "z_phot",
                "columns": ["color_gr", "flux_g"],
                "degree": 1,
                "output_suffix": "_z_detrended",
            },
        }

        cleaned = clean_sample(catalog, config)

        self.assertIn("color_gr_z_detrended", cleaned)
        self.assertIn("flux_g_z_detrended", cleaned)
        pd.testing.assert_series_equal(cleaned["color_gr"], original["color_gr"])
        pd.testing.assert_series_equal(cleaned["flux_g"], original["flux_g"])
        finite_color = np.isfinite(original["z_phot"]) & np.isfinite(
            original["color_gr"]
        )
        np.testing.assert_allclose(
            cleaned.loc[finite_color, "color_gr_z_detrended"],
            np.zeros(int(finite_color.sum())),
            atol=1e-12,
        )
        self.assertTrue(np.isnan(cleaned.loc[3, "color_gr_z_detrended"]))
        self.assertTrue(
            np.isfinite(cleaned.loc[finite_color, "flux_g_z_detrended"]).all()
        )

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

    def test_kcorrect_enrichment_skips_nonpositive_redshifts(self) -> None:
        class FakeKcorrect:
            redshifts: list[np.ndarray] = []

            def __init__(self, *, responses: list[str]) -> None:
                self.responses = responses

            def fit_coeffs(
                self,
                *,
                redshift: np.ndarray,
                maggies: np.ndarray,
                ivar: np.ndarray,
            ) -> np.ndarray:
                self.redshifts.append(np.asarray(redshift, dtype=float).copy())
                return np.ones((len(redshift), len(self.responses)))

            def absmag(
                self,
                *,
                redshift: np.ndarray,
                maggies: np.ndarray,
                ivar: np.ndarray,
                coeffs: np.ndarray,
            ) -> np.ndarray:
                return np.full((len(redshift), len(self.responses)), -20.0)

            def derived(
                self,
                *,
                redshift: np.ndarray,
                coeffs: np.ndarray,
            ) -> dict[str, np.ndarray]:
                return {"mremain": np.full(len(redshift), 1.0e10)}

        package = types.ModuleType("kcorrect")
        package.__path__ = []
        module = types.ModuleType("kcorrect.kcorrect")
        module.Kcorrect = FakeKcorrect
        catalog = pd.DataFrame(
            {
                "z_phot": [0.0, 0.1, -0.1, np.nan],
                "flux_u": [10.0, 10.0, 10.0, 10.0],
                "fluxerr_u": [1.0, 1.0, 1.0, 1.0],
            }
        )

        with patch.dict(
            sys.modules,
            {"kcorrect": package, "kcorrect.kcorrect": module},
        ):
            enriched = apply_kcorrect(
                catalog,
                {
                    "responses": ["u"],
                    "response_bands": ["u"],
                    "absmag_bands": ["u"],
                },
            )

        np.testing.assert_allclose(FakeKcorrect.redshifts[0], [0.1])
        self.assertTrue(np.isnan(enriched.loc[0, "absmag_u"]))
        self.assertEqual(enriched.loc[1, "absmag_u"], -20.0)
        self.assertTrue(np.isnan(enriched.loc[2, "stellar_mass_log"]))
        self.assertTrue(np.isnan(enriched.loc[3, "stellar_mass_log"]))

    def test_kcorrect_response_paths_are_normalized_to_stems(self) -> None:
        responses = KcorrectEnrichment.resolve_response_names(
            [
                "data/bandpasses/bandpass_u_v1p9.dat",
                "sdss_g0",
            ]
        )

        self.assertTrue(responses[0].endswith("data/bandpasses/bandpass_u_v1p9"))
        self.assertFalse(responses[0].endswith(".dat"))
        self.assertTrue(Path(responses[0]).is_absolute())
        self.assertEqual(responses[1], "sdss_g0")

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

    @unittest.skipUnless(SCIPY_AVAILABLE, "scipy is not installed")
    def test_treecorr_stacker_builds_pair_distribution_diagnostics(self) -> None:
        da = cosmo.angular_diameter_distance([0.3]).to_value("kpc")[0]
        background_ra = np.rad2deg(np.array([50.0, 500.0]) / da)
        foreground = pd.DataFrame(
            {
                "ra": [0.0],
                "dec": [0.0],
                "z_phot": [0.3],
            }
        )
        background = pd.DataFrame(
            {
                "ra": background_ra,
                "dec": [0.0, 0.0],
                "z_phot": [0.8, 1.2],
                "flux_g": [10.0, 10.0],
                "fluxerr_g": [1.0, 1.0],
                "flux_r": [10.0, 5.0],
                "fluxerr_r": [1.0, 0.5],
            }
        )
        stacker = TreeCorrStacker(
            foreground=foreground,
            background=background,
            out_dir="unused",
            colors=("g-r",),
            modes=("fcolors",),
            r_bin_edges=[5.0, 100.0, 1000.0],
            jackknife=False,
            random_correction=False,
            flipped_correction=False,
            diagnostic_photoz_bins=[0.0, 1.0, 2.0],
            diagnostic_color_bins=[-2.0, -1.0, 0.0, 1.0],
        )

        stacker._load_samples()
        arrays = stacker._diagnostic_arrays("fcolors", stacker._radial_bins())

        np.testing.assert_allclose(
            arrays["diagnostic_photoz_counts"],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        np.testing.assert_allclose(
            arrays["g-r_diagnostic_color_counts"],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        )

    def test_treecorr_stacker_can_disable_flipped_correction(self) -> None:
        def profile(
            color: list[float],
            err: list[float],
            ref_color: float,
            ref_err: float,
        ) -> types.SimpleNamespace:
            color_array = np.asarray(color, dtype=float)
            err_array = np.asarray(err, dtype=float)
            return types.SimpleNamespace(
                raw=color_array,
                raw_err=err_array,
                color=color_array,
                color_err=err_array,
                weight=np.ones_like(color_array),
                npairs=np.ones_like(color_array),
                ref_raw=ref_color,
                ref_raw_err=ref_err,
                ref_color=ref_color,
                ref_color_err=ref_err,
                ref_npairs=1.0,
                corrs=[],
            )

        stacker = TreeCorrStacker(
            foreground=pd.DataFrame(),
            background=pd.DataFrame(),
            out_dir="unused",
            flipped_correction=False,
        )
        bins = [
            types.SimpleNamespace(center=10.0),
            types.SimpleNamespace(center=20.0),
        ]
        forward = profile([1.0, 2.0], [0.1, 0.2], 0.5, 0.05)
        random_forward = profile([0.2, 0.4], [0.02, 0.04], 0.1, 0.01)

        result, provenance = stacker._result(
            "g-i",
            bins,
            "mcolors",
            forward,
            None,
            random_forward,
            None,
        )

        np.testing.assert_allclose(provenance["g-i_delta_avg"], [0.8, 1.6])
        self.assertEqual(float(provenance["g-i_ref_avg"]), 0.4)
        np.testing.assert_allclose(result["g-i_avg"], [0.4, 1.2])
        np.testing.assert_allclose(provenance["g-i_uncorrected_avg"], [0.5, 1.5])
        self.assertEqual(float(result["g-i_forward_ref_avg"]), 0.5)
        self.assertEqual(float(provenance["g-i_forward_ref_raw_avg"]), 0.5)
        self.assertEqual(float(result["g-i_random_forward_ref_avg"]), 0.1)
        self.assertEqual(float(provenance["g-i_random_forward_ref_raw_avg"]), 0.1)
        self.assertNotIn("g-i_flipped_avg", result)
        self.assertNotIn("g-i_flipped_ref_avg", result)
        self.assertNotIn("g-i_flipped_ref_raw_avg", provenance)
        self.assertNotIn("g-i_random_flipped_avg", result)
        self.assertNotIn("g-i_random_flipped_ref_avg", result)
        self.assertNotIn("g-i_random_flipped_ref_raw_avg", provenance)

    def test_treecorr_stacker_saves_component_apertures_for_plot_variants(
        self,
    ) -> None:
        def profile(
            color: list[float],
            err: list[float],
            ref_color: float,
            ref_err: float,
        ) -> types.SimpleNamespace:
            color_array = np.asarray(color, dtype=float)
            err_array = np.asarray(err, dtype=float)
            return types.SimpleNamespace(
                raw=color_array,
                raw_err=err_array,
                color=color_array,
                color_err=err_array,
                weight=np.ones_like(color_array),
                npairs=np.ones_like(color_array),
                ref_raw=ref_color,
                ref_raw_err=ref_err,
                ref_color=ref_color,
                ref_color_err=ref_err,
                ref_npairs=1.0,
                corrs=[],
            )

        stacker = TreeCorrStacker(
            foreground=pd.DataFrame(),
            background=pd.DataFrame(),
            out_dir="unused",
            flipped_correction=True,
        )
        bins = [
            types.SimpleNamespace(center=10.0),
            types.SimpleNamespace(center=20.0),
        ]
        forward = profile([10.0, 20.0], [1.0, 2.0], 1.0, 0.1)
        flipped = profile([3.0, 5.0], [0.3, 0.5], 0.5, 0.05)
        random_forward = profile([2.0, 4.0], [0.2, 0.4], 0.25, 0.025)
        random_flipped = profile([0.5, 1.0], [0.05, 0.1], 0.125, 0.0125)

        result, provenance = stacker._result(
            "g-i",
            bins,
            "mcolors",
            forward,
            flipped,
            random_forward,
            random_flipped,
        )

        uncorrected = result["g-i_forward_avg"] - result["g-i_forward_ref_avg"]
        flip_corrected = (
            result["g-i_forward_avg"]
            - result["g-i_flipped_avg"]
            - (result["g-i_forward_ref_avg"] - result["g-i_flipped_ref_avg"])
        )
        random_corrected = (
            result["g-i_forward_avg"]
            - result["g-i_random_forward_avg"]
            - (
                result["g-i_forward_ref_avg"]
                - result["g-i_random_forward_ref_avg"]
            )
        )
        flip_and_random_corrected = (
            result["g-i_forward_avg"]
            - result["g-i_flipped_avg"]
            - result["g-i_random_forward_avg"]
            + result["g-i_random_flipped_avg"]
            - (
                result["g-i_forward_ref_avg"]
                - result["g-i_flipped_ref_avg"]
                - result["g-i_random_forward_ref_avg"]
                + result["g-i_random_flipped_ref_avg"]
            )
        )

        np.testing.assert_allclose(uncorrected, [9.0, 19.0])
        np.testing.assert_allclose(flip_corrected, [6.5, 14.5])
        np.testing.assert_allclose(random_corrected, [7.25, 15.25])
        np.testing.assert_allclose(flip_and_random_corrected, result["g-i_avg"])
        np.testing.assert_allclose(result["g-i_avg"], [5.125, 11.625])
        self.assertEqual(float(provenance["g-i_random_flipped_ref_raw_avg"]), 0.125)


if __name__ == "__main__":
    unittest.main()
