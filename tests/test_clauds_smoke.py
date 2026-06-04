from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from astropy.table import Table

from dusty_colors.catalogs import prepare_catalog
from dusty_colors.selection import prepare_sample

FIELDS = {
    "E-COSMOS": {"ra": 150.1208, "dec": 2.2058},
    "DEEP2-3": {"ra": 352.5000, "dec": 0.0000},
    "ELAIS-N1": {"ra": 242.5000, "dec": 54.0000},
    "XMM-LSS": {"ra": 36.2500, "dec": -4.5000},
}
OPTICAL_BANDS = ["u", "uS", "g", "r", "i", "z", "y"]
NIR_BANDS = ["Yv", "J", "H", "Ks"]


class ClaudsSmokeTest(unittest.TestCase):
    def test_synthetic_optical_and_nir_catalog_sample_workflow(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            field_paths = {}
            for field, center in FIELDS.items():
                path = tmp_path / f"{field}.fits"
                _write_clauds_fixture(
                    path,
                    ra=float(center["ra"]),
                    dec=float(center["dec"]),
                    include_nir=field in {"E-COSMOS", "XMM-LSS"},
                )
                field_paths[field] = path

            optical_catalog_dir = tmp_path / "catalog_optical"
            prepare_catalog(
                _catalog_config(field_paths, fields=list(FIELDS), include_nir=False),
                optical_catalog_dir,
            )
            prepare_sample(
                optical_catalog_dir,
                _sample_config(),
                tmp_path / "sample_optical",
                bands=["u", "g", "r", "i", "z", "y"],
                photometry="flux",
            )
            optical_catalog = pd.read_parquet(optical_catalog_dir / "catalog.parquet")
            optical_fg = pd.read_parquet(tmp_path / "sample_optical/foreground.parquet")
            optical_bg = pd.read_parquet(tmp_path / "sample_optical/background.parquet")

            self.assertEqual(sorted(optical_catalog["field"].unique()), sorted(FIELDS))
            self.assertIn("halo_mass_log", optical_catalog)
            self.assertIn("r200_mpc", optical_catalog)
            self.assertEqual(len(optical_fg), 8)
            self.assertEqual(len(optical_bg), 8)

            nir_catalog_dir = tmp_path / "catalog_nir"
            prepare_catalog(
                _catalog_config(
                    field_paths,
                    fields=["E-COSMOS", "XMM-LSS"],
                    include_nir=True,
                ),
                nir_catalog_dir,
            )
            prepare_sample(
                nir_catalog_dir,
                _sample_config(),
                tmp_path / "sample_nir",
                bands=["u", "g", "r", "i", "z", "y"],
                photometry="flux",
            )
            nir_catalog = pd.read_parquet(nir_catalog_dir / "catalog.parquet")
            nir_fg = pd.read_parquet(tmp_path / "sample_nir/foreground.parquet")

            self.assertEqual(sorted(nir_catalog["field"].unique()), ["E-COSMOS", "XMM-LSS"])
            self.assertIn("mag_Yv", nir_catalog)
            self.assertIn("flux_Yv", nir_catalog)
            self.assertIn("mag_Ks", nir_catalog)
            self.assertIn("mag_Yv", nir_fg)
            self.assertIn("flux_Yv", nir_fg)
            self.assertEqual(len(nir_fg), 4)


def _catalog_config(
    field_paths: dict[str, Path],
    *,
    fields: list[str],
    include_nir: bool,
) -> dict:
    bands = OPTICAL_BANDS + (NIR_BANDS if include_nir else [])
    config = {
        "id": "clauds_smoke",
        "adapter": "clauds_sextractor",
        "primary_source": "objects",
        "sources": {
            "objects": {
                "files": [
                    {"path": field_paths[field], "field": field}
                    for field in fields
                ],
                "columns": [
                    "ID",
                    "RA",
                    "DEC",
                    "field",
                    "ZPHOT",
                    "Z_BEST68_LOW",
                    "Z_BEST68_HIGH",
                    "OBJ_TYPE",
                    "CLEAN",
                    "EB_V",
                    "Z_SPEC",
                    "OFFSET_MAG_2s",
                    *[
                        col
                        for band in bands
                        for col in (f"MAG_APER_2s_{band}", f"MAGERR_APER_2s_{band}")
                    ],
                ],
                "optional_columns": [
                    "MASS_MED",
                    "MASS_MED_6B",
                ],
            }
        },
        "bands": ["u", "g", "r", "i", "z", "y"],
        "photometry": "flux",
        "mag_kind": "APER_2s",
        "extinction": {"enabled": False},
        "enrichments": {
            "kcorrect": {"enabled": False},
            "halo_mass": {
                "enabled": True,
                "stellar_mass_col": "stellar_mass_log",
                "stellar_mass_is_log": True,
                "halo_mass_col": "halo_mass_log",
                "r200_col": "r200_mpc",
            },
        },
        "footprint": {
            "nside": 1024,
            "field_radius_deg": 5.0,
            "fields": {field: FIELDS[field] for field in fields},
        },
    }
    if include_nir:
        config["extra_bands"] = NIR_BANDS
    return config


def _sample_config() -> dict:
    return {
        "selection": {
            "foreground_z": [0.2, 0.5],
            "background_z": [0.7, 1.4],
            "shared_query": "z_phot_err < 0.2",
            "snr_min": {
                "flux_template": "flux_{band}",
                "fluxerr_template": "fluxerr_{band}",
                "bands": {"u": 1, "g": 5, "r": 10, "i": 5, "z": 5, "y": 1},
            },
            "magnitude_limits": [
                {"band": "r", "min": 18.0, "max": 24.0, "flux_col": "flux_r"}
            ],
            "foreground_query": "z_phot_err < 0.1 * (1 + z_phot)",
            "background_query": "z_phot_err < 0.1 * (1 + z_phot)",
        },
        "cleaning": {
            "foreground": {"enabled": False},
            "background": {"enabled": False},
        },
    }


def _write_clauds_fixture(
    path: Path,
    *,
    ra: float,
    dec: float,
    include_nir: bool,
) -> None:
    z = np.array([0.3, 0.4, 0.8, 1.0])
    data = {
        "ID": np.arange(1, 5),
        "RA": ra + np.array([0.0, 0.01, 0.02, 0.03]),
        "DEC": dec + np.array([0.0, 0.01, 0.02, 0.03]),
        "ZPHOT": z,
        "Z_BEST68_LOW": z - 0.02,
        "Z_BEST68_HIGH": z + 0.02,
        "OBJ_TYPE": np.zeros(4, dtype=int),
        "MASK": np.zeros(4, dtype=int),
        "CLEAN": np.ones(4, dtype=int),
        "EB_V": np.full(4, 0.02),
        "Z_SPEC": z + 0.001,
        "MASS_MED": np.full(4, 10.3),
        "MASS_MED_6B": np.full(4, 10.1),
        "OFFSET_MAG_2s": np.full(4, 0.05),
    }
    bands = OPTICAL_BANDS + (NIR_BANDS if include_nir else [])
    for index, band in enumerate(bands):
        base_mag = 22.0 + 0.1 * index
        if band == "r":
            base_mag = 22.6
        data[f"MAG_APER_2s_{band}"] = base_mag + np.array([0.0, 0.1, 0.2, 0.3])
        data[f"MAGERR_APER_2s_{band}"] = np.full(4, 0.02)
    Table(data).write(path, overwrite=True)


if __name__ == "__main__":
    unittest.main()
