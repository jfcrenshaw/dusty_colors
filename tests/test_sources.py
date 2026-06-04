from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from astropy.table import Table

from dusty_colors.sources import load_source_table, read_table


class SourceLoadingTest(unittest.TestCase):
    def test_projected_fits_read_uses_requested_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "objects.fits"
            Table(
                {
                    "ID": [1, 2],
                    "RA": [10.0, 10.1],
                    "DEC": [1.0, 1.1],
                    "MAG_APER_2s_g": [24.0, 24.1],
                }
            ).write(path, overwrite=True)

            table = read_table(
                path,
                columns=["ID"],
                optional_columns=["MAG_APER_2s_g", "MISSING_OPTIONAL"],
            )

            self.assertEqual(list(table.columns), ["ID", "MAG_APER_2s_g"])
            self.assertNotIn("RA", table)

    def test_multi_file_fits_concat_injects_field_metadata(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            path_a = tmp_path / "a.fits"
            path_b = tmp_path / "b.fits"
            Table(
                {
                    "ID": [1, 2],
                    "RA": [10.0, 10.1],
                    "MAG_APER_2s_g": [24.0, 24.1],
                }
            ).write(path_a, overwrite=True)
            Table(
                {
                    "ID": [3],
                    "RA": [20.0],
                }
            ).write(path_b, overwrite=True)

            table = load_source_table(
                {
                    "files": [
                        {"path": path_a, "field": "A"},
                        {"path": path_b, "field": "B"},
                    ],
                    "columns": ["ID", "RA", "field"],
                    "optional_columns": ["MAG_APER_2s_g"],
                }
            )

            self.assertEqual(list(table["field"]), ["A", "A", "B"])
            self.assertEqual(list(table["ID"]), [1, 2, 3])
            np.testing.assert_allclose(
                table["MAG_APER_2s_g"].to_numpy(float),
                [24.0, 24.1, np.nan],
            )


if __name__ == "__main__":
    unittest.main()
