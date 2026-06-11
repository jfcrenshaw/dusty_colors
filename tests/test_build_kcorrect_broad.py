from __future__ import annotations

from astropy.io import fits
from contextlib import redirect_stdout
import importlib.util
import io
import numpy as np
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_kcorrect_broad.py"

spec = importlib.util.spec_from_file_location("build_kcorrect_broad", SCRIPT)
build_kcorrect_broad = importlib.util.module_from_spec(spec)
sys.modules["build_kcorrect_broad"] = build_kcorrect_broad
assert spec.loader is not None
spec.loader.exec_module(build_kcorrect_broad)


class BuildKcorrectBroadTest(unittest.TestCase):
    def test_default_args_build_griz_config(self) -> None:
        args = types.SimpleNamespace(
            band=None,
            tag="v1.9.1",
            templates="data/kcorrect/templates_broad.fits",
            output="data/kcorrect/kcorrect_broad.fits",
            redshift_range=(0.0, 0.5),
            nredshift=1000,
            abcorrect=False,
            interpolate_templates=False,
            force=False,
        )

        config = build_kcorrect_broad.config_from_args(args)

        self.assertEqual(config.bands, ("g", "r", "i", "z"))
        self.assertEqual(len(config.responses), 4)
        self.assertTrue(config.responses[0].endswith("rubin_bandpass_g_v1.9.1"))
        self.assertTrue(config.output.is_absolute())

    def test_invalid_bands_grid_and_tag_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate Rubin band"):
            build_kcorrect_broad.normalize_bands(["g", "g"])
        with self.assertRaisesRegex(ValueError, "Unknown Rubin band"):
            build_kcorrect_broad.normalize_bands(["q"])
        with self.assertRaisesRegex(ValueError, "path separators"):
            build_kcorrect_broad.validate_tag("../bad")

        args = types.SimpleNamespace(
            band=["g"],
            tag="v1.9.1",
            templates="templates.fits",
            output="model.fits",
            redshift_range=(0.5, 0.0),
            nredshift=1000,
            abcorrect=False,
            interpolate_templates=False,
            force=False,
        )
        with self.assertRaisesRegex(ValueError, "MIN must be less than MAX"):
            build_kcorrect_broad.config_from_args(args)
        args.redshift_range = (0.0, 0.5)
        args.nredshift = 1
        with self.assertRaisesRegex(ValueError, "at least 2"):
            build_kcorrect_broad.config_from_args(args)

    def test_build_response_file_requires_force_and_writes_atomically(self) -> None:
        class FakeTemplate:
            calls: list[tuple[str, bool]] = []

            def __init__(self, *, filename: str, interpolate: bool) -> None:
                self.calls.append((filename, interpolate))

        class FakeKcorrect:
            calls: list[dict[str, object]] = []

            def __init__(
                self,
                *,
                responses: list[str],
                templates: FakeTemplate,
                redshift_range: list[float],
                nredshift: int,
                abcorrect: bool,
            ) -> None:
                del templates
                self.responses = responses
                self.calls.append(
                    {
                        "responses": responses,
                        "redshift_range": redshift_range,
                        "nredshift": nredshift,
                        "abcorrect": abcorrect,
                    }
                )

            def tofits(self, filename: str) -> None:
                column = fits.Column(
                    name="responses",
                    format="200A",
                    array=np.asarray(self.responses, dtype="S200"),
                )
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        fits.BinTableHDU.from_columns([column], name="RESPONSES"),
                    ]
                ).writeto(filename, overwrite=True)

        package = types.ModuleType("kcorrect")
        package.__path__ = []
        template_module = types.ModuleType("kcorrect.template")
        template_module.Template = FakeTemplate
        kcorrect_module = types.ModuleType("kcorrect.kcorrect")
        kcorrect_module.Kcorrect = FakeKcorrect

        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates_broad.fits"
            templates.write_text("fake templates\n")
            responses = []
            for band in ["g", "r"]:
                path = tmp_path / f"rubin_bandpass_{band}.dat"
                path.write_text("|   lambda |        pass |\n")
                responses.append(str(path.with_suffix("")))
            output = tmp_path / "kcorrect_broad.fits"
            output.write_text("old model\n")
            config = build_kcorrect_broad.BuildConfig(
                templates=templates,
                output=output,
                responses=responses,
                bands=("g", "r"),
                redshift_range=(0.0, 0.5),
                nredshift=1000,
                abcorrect=False,
                interpolate_templates=False,
                force=False,
            )

            with self.assertRaises(FileExistsError):
                build_kcorrect_broad.build_response_file(config)

            forced = build_kcorrect_broad.BuildConfig(
                templates=templates,
                output=output,
                responses=responses,
                bands=("g", "r"),
                redshift_range=(0.0, 0.5),
                nredshift=1000,
                abcorrect=False,
                interpolate_templates=False,
                force=True,
            )
            with patch.dict(
                sys.modules,
                {
                    "kcorrect": package,
                    "kcorrect.template": template_module,
                    "kcorrect.kcorrect": kcorrect_module,
                },
            ):
                build_kcorrect_broad.build_response_file(forced)

            with fits.open(output) as hdul:
                saved = [str(value[0]) for value in hdul["RESPONSES"].data]
            self.assertEqual(saved, responses)
            self.assertEqual(FakeTemplate.calls, [(str(templates), False)])
            self.assertEqual(FakeKcorrect.calls[0]["responses"], responses)
            self.assertEqual(list(tmp_path.glob("*.tmp")), [])

    def test_main_reports_existing_output_refusal_on_stdout(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            templates = tmp_path / "templates_broad.fits"
            output = tmp_path / "kcorrect_broad.fits"
            templates.write_text("fake templates\n")
            output.write_text("old model\n")
            stdout = io.StringIO()

            with patch.object(
                sys,
                "argv",
                [
                    "build_kcorrect_broad.py",
                    "--templates",
                    str(templates),
                    "--output",
                    str(output),
                ],
            ):
                with redirect_stdout(stdout):
                    with self.assertRaises(SystemExit) as caught:
                        build_kcorrect_broad.main()

            self.assertEqual(caught.exception.code, 1)
            text = stdout.getvalue()
            self.assertIn("Refusing to replace existing output", text)
            self.assertIn("Run again with --force", text)


if __name__ == "__main__":
    unittest.main()
