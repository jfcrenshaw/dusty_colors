"""Tests for repository root resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from dusty_colors import get_root
from dusty_colors.paths import ROOT_MARKER

REPO_ROOT = Path(__file__).resolve().parents[1]


class TemporaryRepo:
    """A temporary directory tree, optionally containing the root marker."""

    def __init__(self, *, marker: bool = True) -> None:
        self._marker = marker
        self._tmp: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name).resolve()
        if self._marker:
            (root / ROOT_MARKER).write_text("", encoding="utf-8")
        return root

    def __exit__(self, *exc_info: object) -> None:
        assert self._tmp is not None
        self._tmp.cleanup()


class GetRootTest(unittest.TestCase):
    def test_finds_marker_from_nested_directory(self) -> None:
        """A deeply nested start still resolves to the directory holding the marker."""
        with TemporaryRepo() as root:
            nested = root / "a" / "b" / "c"
            nested.mkdir(parents=True)
            self.assertEqual(get_root(nested), root)

    def test_returns_start_when_marker_is_alongside(self) -> None:
        with TemporaryRepo() as root:
            self.assertEqual(get_root(root), root)

    def test_accepts_a_file_and_searches_from_its_directory(self) -> None:
        with TemporaryRepo() as root:
            nested = root / "pkg"
            nested.mkdir()
            module = nested / "thing.py"
            module.write_text("", encoding="utf-8")
            self.assertEqual(get_root(module), root)

    def test_defaults_to_the_working_directory(self) -> None:
        with TemporaryRepo() as root:
            nested = root / "deep"
            nested.mkdir()
            previous = Path.cwd()
            os.chdir(nested)
            try:
                self.assertEqual(get_root(), root)
            finally:
                os.chdir(previous)

    def test_raises_when_no_marker_is_found(self) -> None:
        """Failing loudly beats returning a wrong root that breaks much later."""
        with TemporaryRepo(marker=False) as root:
            with self.assertRaises(FileNotFoundError) as caught:
                get_root(root)
            self.assertIn(ROOT_MARKER, str(caught.exception))

    def test_resolves_this_repository(self) -> None:
        self.assertEqual(get_root(__file__), REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
