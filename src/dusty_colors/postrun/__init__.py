"""Analyses that run automatically after a stack completes.

Each analysis lives in its own module here and registers itself with
``@register``. To add one:

1. Create ``postrun/my_analysis.py`` with a ``@register("my_analysis")``
   function taking ``(context, mode)`` and returning the paths it wrote.
2. Add it to the import list below.

Nothing in ``pipeline.py`` needs to change, and the runner supplies the shared
context, the config lookup, and the error handling.

Options are read from the ``postrun`` block of the analysis YAML, which is
deliberately excluded from the stack config hash so that retuning a fit does
not invalidate the stack it was fitted to.
"""

from __future__ import annotations

from .base import (
    PostRunAnalysis,
    PostRunContext,
    register,
    registered_analyses,
    run_post_run_analyses,
)

# Importing a stage module is what registers it, and the order of these imports
# is the order the analyses run in. Kept explicit rather than auto-discovered so
# that the order is obvious and mypy can see the modules.
from . import analysis_stats  # noqa: F401,E402  isort:skip
from . import figures  # noqa: F401,E402  isort:skip
from . import dust_extinction_fit  # noqa: F401,E402  isort:skip
from . import color_power_law_fit  # noqa: F401,E402  isort:skip
from . import chromaticity  # noqa: F401,E402  isort:skip

__all__ = [
    "PostRunAnalysis",
    "PostRunContext",
    "register",
    "registered_analyses",
    "run_post_run_analyses",
]
