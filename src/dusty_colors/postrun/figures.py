"""Standard stack figures as a registered post-run analysis.

The plotting itself stays in :mod:`dusty_colors.plotting`, which notebooks
import directly and which owns the packaged Matplotlib style. This module is
only the adapter that runs it after a stack.
"""

from __future__ import annotations

from pathlib import Path

from ..plotting import save_stack_diagnostic_figures, save_stack_figures
from .base import PostRunContext, register


@register("figures")
def _stage(context: PostRunContext, mode: str) -> tuple[Path, ...]:
    """Write the science figures plus any radial-bin diagnostic figures.

    The diagnostic figures need ``stack_<mode>_diagnostics.npz``, which only
    exists when the stack ran with ``diagnostic_plots`` enabled;
    ``save_stack_diagnostic_figures`` returns an empty tuple when it is absent,
    so no separate check is needed here.
    """

    results = context.results(mode)
    paths = list(save_stack_figures(results, context.stack_dir, root=context.root))
    paths.extend(
        save_stack_diagnostic_figures(results, context.stack_dir, root=context.root)
    )
    return tuple(paths)
