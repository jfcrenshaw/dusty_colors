"""Paper appendix diagnostics, kept at a lower maintenance tier than the core.

The modules here support the inner-bin contamination and blend-review appendices.
They are still used to produce paper material, so they must keep working, but they
are deliberately excluded from linting and type checking and are not refactored
alongside the core package.

Do not import from here in `dusty_colors` core modules.
The dependency runs one way only: appendix code may use the core, never the reverse.
"""
