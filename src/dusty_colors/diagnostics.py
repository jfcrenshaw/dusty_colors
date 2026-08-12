"""Pair-weighted histograms of background properties, binned by separation.

These back the diagnostic figures that check whether the background sample
looks different close to foreground galaxies than far from them. A trend in
photo-z or color with separation would mean the measured reddening signal has
a selection component rather than being purely dust.

The functions here work on plain arrays of per-object values, so they carry no
notion of colors, observables, or stack modes. The caller decides what to
histogram and what to name the result.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

# Per-object values plus a mask marking which of them are usable.
ValuesAndMask = tuple[np.ndarray, np.ndarray]


def histogram_bin_edges(
    values: np.ndarray,
    good: np.ndarray,
    bins: int | list[float] | np.ndarray,
    label: str,
) -> np.ndarray:
    """Resolve a bin count or explicit edges into a concrete edge array.

    Parameters
    ----------
    values : numpy.ndarray
        Per-object values the bins must span.
    good : numpy.ndarray
        Boolean mask selecting usable values.
    bins : int or sequence of float
        Either a number of equal-width bins, or explicit edges to validate.
    label : str
        Setting name, used in error messages.

    Returns
    -------
    numpy.ndarray
        Monotonically increasing bin edges, one longer than the bin count.

    Raises
    ------
    ValueError
        If the count is not positive, or explicit edges are too few,
        non-finite, or not increasing.
    """
    if isinstance(bins, int):
        if bins < 1:
            raise ValueError(f"{label} must be positive")
        finite = values[good & np.isfinite(values)]
        # With nothing usable to span, fall back to a unit range so the output
        # array still has the shape downstream plotting expects.
        if len(finite) == 0:
            return np.linspace(0.0, 1.0, bins + 1)
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return np.linspace(0.0, 1.0, bins + 1)
        # All values identical would give zero-width bins, so widen slightly.
        if lo == hi:
            pad = 0.5 if lo == 0 else 0.05 * abs(lo)
            lo -= pad
            hi += pad
        return np.linspace(lo, hi, bins + 1)

    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError(f"{label} must contain at least two edges")
    if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError(f"{label} edges must be finite and increasing")
    return edges


def pair_weighted_histograms(
    *,
    foreground_vectors: np.ndarray,
    background_vectors: np.ndarray,
    foreground_da: np.ndarray,
    values: Mapping[str, ValuesAndMask],
    edges: Mapping[str, np.ndarray],
    radial_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    """Histogram background properties over foreground-background pairs.

    Every pair contributes one count, so a background object near many
    foreground galaxies is counted many times. That is deliberate: the stack
    is itself pair-weighted, so the diagnostic must be weighted the same way
    to describe the sample the estimator actually sees.

    Parameters
    ----------
    foreground_vectors, background_vectors : numpy.ndarray
        Unit position vectors, shape ``(n, 3)``.
    foreground_da : numpy.ndarray
        Angular diameter distance per foreground object, in kpc, converting
        angular separation to projected separation.
    values : Mapping of str to tuple of numpy.ndarray
        Per-background-object values to histogram, each with a mask of which
        entries are usable.
    edges : Mapping of str to numpy.ndarray
        Bin edges per key, matching the keys of `values`.
    radial_edges : numpy.ndarray
        Projected separation bin edges, in kpc.

    Returns
    -------
    dict of str to numpy.ndarray
        Counts per key, each of shape ``(n_radial_bins, n_value_bins)``.
    """
    # Imported lazily so that importing this module does not require scipy.
    from scipy.spatial import cKDTree

    n_radial = len(radial_edges) - 1
    counts = {
        key: np.zeros((n_radial, len(key_edges) - 1), dtype=float)
        for key, key_edges in edges.items()
    }

    background_tree = cKDTree(background_vectors)
    max_r = radial_edges[-1]

    for fg_vector, da in zip(foreground_vectors, foreground_da):
        if not np.isfinite(da) or da <= 0:
            continue
        # The tree indexes 3D unit vectors, so the angular search radius has to
        # become the straight-line chord subtending that angle.
        max_theta = max_r / da
        max_chord = 2.0 * np.sin(0.5 * max_theta)
        neighbors = background_tree.query_ball_point(fg_vector, max_chord)
        if not neighbors:
            continue

        bg_index = np.asarray(neighbors, dtype=int)
        chords = np.linalg.norm(background_vectors[bg_index] - fg_vector, axis=1)
        # Chord back to angle, then to projected separation at this lens.
        theta = 2.0 * np.arcsin(np.clip(0.5 * chords, 0.0, 1.0))
        radius = theta * da
        radial_bin = np.searchsorted(radial_edges, radius, side="right") - 1
        in_range = (radial_bin >= 0) & (radial_bin < n_radial)
        if not np.any(in_range):
            continue

        bg_index = bg_index[in_range]
        radial_bin = radial_bin[in_range]
        for key, key_edges in edges.items():
            key_values, good = values[key]
            good_pair = good[bg_index]
            if not np.any(good_pair):
                continue
            _add_pair_histograms(
                counts[key],
                radial_bin[good_pair],
                key_values[bg_index][good_pair],
                key_edges,
            )

    return counts


def _add_pair_histograms(
    counts: np.ndarray,
    radial_bin: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> None:
    value_bin = np.searchsorted(edges, values, side="right") - 1
    # searchsorted puts values equal to the last edge one bin past the end;
    # fold them back into the final bin so the top of the range is included.
    value_bin[values == edges[-1]] = len(edges) - 2
    good = (value_bin >= 0) & (value_bin < counts.shape[1])
    if np.any(good):
        # np.add.at rather than += so repeated (radial, value) pairs each
        # accumulate instead of overwriting one another.
        np.add.at(counts, (radial_bin[good], value_bin[good]), 1.0)
