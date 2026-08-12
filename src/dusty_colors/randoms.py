"""Random catalogs for the TreeCorr stack, and their depth-matched weights.

The estimator subtracts a stack measured around random positions to remove
footprint and depth-driven artefacts. This module builds those random catalogs
inside the sample footprint, and optionally reweights them so their local depth
distribution matches the real sample.

Everything here is a plain function taking explicit arguments, so the random
machinery can be read and tested without constructing a stacker.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import healpy as hp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RandomCatalogs:
    """Random foreground and background catalogs with their weights.

    Attributes
    ----------
    foreground, background : pandas.DataFrame
        Random positions carrying ``ra``, ``dec``, ``pixel`` and the TreeCorr
        patch column.
    foreground_da : numpy.ndarray or None
        Angular diameter distances resampled from the real foreground, or None
        when no distances were supplied.
    foreground_weight, background_weight : numpy.ndarray
        Per-object weights, all ones unless depth matching is configured.
    """

    foreground: pd.DataFrame
    background: pd.DataFrame
    foreground_da: np.ndarray | None
    foreground_weight: np.ndarray
    background_weight: np.ndarray


def build_random_catalogs(
    *,
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    footprint: pd.DataFrame | None,
    foreground_da: np.ndarray | None,
    patch_col: str,
    treecorr_patch_col: str,
    use_jackknife: bool,
    multiplier: float,
    seed: int,
    nside: int,
    weighting: Mapping[str, Any] | bool | None,
    colors: tuple[str, ...],
) -> RandomCatalogs:
    """Build random catalogs matching the sample footprint, patch by patch.

    Parameters
    ----------
    foreground, background : pandas.DataFrame
        The real samples, already carrying the TreeCorr patch column.
    footprint : pandas.DataFrame or None
        Allowed HEALPix pixels, requiring a ``pixel`` column.
    foreground_da : numpy.ndarray or None
        Angular diameter distances for the real foreground, resampled onto the
        randoms so they inherit the same redshift distribution.
    patch_col : str
        Jackknife region column in the samples and footprint.
    treecorr_patch_col : str
        Internal contiguous patch index column used by TreeCorr.
    use_jackknife : bool
        Whether patches are in play; when False everything is one patch.
    multiplier : float
        Random objects per real object, per patch.
    seed : int
        Seed for the random position draw.
    nside : int
        HEALPix nside for footprint pixel lookups.
    weighting : Mapping, bool or None
        Depth-matching configuration; see :func:`sample_weighting_config`.
    colors : tuple of str
        Configured colors, used to infer bands when depth matching does not
        name them explicitly.

    Returns
    -------
    RandomCatalogs
        The random catalogs and their weights.

    Raises
    ------
    ValueError
        If the footprint is missing, lacks required columns, the multiplier is
        not positive, or a sample patch has no footprint pixels.
    """
    if footprint is None:
        raise ValueError("Random correction requires a prepared footprint table")
    if "pixel" not in footprint:
        raise ValueError("Random correction requires footprint column 'pixel'")
    if multiplier <= 0:
        raise ValueError("random_multiplier must be positive")

    if use_jackknife:
        if patch_col not in footprint:
            raise ValueError(
                f"Random correction requires footprint column '{patch_col}'"
            )
        patch_map = dict(
            zip(
                foreground[patch_col].to_numpy(int),
                foreground[treecorr_patch_col].to_numpy(int),
            )
        )
        patch_map.update(
            zip(
                background[patch_col].to_numpy(int),
                background[treecorr_patch_col].to_numpy(int),
            )
        )
        footprint = footprint[footprint[patch_col].isin(patch_map)]
        footprint = footprint.assign(
            **{
                treecorr_patch_col: [
                    patch_map[patch] for patch in footprint[patch_col].to_numpy(int)
                ]
            }
        )
    else:
        footprint = footprint.assign(**{treecorr_patch_col: 0})

    pixels_by_patch = {
        patch: np.unique(group["pixel"].dropna().to_numpy(int))
        for patch, group in footprint.groupby(treecorr_patch_col)
    }
    pixels_by_patch = {
        patch: pixels for patch, pixels in pixels_by_patch.items() if len(pixels)
    }
    missing = set(foreground[treecorr_patch_col].unique()) - set(pixels_by_patch)
    missing |= set(background[treecorr_patch_col].unique()) - set(pixels_by_patch)
    if missing:
        raise ValueError(f"Missing random-footprint pixels for patches: {missing}")

    # One generator drives both draws, foreground first, so the stream is
    # reproducible for a given seed.
    rng = np.random.default_rng(seed)
    random_foreground, random_foreground_da = _random_catalog_like(
        foreground,
        foreground_da,
        pixels_by_patch,
        rng,
        multiplier=multiplier,
        nside=nside,
        treecorr_patch_col=treecorr_patch_col,
    )
    random_background, _ = _random_catalog_like(
        background,
        None,
        pixels_by_patch,
        rng,
        multiplier=multiplier,
        nside=nside,
        treecorr_patch_col=treecorr_patch_col,
    )

    foreground_weight, background_weight = _build_weights(
        foreground=foreground,
        background=background,
        random_foreground=random_foreground,
        random_background=random_background,
        weighting=weighting,
        colors=colors,
        treecorr_patch_col=treecorr_patch_col,
    )

    print(
        "   TreeCorr random correction using "
        f"{len(random_foreground)} foreground and "
        f"{len(random_background)} background random positions"
    )
    return RandomCatalogs(
        foreground=random_foreground,
        background=random_background,
        foreground_da=random_foreground_da,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
    )


def sample_weighting_config(
    weighting: Mapping[str, Any] | bool | None,
    sample: str,
) -> dict[str, Any] | None:
    """Resolve the depth-matching configuration for one sample.

    Top-level keys apply to both samples; a ``foreground`` or ``background``
    block overrides them, and either may be switched off independently.

    Parameters
    ----------
    weighting : Mapping, bool or None
        The ``random_weighting`` stack setting.
    sample : {'foreground', 'background'}
        Which sample to resolve.

    Returns
    -------
    dict or None
        The merged configuration, or None when weighting is off for it.

    Raises
    ------
    ValueError
        If the configuration is neither a mapping nor a boolean.
    """
    config = weighting
    if config is None or config is False:
        return None
    if config is True:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("random_weighting must be a mapping or boolean")
    if not bool(config.get("enabled", True)):
        return None

    base = {
        str(key): value
        for key, value in config.items()
        if key not in {"foreground", "background"}
    }
    sample_config = config.get(sample)
    if sample_config is None:
        return dict(base)
    if sample_config is False:
        return None
    if sample_config is True:
        return dict(base)
    if not isinstance(sample_config, Mapping):
        raise ValueError(f"random_weighting.{sample} must be a mapping")
    merged = dict(base)
    merged.update(sample_config)
    if not bool(merged.get("enabled", True)):
        return None
    return merged


def bands_for_colors(colors: tuple[str, ...]) -> list[str]:
    """Return the distinct bands appearing in the configured colors."""
    bands: list[str] = []
    for color in colors:
        for band in str(color).split("-"):
            if band and band not in bands:
                bands.append(band)
    return bands


def _build_weights(
    *,
    foreground: pd.DataFrame,
    background: pd.DataFrame,
    random_foreground: pd.DataFrame,
    random_background: pd.DataFrame,
    weighting: Mapping[str, Any] | bool | None,
    colors: tuple[str, ...],
    treecorr_patch_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    foreground_weight = np.ones(len(random_foreground), dtype=float)
    background_weight = np.ones(len(random_background), dtype=float)

    foreground_config = sample_weighting_config(weighting, "foreground")
    background_config = sample_weighting_config(weighting, "background")
    if foreground_config is None and background_config is None:
        return foreground_weight, background_weight

    if foreground_config is not None:
        foreground_weight = _catalog_weights(
            foreground,
            random_foreground,
            foreground_config,
            label="foreground",
            colors=colors,
            treecorr_patch_col=treecorr_patch_col,
        )
    if background_config is not None:
        background_weight = _catalog_weights(
            background,
            random_background,
            background_config,
            label="background",
            colors=colors,
            treecorr_patch_col=treecorr_patch_col,
        )
    print("   TreeCorr random correction using depth-aware random weights")
    return foreground_weight, background_weight


def _random_catalog_like(
    catalog: pd.DataFrame,
    radial_distance: np.ndarray | None,
    pixels_by_patch: dict[int, np.ndarray],
    rng: np.random.Generator,
    *,
    multiplier: float,
    nside: int,
    treecorr_patch_col: str,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    rows = []
    distances = []
    for patch in sorted(catalog[treecorr_patch_col].unique()):
        in_patch = catalog[treecorr_patch_col].to_numpy(int) == patch
        indices = np.where(in_patch)[0]
        n_random = max(1, int(np.ceil(multiplier * len(indices))))
        # Randoms get fresh sky positions but borrow their distances from real
        # objects in the same patch, so they inherit the real redshift
        # distribution and the radial binning stays comparable.
        templates = rng.choice(indices, size=n_random, replace=True)
        ra, dec, pixel = _sample_patch_positions(
            pixels_by_patch[int(patch)],
            n_random,
            rng,
            nside=nside,
        )
        rows.append(
            pd.DataFrame(
                {
                    "ra": ra,
                    "dec": dec,
                    "pixel": pixel,
                    treecorr_patch_col: int(patch),
                }
            )
        )
        if radial_distance is not None:
            distances.append(radial_distance[templates])

    random_catalog = pd.concat(rows, ignore_index=True)
    random_distance = np.concatenate(distances) if distances else None
    return random_catalog, random_distance


def _sample_patch_positions(
    pixels: np.ndarray,
    n_random: int,
    rng: np.random.Generator,
    *,
    nside: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Rejection sampling: draw uniformly inside a bounding box around the
    # allowed pixels, then keep only the draws that land in one of them.
    pixels = np.asarray(pixels, dtype=int)
    pixel_set = set(pixels.tolist())
    lon, lat = hp.pix2ang(nside, pixels, nest=True, lonlat=True)
    # pix2ang returns pixel centres, so pad the box by a pixel to avoid
    # clipping the outer edge of the footprint.
    pixel_radius = np.rad2deg(np.sqrt(hp.nside2pixarea(nside)))
    ra_min = np.min(lon) - pixel_radius
    ra_max = np.max(lon) + pixel_radius
    dec_min = np.max([np.min(lat) - pixel_radius, -90.0])
    dec_max = np.min([np.max(lat) + pixel_radius, 90.0])

    ras: list[np.ndarray] = []
    decs: list[np.ndarray] = []
    sampled_pixels: list[np.ndarray] = []
    n_have = 0
    while n_have < n_random:
        # Oversample, since most of the box is usually outside the footprint.
        batch = max(1024, 4 * (n_random - n_have))
        ra = rng.uniform(ra_min, ra_max, size=batch)
        # Uniform in sin(dec), not in dec: sampling dec directly would pile
        # points up towards the poles instead of spreading them evenly on sky.
        sin_dec = rng.uniform(
            np.sin(np.deg2rad(dec_min)),
            np.sin(np.deg2rad(dec_max)),
            size=batch,
        )
        dec = np.rad2deg(np.arcsin(sin_dec))
        pix = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
        keep = np.fromiter((p in pixel_set for p in pix), dtype=bool, count=batch)
        if np.any(keep):
            ras.append(ra[keep])
            decs.append(dec[keep])
            sampled_pixels.append(pix[keep])
            n_have += int(np.sum(keep))

    return (
        np.concatenate(ras)[:n_random],
        np.concatenate(decs)[:n_random],
        np.concatenate(sampled_pixels)[:n_random],
    )


def _catalog_weights(
    real_catalog: pd.DataFrame,
    random_catalog: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    label: str,
    colors: tuple[str, ...],
    treecorr_patch_col: str,
) -> np.ndarray:
    pixel_col = str(config.get("pixel_col", "pixel"))
    if pixel_col not in real_catalog or pixel_col not in random_catalog:
        raise ValueError(
            f"random_weighting for {label} requires pixel column '{pixel_col}'"
        )

    features = _pixel_weight_features(
        real_catalog,
        config,
        pixel_col=pixel_col,
        label=label,
        colors=colors,
    )
    if features.empty:
        raise ValueError(f"random_weighting for {label} has no usable features")

    real_features = _features_for_pixels(real_catalog[pixel_col], features)
    random_features = _features_for_pixels(random_catalog[pixel_col], features)

    n_bins = int(config.get("n_bins", config.get("bins", 5)))
    if n_bins < 1:
        raise ValueError("random_weighting.n_bins must be positive")
    stratify_by_patch = bool(config.get("stratify_by_patch", True))
    normalize = bool(config.get("normalize", True))
    max_weight = config.get("max_weight")
    max_weight = None if max_weight is None else float(max_weight)
    min_real = int(config.get("min_real_per_stratum", 2))

    weights = np.ones(len(random_catalog), dtype=float)
    if (
        stratify_by_patch
        and treecorr_patch_col in real_catalog
        and treecorr_patch_col in random_catalog
    ):
        # Match within each patch separately, so a patch is never reweighted
        # to look like the depth distribution of a different part of the sky.
        real_patch = real_catalog[treecorr_patch_col].to_numpy(int)
        random_patch = random_catalog[treecorr_patch_col].to_numpy(int)
        patches = np.intersect1d(np.unique(real_patch), np.unique(random_patch))
        for patch in patches:
            real_use = real_patch == patch
            random_use = random_patch == patch
            weights[random_use] = _match_feature_weights(
                real_features[real_use],
                random_features[random_use],
                n_bins=n_bins,
                normalize=normalize,
                max_weight=max_weight,
                min_real=min_real,
            )
        # Randoms in a patch with no real objects have nothing to match
        # against, so drop them rather than let them count unweighted.
        missing_random = ~np.isin(random_patch, patches)
        weights[missing_random] = 0.0
        return weights

    return _match_feature_weights(
        real_features,
        random_features,
        n_bins=n_bins,
        normalize=normalize,
        max_weight=max_weight,
        min_real=min_real,
    )


def _pixel_weight_features(
    catalog: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    pixel_col: str,
    label: str,
    colors: tuple[str, ...],
) -> pd.DataFrame:
    columns = [str(column) for column in config.get("columns", [])]
    feature_data: dict[str, np.ndarray] = {}
    for column in columns:
        if column not in catalog:
            raise ValueError(
                f"random_weighting for {label} requested missing column: {column}"
            )
        feature_data[column] = catalog[column].to_numpy(float)

    depth_config = config.get("depth", config.get("pixel_depth"))
    use_depth = depth_config is not None and depth_config is not False
    if depth_config is True:
        depth_config = {}
    if use_depth:
        if not isinstance(depth_config, Mapping):
            raise ValueError("random_weighting.depth must be a mapping")
        bands = [str(band) for band in depth_config.get("bands", [])]
        if not bands:
            bands = bands_for_colors(colors)
        fluxerr_template = str(depth_config.get("fluxerr_template", "fluxerr_{band}"))
        depth_sigma = float(depth_config.get("depth_sigma", 5.0))
        zero_point = float(depth_config.get("zero_point", 31.4))
        if depth_sigma <= 0:
            raise ValueError("random_weighting.depth.depth_sigma must be positive")
        for band in bands:
            fluxerr_col = fluxerr_template.format(band=band)
            if fluxerr_col not in catalog:
                raise ValueError(
                    f"random_weighting.depth requested missing column: {fluxerr_col}"
                )
            fluxerr = catalog[fluxerr_col].to_numpy(float)
            # Convert the flux error to an n-sigma limiting magnitude, which is
            # the depth an object at that position was actually observed to.
            # Non-positive errors give NaN here and are filtered out below.
            with np.errstate(divide="ignore", invalid="ignore"):
                feature_data[f"depth_{band}"] = zero_point - 2.5 * np.log10(
                    depth_sigma * fluxerr
                )

    if not feature_data:
        for column in _inferred_weight_columns(catalog, colors):
            feature_data[column] = catalog[column].to_numpy(float)

    if not feature_data:
        raise ValueError(
            f"random_weighting for {label} needs columns or depth settings"
        )

    data = pd.DataFrame({"pixel": catalog[pixel_col].to_numpy(int)})
    for name, values in feature_data.items():
        data[name] = np.asarray(values, dtype=float)
    finite_cols = list(feature_data)
    finite = np.ones(len(data), dtype=bool)
    for column in finite_cols:
        finite &= np.isfinite(data[column].to_numpy(float))
    if not np.any(finite):
        return pd.DataFrame(index=pd.Index([], name="pixel"))
    # Collapse to one median value per pixel: the feature describes a location
    # on the sky, so randoms can look it up by pixel even though they have no
    # photometry of their own.
    return data.loc[finite].groupby("pixel")[finite_cols].median()


def _inferred_weight_columns(
    catalog: pd.DataFrame,
    colors: tuple[str, ...],
) -> list[str]:
    bands = bands_for_colors(colors)
    depth_columns = [f"depth5_{band}" for band in bands]
    if bands and all(column in catalog for column in depth_columns):
        return depth_columns

    cmodel_error_columns = [f"cmodel_fluxerr_{band}" for band in bands]
    if bands and all(column in catalog for column in cmodel_error_columns):
        return cmodel_error_columns

    flux_error_columns = [f"fluxerr_{band}" for band in bands]
    if bands and all(column in catalog for column in flux_error_columns):
        return flux_error_columns
    return []


def _features_for_pixels(
    pixels: pd.Series | np.ndarray,
    features: pd.DataFrame,
) -> np.ndarray:
    indexed = features.reindex(np.asarray(pixels, dtype=int))
    return indexed.to_numpy(float)


def _match_feature_weights(
    real_features: np.ndarray,
    random_features: np.ndarray,
    *,
    n_bins: int,
    normalize: bool,
    max_weight: float | None,
    min_real: int,
) -> np.ndarray:
    random_weights = np.ones(len(random_features), dtype=float)
    if len(random_features) == 0:
        return random_weights
    if len(real_features) < min_real:
        return random_weights

    real_ids, random_ids = _feature_bin_ids(
        real_features,
        random_features,
        n_bins=n_bins,
    )
    real_good = real_ids >= 0
    random_good = random_ids >= 0
    if np.sum(real_good) < min_real or not np.any(random_good):
        random_weights[~random_good] = 0.0
        return random_weights

    max_id = int(max(np.max(real_ids[real_good]), np.max(random_ids[random_good])))
    real_counts = np.bincount(real_ids[real_good], minlength=max_id + 1).astype(float)
    random_counts = np.bincount(
        random_ids[random_good],
        minlength=max_id + 1,
    ).astype(float)
    real_fraction = real_counts / np.sum(real_counts)
    random_fraction = random_counts / np.sum(random_counts)

    # Importance weights: a bin holding more of the real sample than of the
    # randoms gets weight above one, and vice versa, so the weighted randoms
    # end up with the same depth distribution as the real objects.
    ratio = np.zeros(max_id + 1, dtype=float)
    valid = random_fraction > 0
    ratio[valid] = real_fraction[valid] / random_fraction[valid]
    random_weights[random_good] = ratio[random_ids[random_good]]
    random_weights[~random_good] = 0.0

    if max_weight is not None:
        random_weights = np.clip(random_weights, 0.0, max_weight)
    if normalize:
        # Rescale to mean one so reweighting changes the distribution but not
        # the effective number of randoms.
        mean_weight = float(np.mean(random_weights))
        if mean_weight > 0 and np.isfinite(mean_weight):
            random_weights = random_weights / mean_weight
    return random_weights


def _feature_bin_ids(
    real_features: np.ndarray,
    random_features: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    real_features = np.asarray(real_features, dtype=float)
    random_features = np.asarray(random_features, dtype=float)
    if real_features.ndim == 1:
        real_features = real_features[:, None]
    if random_features.ndim == 1:
        random_features = random_features[:, None]
    if real_features.shape[1] != random_features.shape[1]:
        raise ValueError("Real and random feature dimensions do not match")

    # Bin each feature separately, then combine the per-feature bin numbers
    # into one integer id per object, mixed-radix style: `base` is the running
    # product of the widths already consumed, so every combination of
    # per-feature bins maps to a distinct id. That turns an N-dimensional
    # histogram into a single bincount below.
    real_ids = np.zeros(len(real_features), dtype=np.int64)
    random_ids = np.zeros(len(random_features), dtype=np.int64)
    real_good = np.ones(len(real_features), dtype=bool)
    random_good = np.ones(len(random_features), dtype=bool)
    base = 1

    for index in range(real_features.shape[1]):
        real_values = real_features[:, index]
        random_values = random_features[:, index]
        finite_real = np.isfinite(real_values)
        finite_random = np.isfinite(random_values)
        real_good &= finite_real
        random_good &= finite_random
        reference = real_values[finite_real]
        if len(reference) == 0:
            real_good[:] = False
            random_good[:] = False
            break
        # Quantile edges come from the real sample only, so both sides are
        # binned on the same scale and the bins are populated by construction.
        # np.unique collapses duplicate edges from heavily tied values.
        edges = np.nanquantile(reference, np.linspace(0.0, 1.0, n_bins + 1))
        edges = np.unique(edges[np.isfinite(edges)])
        if len(edges) <= 1:
            codes_real = np.zeros(len(real_values), dtype=np.int64)
            codes_random = np.zeros(len(random_values), dtype=np.int64)
            width = 1
        else:
            inner_edges = edges[1:-1]
            codes_real = np.searchsorted(inner_edges, real_values, side="right")
            codes_random = np.searchsorted(inner_edges, random_values, side="right")
            width = len(edges) - 1
        real_ids += base * codes_real
        random_ids += base * codes_random
        base *= max(width, 1)

    real_ids[~real_good] = -1
    random_ids[~random_good] = -1
    return real_ids, random_ids
