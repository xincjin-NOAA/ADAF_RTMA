"""Native-grid helpers for reading Ocelot3's URMA2p5 Parquet data directly.

Design decision (per project discussion): ADAF's grid is changed to whatever
native grid the Ocelot3 Parquet files already carry -- no interpolation /
re-gridding of the dense ``ges``/``anal``/satellite fields. Point observations
(``diag_t``/``diag_q``/``diag_uv``) are still rasterized onto that grid
(nearest-cell assignment), since they are scattered station reports, not a
grid, in the source data -- that is unavoidable and is the same kind of step
ADAF's existing WeatherReal-Synoptic preprocessing already performs.

The "no re-grid" contract only holds if the row order of a ``ges_urma``/
``anal_urma`` Parquet partition is truly 1:1, row-major, with
``urma2p5_terrain.npy``'s native (H, W) shape. Ocelot3's own
``DARegionalGraphDataset`` relies on exactly this same alignment (see
``ocelot/training/data/graph_dataset.py``, around the ``self.terrain``
load), and its own GRIB->Parquet converter flags it as an **unverified
assumption**. Treat it the same way here: verify once per new static-data
directory (``verify_grid_alignment`` below) before trusting a training run.

Border padding (``pad_shape``/``pad_to_shape`` below) is a separate concern
from re-gridding: EncDec's Swin-window attention needs the token grid's
resolution divisible by ``window_size`` (a one-time mask precompute at
model-construction time crashes otherwise -- hit for real on the
``urma_ok`` grid, whose 331-pixel dimension is prime, so no
``window_size > 1`` divides it evenly without padding). Padding only
extends the border with zeros (data) / False (validity masks), split
across all 4 sides (``split_pad``) -- it does not interpolate or alter a
single real data value, so it doesn't violate the "no re-grid" decision
above.
"""
import os

import numpy as np


def load_static_grid(static_data_dir):
    """Load the URMA2p5 native grid shape and static per-cell context.

    Returns
    -------
    terrain_hw : np.ndarray, shape (H, W)
        Terrain height, in the file's own native 2D shape (not flattened).
    slmask_hw : np.ndarray, shape (H, W)
        Land/sea mask (1 = land), same shape/order as ``terrain_hw``.
    grid_shape : tuple(int, int)
        ``(H, W)`` -- the source of truth for every other reshape in this
        module, matching what Ocelot3's own DARegionalGraphDataset assumes.
    """
    terrain = np.load(
        os.path.join(static_data_dir, "urma2p5_terrain.npy")
    ).astype(np.float32)
    slmask = np.load(
        os.path.join(static_data_dir, "urma2p5_slmask_nolakes.npz")
    )["slmask"].astype(np.float32)

    if terrain.shape != slmask.shape:
        raise ValueError(
            f"terrain shape {terrain.shape} != slmask shape {slmask.shape}; "
            "these two static files are supposed to describe the same grid."
        )
    if terrain.ndim != 2:
        raise ValueError(
            f"expected a 2D (H, W) terrain array, got shape {terrain.shape}. "
            "If this file is already flattened to 1D, the native (H, W) "
            "shape must be obtained from elsewhere (e.g. the GRIB source) "
            "and passed in explicitly -- do not guess it."
        )

    return terrain, slmask, terrain.shape


def flat_to_grid(values_1d, grid_shape, name=""):
    """Reshape one flat per-grid-cell Parquet column into (H, W).

    No sorting / interpolation is performed -- the row order of
    ``values_1d`` as read from Parquet is trusted to already be row-major
    over ``grid_shape``. This is the crux of the "no re-grid" design; if
    that trust is misplaced, this function will not detect it beyond a
    row-count check.
    """
    values_1d = np.asarray(values_1d)
    h, w = grid_shape
    n_expected = h * w
    if values_1d.shape[0] != n_expected:
        raise ValueError(
            f"{name}: got {values_1d.shape[0]} grid-cell rows, expected "
            f"{n_expected} ({h}x{w}, from urma2p5_terrain.npy). A row-count "
            "mismatch means this Parquet partition's rows do not correspond "
            "1:1 with the static terrain grid -- do not reshape blindly."
        )
    return values_1d.reshape(h, w)


def verify_grid_alignment(lat_deg, lon_deg, grid_shape, atol_deg=1e-3):
    """Best-effort sanity check for the row-order assumption above.

    Takes the same flat ``lat_deg``/``lon_deg`` arrays
    ``ParquetDataManager.get_data_for_bin`` returns per instrument (see
    ``utils/dataloader_ocelot3_parquet.py``'s ``Ocelot3ParquetDataset``,
    which reads these from a ``state/ges`` bin). Reshapes them to (H, W)
    and checks that each row of the reshaped latitude is (roughly) constant,
    and likewise each column of longitude -- the signature of a true
    row-major raster. Raises AssertionError with a description of what
    failed if not. This does NOT prove alignment with the terrain .npy
    specifically (there is no independent coordinate stored there to
    cross-check against), only that the Parquet rows themselves form a
    coherent raster in the assumed order. Run this once against a real
    partition before trusting a training run; see module docstring.
    """
    lat = flat_to_grid(np.asarray(lat_deg), grid_shape, name="lat_deg")
    lon = flat_to_grid(np.asarray(lon_deg), grid_shape, name="lon_deg")

    lat_row_spread = np.nanmax(lat, axis=1) - np.nanmin(lat, axis=1)
    lon_col_spread = np.nanmax(lon, axis=0) - np.nanmin(lon, axis=0)

    if np.nanmax(lat_row_spread) > atol_deg:
        raise AssertionError(
            "latitude is not constant along rows after reshape to "
            f"{grid_shape} -- row-major flatten assumption looks wrong "
            f"(max within-row latitude spread = {np.nanmax(lat_row_spread):.4f} deg)."
        )
    if np.nanmax(lon_col_spread) > atol_deg:
        raise AssertionError(
            "longitude is not constant along columns after reshape to "
            f"{grid_shape} -- row-major flatten assumption looks wrong "
            f"(max within-column longitude spread = {np.nanmax(lon_col_spread):.4f} deg)."
        )
    return lat, lon


def build_grid_tree(grid_lat, grid_lon):
    """KDTree over the grid's (lat, lon) cell centers, for reuse across many
    ``rasterize_points_to_grid`` calls on the same grid."""
    import scipy.spatial

    grid_xy = np.stack([np.asarray(grid_lat).ravel(), np.asarray(grid_lon).ravel()], axis=1)
    return scipy.spatial.KDTree(grid_xy)


def rasterize_points_to_grid(obs_lat, obs_lon, obs_value, grid_lat, grid_lon,
                              grid_shape, max_dist_deg=None, tree=None):
    """Nearest-cell rasterization of scattered point obs onto the native grid.

    Unlike ``flat_to_grid``, this *is* a real point -> grid assignment
    (unavoidable: station reports are not natively on grid rows). Mirrors
    the nearest-neighbor approach Ocelot3 itself uses for obs->mesh edges
    (``scipy.spatial.KDTree`` in ``graph_dataset.py``), just producing a
    dense raster instead of graph edges. Multiple obs snapping to the same
    cell are averaged; cells with no obs are filled with 0, matching ADAF's
    existing ``sta_*`` convention (0 = no station). Pass ``tree`` (from
    ``build_grid_tree``) to skip rebuilding the KDTree on every call.
    """
    h, w = grid_shape
    out = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.int32)

    valid = ~(np.isnan(obs_lat) | np.isnan(obs_lon) | np.isnan(obs_value))
    if not np.any(valid):
        return out

    if tree is None:
        tree = build_grid_tree(grid_lat, grid_lon)

    obs_xy = np.stack([obs_lat[valid], obs_lon[valid]], axis=1)
    dist, idx = tree.query(obs_xy, k=1)

    if max_dist_deg is not None:
        keep = dist <= max_dist_deg
        idx = idx[keep]
        vals = obs_value[valid][keep]
    else:
        vals = obs_value[valid]

    row = idx // w
    col = idx % w
    np.add.at(out, (row, col), vals)
    np.add.at(counts, (row, col), 1)

    nonzero = counts > 0
    out[nonzero] = out[nonzero] / counts[nonzero]
    return out


def pad_shape(h, w, multiple):
    """Round (h, w) up so each dimension gains at least one full
    ``multiple`` (e.g. ``patch_size * window_size``) block of padding.

    Rounding up to the *nearest* multiple (plain ceil division) can add as
    little as 1 pixel of padding, or even 0 if ``h``/``w`` already divides
    evenly -- both still satisfy EncDec's Swin-window mask-precompute
    divisibility requirement (see module docstring), but leave too thin a
    synthetic border to buffer real data from the window/conv contamination
    that decays over several cells from any masked-out edge (see
    adaf repo's ``docs/BOUNDARY_ERROR_DIAGNOSTIC.md``).
    Ceiling to the nearest multiple always leaves a remainder strictly less
    than ``multiple``, so unconditionally adding one more ``multiple``
    guarantees at least a full block of padding while keeping the result
    divisible by ``multiple``. ``multiple <= 1`` is a no-op, returning
    (h, w) unchanged.
    """
    if multiple <= 1:
        return h, w
    pad_h = (-(-h // multiple) + 1) * multiple  # ceil to multiple, plus one more
    pad_w = (-(-w // multiple) + 1) * multiple
    return pad_h, pad_w


def split_pad(total):
    """Split a padding amount between the two sides of a dimension.

    Floor half goes before, the remainder (ceil half) goes after -- so an
    odd amount puts the extra pixel on the bottom/right, matching
    ``pad_to_shape``'s and ``EncDec.check_image_size``'s shared bottom/right
    bias for whichever side can't be split evenly.
    """
    before = total // 2
    return before, total - before


def pad_to_shape(arr, target_shape, mode="constant", constant_values=0):
    """Pad an array's last two (spatial) dims up to ``target_shape``.

    Splits the padding across all 4 sides (top/bottom/left/right) via
    ``split_pad``, rather than dumping it all on the bottom/right -- this
    keeps the real (unpadded) grid data away from every edge of the padded
    array, not just two of them, so window/conv layers see synthetic border
    cells on all sides equally instead of only two. Downstream code does not
    assume a top-left origin: ``field_mask``/``obs_tar_mask`` are padded
    with this same split and used purely elementwise (e.g. train.py's
    ``masked_fill(~field_mask, 0)``), so wherever the valid region ends up
    within the padded array, it stays correctly marked.
    For boolean arrays (validity masks), ``constant_values=0`` pads with
    False, so the padded border reads as "not valid" wherever that mask is
    later used to exclude cells (e.g. ADAF's field_mask-driven loss
    masking) -- the same convention already used for HRRR's own
    out-of-domain cells elsewhere in this codebase.
    """
    *lead, h, w = arr.shape
    th, tw = target_shape
    pad_h, pad_w = th - h, tw - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(
            f"target_shape {target_shape} is smaller than array's spatial "
            f"shape {(h, w)} -- pad_to_shape only grows arrays."
        )
    top, bottom = split_pad(pad_h)
    left, right = split_pad(pad_w)
    pad_width = [(0, 0)] * len(lead) + [(top, bottom), (left, right)]
    if mode == "constant":
        return np.pad(arr, pad_width, mode=mode, constant_values=constant_values)
    return np.pad(arr, pad_width, mode=mode)
