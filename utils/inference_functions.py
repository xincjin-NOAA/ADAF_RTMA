import os
import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
import argparse
import datetime as dt
import hdf5plugin

from utils.misc_functions import *
from utils.YParams import *

##################


# ============================================================================
# HELPER FUNCTIONS & CHECKPOINT LOADING
# ============================================================================

def load_checkpoint_weights(model, checkpoint_file, map_device):
    """Loads model weights, iteratively stripping 'module.' and '_orig_mod.' prefixes,
    and filtering out resolution-dependent 'attn_mask' buffers.
    """
    ck = torch.load(checkpoint_file, map_location=map_device)
    state = ck.get("model_state", ck.get("state_dict"))
    if state is None:
        raise KeyError("Checkpoint does not contain 'model_state' or 'state_dict'.")

    def _strip(k):
        changed = True
        while changed:
            changed = False
            for pre in ("_orig_mod.", "module."):
                if k.startswith(pre):
                    k = k[len(pre):]
                    changed = True
        return k

    #Filter out 'attn_mask' entries alongside key stripping
    clean_state = {
        _strip(key): val 
        for key, val in state.items() 
        if "attn_mask" not in key
    }

    # Set strict=False to allow skipping filtered attention masks
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    return ck, missing, unexpected


def load_stats(stats_path, var_names):
    """Extracts min and max values for specified variables from stats.csv."""
    stats_df = pd.read_csv(stats_path).set_index("variable")
    vmin = np.array([stats_df.loc[v, "min"] for v in var_names], dtype=np.float32)
    vmax = np.array([stats_df.loc[v, "max"] for v in var_names], dtype=np.float32)
    return vmin, vmax


def reverse_norm(arr, vmin, vmax, channel_axis=0):
    """Reverses min-max normalization from [-1, 1] back to physical units."""
    arr = np.asarray(arr, dtype=np.float32)
    vmin = np.asarray(vmin, dtype=np.float32).reshape(-1)
    vmax = np.asarray(vmax, dtype=np.float32).reshape(-1)

    if channel_axis < 0:
        channel_axis = arr.ndim + channel_axis

    if arr.shape[channel_axis] != vmin.size:
        raise ValueError(
            f"Channel mismatch: axis {channel_axis} has {arr.shape[channel_axis]} channels "
            f"but stats vector has {vmin.size}."
        )

    reshape = [1] * arr.ndim
    reshape[channel_axis] = vmin.size
    vmin_b = vmin.reshape(reshape)
    vmax_b = vmax.reshape(reshape)
    return (arr + 1.0) * (vmax_b - vmin_b) / 2.0 + vmin_b


# ============================================================================
# DATA PREPARATION & MODEL INFERENCE
# ============================================================================

def build_model_input_from_netcdf(nc_file, p, include_metar=True):
    """Loads NetCDF inputs matching train/serve pipeline transformations."""
    ds = xr.open_dataset(nc_file, engine="netcdf4")

    try:
        H = p.img_size_y
        W = p.img_size_x

        lon = np.array(ds.coords["lon"].values)
        lat = np.array(ds.coords["lat"].values)

        # Topo
        topo = ds[["z"]].to_array().to_numpy()

        # Background prediction input (formerly HRRR)
        inp_pred = ds[p.inp_pred_vars].to_array().to_numpy()
        inp_pred = np.squeeze(inp_pred)

        # Observations
        obs = ds[p.inp_obs_vars].to_array().to_numpy()[:, -p.obs_time_window:, :]

        # Satellite Data (4 bands x 3 time steps -> 12 channels)
        sat_window = getattr(p, "obs_time_window", 3)
        sat = ds[p.inp_sat_vars].to_array().to_numpy()[:, -sat_window:, :]
        inp_sat = sat.reshape((-1, H, W))

        # Optional METAR filtering from CLI/flag
        if (not include_metar) and ("obs_source" in ds):
            obs_source = ds["obs_source"].to_numpy()
            obs[:, :, obs_source == 2] = 0

        # Optional train_obs_source filtering (e.g. metar-only runs)
        _tsrc = getattr(p, "train_obs_source", "all")
        if _tsrc in ("metar", "mesonet") and ("obs_source" in ds):
            _code = 2 if _tsrc == "metar" else 1
            obs = obs * (ds["obs_source"].to_numpy() == _code)

        obs_tar = obs[:, -1]
        obs_tar_mask = (obs_tar != 0).astype(np.float32)

        # Hold-out mask generation matching training dataloader
        if p.hold_out_obs:
            obs_idx = np.flatnonzero((obs[:, -1] != 0).any(axis=0).ravel())
            hold_out_num = int(len(obs_idx) * p.hold_out_obs_ratio)

            if p.obs_mask_seed is None or p.obs_mask_seed < 0:
                rng = np.random.default_rng()
            else:
                digits = "".join(c for c in os.path.basename(nc_file) if c.isdigit())
                rng = np.random.default_rng([int(p.obs_mask_seed), int(digits or 0)])

            hold_out_idx = rng.choice(obs_idx, size=hold_out_num, replace=False)

            obs_mask = np.zeros(H * W, dtype=np.float32)
            obs_mask[hold_out_idx] = 1.0
            obs_mask = obs_mask.reshape(H, W)

            inp_obs = obs * (1.0 - obs_mask)
            inp_obs = inp_obs.reshape((-1, H, W))
        else:
            inp_obs = obs.reshape((-1, H, W))
            obs_mask = np.zeros((H, W), dtype=np.float32)

        # Targets (Normalized space)
        field_tar = ds[p.field_tar_vars].to_array().to_numpy()[:, :H, :W]

        field_obs_tar = field_tar.copy()
        field_obs_tar[obs_tar_mask == 1] = 0
        field_obs_tar += obs_tar

        # Target residual in normalized space if model learns residual
        if p.learn_residual:
            field_tar_res = field_tar - inp_pred
            obs_tar_res = obs_tar - inp_pred
            field_obs_tar_res = field_obs_tar - inp_pred
        else:
            field_tar_res = field_tar
            obs_tar_res = obs_tar
            field_obs_tar_res = field_obs_tar

        # Concatenate in training channel order: [inp_pred, inp_obs, inp_sat, topo]
        inp = np.concatenate((inp_pred, inp_obs, inp_sat, topo), axis=0).astype(np.float32)

        aux = {
            "lat": lat,
            "lon": lon,
            "inp_pred": inp_pred.astype(np.float32),
            "inp_obs": inp_obs.astype(np.float32),
            "inp_sat": inp_sat.astype(np.float32),
            "topo": topo.astype(np.float32),
            "target_field_norm": field_tar.astype(np.float32),
            "target_field_res_norm": field_tar_res.astype(np.float32),
            "target_obs_res_norm": obs_tar_res.astype(np.float32),
            "target_field_obs_res_norm": field_obs_tar_res.astype(np.float32),
            "obs_tar_mask": obs_tar_mask.astype(np.float32),
            "obs_mask": obs_mask.astype(np.float32),
        }
        return inp, aux
    finally:
        ds.close()


def build_model_input_from_ocelot3(dataset, idx, p):
    """Ocelot3 counterpart of build_model_input_from_netcdf: assembles sample idx of an
    Ocelot3ParquetDataset (utils/dataloader_ocelot3_parquet.py) into the same (inp, aux) pair.

    Everything is in Ocelot3's z-scored space, padded to the model's img_size. Hold-out follows
    p.hold_out_obs / p.hold_out_obs_ratio / p.obs_mask_seed and picks stations the same way as
    the NetCDF builder. There is no satellite input and no METAR/mesonet split on this path.
    """
    from utils.ocelot3_grid_source import pad_to_shape

    bin_name = dataset.binned_samples[idx]
    H, W = dataset.grid_shape

    inp_pred, field_mask = dataset._state_grid_and_mask(dataset.loader, "ges", bin_name)
    field_tar, _anal_mask = dataset._state_grid_and_mask(dataset.anal_loader, "anal", bin_name)

    # Observations (n_vars, T, H, W), 0 = no station
    obs = dataset._read_obs_window(bin_name)
    obs_tar = obs[:, -1]
    obs_tar_mask = (obs_tar != 0).astype(np.float32)

    # Hold-out mask generation matching build_model_input_from_netcdf
    if p.hold_out_obs:
        obs_idx = np.flatnonzero((obs_tar != 0).any(axis=0).ravel())
        hold_out_num = int(len(obs_idx) * p.hold_out_obs_ratio)

        if p.obs_mask_seed is None or p.obs_mask_seed < 0:
            rng = np.random.default_rng()
        else:
            digits = "".join(c for c in bin_name if c.isdigit())
            rng = np.random.default_rng([int(p.obs_mask_seed), int(digits or 0)])

        hold_out_idx = rng.choice(obs_idx, size=hold_out_num, replace=False)

        obs_mask = np.zeros(H * W, dtype=np.float32)
        obs_mask[hold_out_idx] = 1.0
        obs_mask = obs_mask.reshape(H, W)

        inp_obs = obs * (1.0 - obs_mask)
        inp_obs = inp_obs.reshape((-1, H, W))
    else:
        inp_obs = obs.reshape((-1, H, W))
        obs_mask = np.zeros((H, W), dtype=np.float32)

    field_obs_tar = field_tar.copy()
    field_obs_tar[obs_tar_mask == 1] = 0
    field_obs_tar += obs_tar

    if p.learn_residual:
        field_tar_res = field_tar - inp_pred
        obs_tar_res = obs_tar - inp_pred
        field_obs_tar_res = field_obs_tar - inp_pred
    else:
        field_tar_res = field_tar
        obs_tar_res = obs_tar
        field_obs_tar_res = field_obs_tar

    topo = dataset.terrain[np.newaxis, :, :]
    inp_sat = np.zeros((0, H, W), dtype=np.float32)

    # Same channel order as Ocelot3ParquetDataset / Trainer.prepare_batch: [inp_pred, inp_obs, topo]
    inp = np.concatenate((inp_pred, inp_obs, topo), axis=0).astype(np.float32)

    # Pad from the native grid up to params.img_size_y/x (no-op when equal)
    pad = lambda a: pad_to_shape(a, dataset.padded_shape)
    lat, lon = dataset._padded_lat_lon()

    aux = {
        "lat": lat,
        "lon": lon,
        "inp_pred": pad(inp_pred).astype(np.float32),
        "inp_obs": pad(inp_obs).astype(np.float32),
        "inp_sat": pad(inp_sat).astype(np.float32),
        "topo": pad(topo).astype(np.float32),
        "target_field_norm": pad(field_tar).astype(np.float32),
        "target_field_res_norm": pad(field_tar_res).astype(np.float32),
        "target_obs_res_norm": pad(obs_tar_res).astype(np.float32),
        "target_field_obs_res_norm": pad(field_obs_tar_res).astype(np.float32),
        "obs_tar_mask": pad(obs_tar_mask).astype(np.float32),
        "obs_mask": pad(obs_mask).astype(np.float32),
        "field_mask": pad(field_mask),
    }
    return pad(inp), aux


def ocelot3_state_stats(dataset):
    """Per-channel (mean, std) that undo Ocelot3's z-score for the ges/anal fields, in
    dataset.state_vars order. anal is normalized with ges's stats, so one set covers both.

    Temperature comes back in C (Ocelot3 stores K) so units match the NetCDF path.
    """
    from utils.dataloader_ocelot3_parquet import STATE_FEATURE_NAMES

    ges_stats = dataset.feature_stats["ges"]
    mean, std = [], []
    for v in dataset.state_vars:
        m, s = ges_stats.get(STATE_FEATURE_NAMES[v], [0.0, 1.0])
        mean.append(m - 273.15 if v == "t" else m)
        std.append(s if s > 0 else 1.0)  # same zero-std guard as orca_common's normalization
    return np.array(mean, dtype=np.float32), np.array(std, dtype=np.float32)


def ocelot3_obs_stats(dataset):
    """Per-channel (mean, std) that undo Ocelot3's z-score for the station obs, in
    dataset.state_vars order. Obs use their conventional instrument's own stats (diag_t, ...),
    which differ from ges's; falls back to ges's stats if an instrument has none.

    Temperature comes back in C, as in ocelot3_state_stats.
    """
    from utils.dataloader_ocelot3_parquet import CONVENTIONAL_INSTRUMENTS

    mean, std = ocelot3_state_stats(dataset)
    for i, v in enumerate(dataset.state_vars):
        inst_stats = dataset.feature_stats.get(CONVENTIONAL_INSTRUMENTS[v]) or {}
        if len(inst_stats) == 1:  # single-feature instrument
            m, s = next(iter(inst_stats.values()))
            mean[i] = m - 273.15 if v == "t" else m
            std[i] = s if s > 0 else 1.0
    return mean, std


def reverse_zscore(arr, mean, std, channel_axis=0):
    """Reverses z-score normalization (x - mean) / std back to physical units."""
    arr = np.asarray(arr, dtype=np.float32)
    reshape = [1] * arr.ndim
    reshape[channel_axis] = -1
    return arr * np.asarray(std).reshape(reshape) + np.asarray(mean).reshape(reshape)


def _run_and_package(model, inp_np, aux, params, device, unnorm_pred, unnorm_anl,
                     inp_pred_vars, inp_obs_vars, inp_sat_vars, field_tar_vars, unnorm_obs=None):
    """Shared by run_model_inference and run_model_inference_ocelot3: runs the model on one
    assembled input, reconstructs the analysis, unnormalizes it with unnorm_pred/unnorm_anl
    (and the station obs with unnorm_obs, default unnorm_anl) and packages the results
    dictionary used by the plotting functions below.
    """
    # 1. Inference
    inp_tensor = torch.from_numpy(inp_np).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred_tensor = model(inp_tensor)

    pred_norm = pred_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)

    inp_pred_norm = aux["inp_pred"].copy()

    # 2. Analysis Reconstruction (in Normalized Space)
    if params.learn_residual:
        pred_analysis_norm = pred_norm + inp_pred_norm
        target_analysis_norm = aux["target_field_res_norm"] + inp_pred_norm
    else:
        pred_analysis_norm = pred_norm
        target_analysis_norm = aux["target_field_res_norm"]

    # 3. Reverse Normalization to Physical Units
    inp_pred_unnorm = unnorm_pred(inp_pred_norm)
    pred_analysis_unnorm = unnorm_anl(pred_analysis_norm)
    target_analysis_unnorm = unnorm_anl(target_analysis_norm)

    # Cells the loader marks invalid (Ocelot3 padding / missing ges) -> NaN
    if "field_mask" in aux:
        invalid = ~np.asarray(aux["field_mask"], dtype=bool)
        for arr in (inp_pred_unnorm, pred_analysis_unnorm, target_analysis_unnorm):
            arr[invalid] = np.nan

    # Physical residual (innovation) = Analysis - Background Prediction
    pred_residual_unnorm = pred_analysis_unnorm - inp_pred_unnorm
    target_residual_unnorm = target_analysis_unnorm - inp_pred_unnorm

    # Station obs at analysis time (all stations, including held-out ones), NaN elsewhere
    obs_tar_norm = aux["target_obs_res_norm"] + inp_pred_norm if params.learn_residual else aux["target_obs_res_norm"]
    obs_unnorm = (unnorm_obs or unnorm_anl)(obs_tar_norm)
    obs_unnorm[aux["obs_tar_mask"] == 0] = np.nan

    # 4. Channel maps and result packaging
    output_channel_names = [
        f"output_{v.split('rtma_anl_', 1)[1]}" if v.startswith("rtma_anl_") else f"output_{v}"
        for v in field_tar_vars
    ]

    channel_maps = {
        "input_pred": {i: v for i, v in enumerate(inp_pred_vars)},
        "input_obs": {i: v for i, v in enumerate(inp_obs_vars)},
        "input_sat": {i: v for i, v in enumerate(inp_sat_vars)},
        "output": {i: v for i, v in enumerate(output_channel_names)},
        "target_field": {i: v for i, v in enumerate(field_tar_vars)},
    }

    results = {
        # Tensors & Normalized Arrays
        "prediction_tensor": pred_tensor,
        "prediction_array_norm": pred_norm,
        "input_tensor": inp_tensor,
        "input_array_norm": inp_np,
        "target_field_array_norm": aux["target_field_norm"],
        "target_obs_array_norm": aux["target_obs_res_norm"],
        "target_field_obs_array_norm": aux["target_field_obs_res_norm"],

        # Unnormalized Arrays (Physical Units)
        "prediction_residual_unnorm": pred_residual_unnorm,
        "target_residual_unnorm": target_residual_unnorm,
        "inp_pred_unnorm": inp_pred_unnorm,
        "prediction_analysis_unnorm": pred_analysis_unnorm,
        "target_analysis_unnorm": target_analysis_unnorm,
        "obs_unnorm": obs_unnorm,

        # Metadata & Coordinates
        "channel_maps": channel_maps,
        "output_channel_names": output_channel_names,
        "obs_tar_mask_array": aux["obs_tar_mask"],
        "heldout_mask": aux["obs_mask"],
        "lat": aux["lat"],
        "lon": aux["lon"],
    }

    return results


def run_model_inference(model, nc_path, params, stats_path, device, include_metar=True):
    """Runs model inference on a NetCDF file, unnormalizes outputs with the min-max stats in
    stats_path, and returns a packaged dictionary of results.
    """
    inp_np, aux = build_model_input_from_netcdf(nc_path, params, include_metar=include_metar)

    rtma_anl_vmin, rtma_anl_vmax = load_stats(stats_path, params.field_tar_vars)
    pred_input_vmin, pred_input_vmax = load_stats(stats_path, params.inp_pred_vars)
    obs_vmin, obs_vmax = load_stats(stats_path, params.inp_obs_vars)

    return _run_and_package(
        model, inp_np, aux, params, device,
        unnorm_pred=lambda a: reverse_norm(a, pred_input_vmin, pred_input_vmax, channel_axis=0),
        unnorm_anl=lambda a: reverse_norm(a, rtma_anl_vmin, rtma_anl_vmax, channel_axis=0),
        unnorm_obs=lambda a: reverse_norm(a, obs_vmin, obs_vmax, channel_axis=0),
        inp_pred_vars=params.inp_pred_vars,
        inp_obs_vars=params.inp_obs_vars,
        inp_sat_vars=getattr(params, "inp_sat_vars", []),
        field_tar_vars=params.field_tar_vars,
    )


def run_model_inference_ocelot3(model, dataset, idx, params, device):
    """Runs model inference on sample idx of an Ocelot3ParquetDataset, unnormalizes outputs with
    Ocelot3's z-score stats, and returns the same results dictionary as run_model_inference.
    """
    inp_np, aux = build_model_input_from_ocelot3(dataset, idx, params)

    mean, std = ocelot3_state_stats(dataset)
    unnorm = lambda a: reverse_zscore(a, mean, std, channel_axis=0)
    obs_mean, obs_std = ocelot3_obs_stats(dataset)

    return _run_and_package(
        model, inp_np, aux, params, device,
        unnorm_pred=unnorm,
        unnorm_anl=unnorm,  # anal is normalized with ges's stats
        unnorm_obs=lambda a: reverse_zscore(a, obs_mean, obs_std, channel_axis=0),
        inp_pred_vars=[f"ges_{v}" for v in dataset.state_vars],
        inp_obs_vars=[f"obs_{v}" for v in dataset.state_vars],
        inp_sat_vars=[],
        field_tar_vars=list(dataset.state_vars),
    )


# ============================================================================
# PLOTTING
# ============================================================================

def plot_output_channel(results_dict, channel_name, 
                        channel_to_select="prediction_analysis_unnorm", 
                        colorbar_scale_style="normal",
                        abs_min=None, abs_max=None,
                        title_str=None, 
                        colorbar_label=None, 
                        plot_savepath=None, 
                        cmap='bwr'):
    """Plots one unnormalized output channel by variable name (e.g., 'output_t')."""
    output_names = results_dict['output_channel_names']
    if channel_name not in output_names:
        raise KeyError(
            f"Unknown channel '{channel_name}'. Available: {output_names}"
        )

    idx = output_names.index(channel_name)
    arr = results_dict[channel_to_select][idx]

    lat = np.asarray(results_dict['lat'])
    lon = np.asarray(results_dict['lon'])
    extent = [float(np.min(lon)), float(np.max(lon)), float(np.min(lat)), float(np.max(lat))]

    if colorbar_scale_style == "centered":
        abs_val = np.nanmax(np.abs(arr))
        vmin, vmax = -abs_val, abs_val
    elif colorbar_scale_style == 'extreme':
        vmin, vmax = abs_min, abs_max
    elif colorbar_scale_style == "normal":
        vmin, vmax = None, None
    else:
        raise ValueError("colorbar_scale_style must be 'normal', 'extreme', or 'centered'")

    fig, ax = plt.subplots(figsize=(12, 6)) 
    im = ax.imshow(arr, origin='lower', cmap=cmap, extent=extent, aspect='auto', vmin=vmin, vmax=vmax)

    if title_str is None:
        ax.set_title(f"{channel_name} ({channel_to_select}) | min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f}")
    else:
        ax.set_title(title_str)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    cbar = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.016)
    if colorbar_label is None:
        cbar.set_label(f"{channel_name} value")
    else:
        cbar.set_label(colorbar_label)

    if plot_savepath is not None:
        plt.savefig(plot_savepath, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def plot_scatter_pred_vs_target(results_dict, channel_name,
                                pred_key="prediction_analysis_unnorm",
                                target_key="target_analysis_unnorm",
                                points="all",
                                max_points=200000,
                                seed=0,
                                axis_min=None, axis_max=None,
                                title_str=None,
                                units=None,
                                plot_savepath=None,
                                ax=None):
    """Scatter of predicted vs. target values for one output channel (e.g., 'output_t').

    points selects which grid cells are compared:
        "all"     -- every valid (non-NaN) grid cell
        "obs"     -- cells with a station ob at analysis time (obs_tar_mask)
        "heldout" -- station cells withheld from the model input (heldout_mask & obs_tar_mask)
    The target is always results_dict[target_key] (RTMA analysis by default), including at
    station cells -- results_dict holds the obs themselves only in normalized residual form.

    Plots at most max_points randomly sampled points (the full grid is ~3.7M cells), but the
    statistics in the legend (N, bias, RMSE, correlation) use every selected point.
    Pass ax to draw into an existing subplot; otherwise a new figure is created and shown.
    Returns the stats dict.
    """
    output_names = results_dict['output_channel_names']
    if channel_name not in output_names:
        raise KeyError(
            f"Unknown channel '{channel_name}'. Available: {output_names}"
        )
    idx = output_names.index(channel_name)

    pred = np.asarray(results_dict[pred_key][idx], dtype=np.float64)
    target = np.asarray(results_dict[target_key][idx], dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred_key} {pred.shape} vs {target_key} {target.shape}")

    valid = np.isfinite(pred) & np.isfinite(target)
    if points == "obs":
        valid &= np.asarray(results_dict['obs_tar_mask_array'][idx]) > 0
    elif points == "heldout":
        valid &= (np.asarray(results_dict['obs_tar_mask_array'][idx]) > 0) & (np.asarray(results_dict['heldout_mask']) > 0)
    elif points != "all":
        raise ValueError("points must be 'all', 'obs', or 'heldout'")

    x = target[valid]
    y = pred[valid]
    n = x.size
    if n == 0:
        raise ValueError(f"No valid points for {channel_name} with points='{points}'")

    diff = y - x
    stats = {
        "n": int(n),
        "bias": float(np.mean(diff)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "corr": float(np.corrcoef(x, y)[0, 1]) if n > 1 else np.nan,
    }

    if n > max_points:
        sample = np.random.default_rng(seed).choice(n, size=max_points, replace=False)
        x_plot, y_plot = x[sample], y[sample]
    else:
        x_plot, y_plot = x, y

    if axis_min is None or axis_max is None:
        lo = min(np.min(x_plot), np.min(y_plot))
        hi = max(np.max(x_plot), np.max(y_plot))
        pad = 0.02 * (hi - lo if hi > lo else 1.0)
        axis_min = lo - pad if axis_min is None else axis_min
        axis_max = hi + pad if axis_max is None else axis_max

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(7, 7))

    marker_size = 1 if points == "all" else 6
    ax.scatter(x_plot, y_plot, s=marker_size, alpha=0.3, edgecolors='none', rasterized=True)
    ax.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', linewidth=1, label='1:1')
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.set_aspect('equal')

    unit_str = f" ({units})" if units else ""
    ax.set_xlabel(f"Target: {target_key}{unit_str}")
    ax.set_ylabel(f"Predicted: {pred_key}{unit_str}")
    if title_str is None:
        ax.set_title(f"{channel_name}, {points} points")
    else:
        ax.set_title(title_str)

    stats_text = (f"N = {stats['n']:,}" + (f" (plotted {x_plot.size:,})" if x_plot.size < n else "") + "\n"
                  f"bias = {stats['bias']:.3f}\n"
                  f"RMSE = {stats['rmse']:.3f}\n"
                  f"r = {stats['corr']:.4f}")
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    if own_figure:
        if plot_savepath is not None:
            plt.savefig(plot_savepath, dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

    return stats
