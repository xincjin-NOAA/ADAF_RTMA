"""Batch inference for ADAF-RTMA: run a trained checkpoint over a range of analysis times,
save the analyses and compute verification metrics.

The command-line counterpart of evaluation.py, which handles one hour with hard-coded paths.
Settings come from a predict YAML (configs/predict_example.yaml,
configs/predict_ocelot3_example.yaml); see docs/PREDICTING.md for every option.

For each analysis time and each obs mode it runs the model and writes
  <output_dir>/metrics_per_time.csv   one row per time / mode / source / verification / variable
  <output_dir>/metrics_summary.csv    the same, pooled over all times
  <output_dir>/fields/<YYYY-MM-DD_HH>.nc|.pt   analyses in physical units (save_fields, field_formats)
  <output_dir>/plots/                 error maps and scatter plots (plot_channels)

Obs modes:
  all_obs  -- the model sees every station
  heldout  -- a seeded fraction of stations is withheld (heldout_ratio, default the config's
              hold_out_obs_ratio) and verified separately ("heldout_obs")
  no_obs   -- every station is withheld, so the "obs" verification is fully independent

Usage (from the repo root, on a GPU node -- or submit with ./submit_predict.sh):
  python predict.py configs/predict_example.yaml [key=value ...]

key=value overrides are parsed as YAML; use dotted keys for nested ones, e.g.
  python predict.py configs/predict_example.yaml start_time=2023-06-13T00:00 params.hold_out_obs_ratio=0.2
"""
import argparse
import copy
import datetime as dt
import difflib
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from ruamel.yaml import YAML

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models.encdec_lowres import LowResEncDec
from utils.YParams import YParams
from utils.misc_functions import to_builtin
from utils.inference_functions import (
    load_checkpoint_weights,
    plot_output_channel,
    plot_scatter_pred_vs_target,
    run_model_inference,
    run_model_inference_ocelot3,
)

PREDICT_DEFAULTS = {
    "name": None,                  # defaults to the config file's stem
    "description": "",
    # --- model ---
    "config_filepath": "./config/params_lowres_ges_goes.yaml",
    "checkpoint": None,            # required, e.g. training_runs/<name>/best_ckpt.tar
    "params": {},                  # overrides of config_filepath keys (must match training!)
    # --- times ---
    "start_time": None,            # first analysis time, e.g. 2023-06-13T00:00
    "end_time": None,              # last analysis time (inclusive); default start_time
    "hour_step": 1,
    "times": None,                 # explicit list of times instead of start/end/step
    # --- data (NetCDF path only; Ocelot3 reads params.ocelot3_*) ---
    "data_dir": None,              # dir of YYYY-MM-DD_HH.nc files; default params.test_data_path
    "stats_path": "./data_preparation_ges/stats_ges.csv",
    # --- what to run ---
    "modes": ["all_obs", "heldout", "no_obs"],
    "heldout_ratio": None,         # heldout mode; default params.hold_out_obs_ratio
    "device": "auto",              # auto | cuda | cpu
    # --- outputs ---
    "output_dir": "predictions/{name}",
    "save_fields": True,
    "field_formats": ["nc"],       # any of nc (NetCDF), pt (torch.save dict of tensors)
    "plot_channels": [],           # e.g. [t, u10]: error map + scatter per time (all_obs, else first mode)
    "skip_existing": False,        # skip times already in metrics_per_time.csv (resume after a timeout)
    "continue_on_error": True,     # log and skip a time whose data can't be read
}
IGNORED_KEYS = {"slurm"}           # read by tools/submit_predict.py only
MODES = ("all_obs", "heldout", "no_obs")
FIELD_FORMATS = ("nc", "pt")
UNITS = {"t": "C", "q": "kg/kg", "u10": "m/s", "v10": "m/s"}

####################

def load_yaml(path):
    with open(path) as f:
        return to_builtin(YAML(typ="safe", pure=True).load(f) or {})


def set_dotted(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_predict_config(path, overrides):
    cfg = load_yaml(path)
    for kv in overrides:
        if "=" not in kv:
            sys.exit(f"Expected KEY=VALUE, got {kv!r}")
        key, value = kv.split("=", 1)
        set_dotted(cfg, key.strip(), YAML(typ="safe", pure=True).load(value))

    unknown = [k for k in cfg if k not in PREDICT_DEFAULTS and k not in IGNORED_KEYS]
    for k in unknown:
        hint = difflib.get_close_matches(k, list(PREDICT_DEFAULTS), n=1)
        print(f"ERROR: unknown option '{k}'" + (f" (did you mean '{hint[0]}'?)" if hint else ""), file=sys.stderr)
    if unknown:
        sys.exit(1)

    cfg = {**copy.deepcopy(PREDICT_DEFAULTS), **{k: v for k, v in cfg.items() if k not in IGNORED_KEYS}}
    for key in ("params", "plot_channels", "field_formats"):  # an empty YAML block loads as None
        cfg[key] = cfg[key] or PREDICT_DEFAULTS[key]
    cfg["name"] = cfg["name"] or os.path.splitext(os.path.basename(path))[0]
    cfg["output_dir"] = str(cfg["output_dir"]).format(name=cfg["name"])

    if not cfg["checkpoint"]:
        sys.exit("checkpoint is required")
    if isinstance(cfg["field_formats"], str):
        cfg["field_formats"] = [cfg["field_formats"]]
    bad_formats = [f for f in cfg["field_formats"] if f not in FIELD_FORMATS]
    if bad_formats or (cfg["save_fields"] and not cfg["field_formats"]):
        sys.exit(f"field_formats must be a non-empty subset of {list(FIELD_FORMATS)}; got {cfg['field_formats']}")
    bad_modes = [m for m in cfg["modes"] if m not in MODES]
    if bad_modes or not cfg["modes"]:
        sys.exit(f"modes must be a non-empty subset of {list(MODES)}; got {cfg['modes']}")
    return cfg


def analysis_times(cfg):
    if cfg["times"]:
        times = [pd.Timestamp(str(t)).to_pydatetime() for t in cfg["times"]]
    else:
        if cfg["start_time"] is None:
            sys.exit("set start_time (and end_time), or times")
        start = pd.Timestamp(str(cfg["start_time"])).to_pydatetime()
        end = pd.Timestamp(str(cfg["end_time"] or cfg["start_time"])).to_pydatetime()
        if end < start:
            sys.exit(f"end_time {end} is before start_time {start}")
        step = dt.timedelta(hours=int(cfg["hour_step"]))
        times = []
        t = start
        while t <= end:
            times.append(t)
            t += step
    return sorted(set(times))

####################

def build_params(cfg):
    params = YParams(cfg["config_filepath"])
    unknown = [k for k in cfg["params"] if k not in params]
    if unknown:
        sys.exit(f"params: unknown keys {unknown} -- not in {cfg['config_filepath']}")
    for key, val in cfg["params"].items():
        params[key] = None if val == "None" else to_builtin(val)

    if getattr(params, "data_source", "netcdf") == "ocelot3":
        from utils.dataloader_ocelot3_parquet import infer_grid_shape, ocelot3_in_chans, ocelot3_pad_multiple

        # The model is sized from img_size_x/y and in_chans -- same as train_ges_goes.py
        params["img_size_y"], params["img_size_x"] = infer_grid_shape(params.ocelot3_static_data_dir,
                                                                      pad_multiple=ocelot3_pad_multiple(params))
        params["in_chans"] = ocelot3_in_chans(params)
    return params


def load_model(params, checkpoint, device):
    model = LowResEncDec(params).to(device)
    ckpt, missing, unexpected = load_checkpoint_weights(model, checkpoint, device)
    missing = [k for k in missing if "attn_mask" not in k]
    if missing:
        raise RuntimeError(f"{len(missing)} model weights missing from {checkpoint} (first: {missing[:5]}) -- "
                           "do config_filepath/params match the training run?")
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys in checkpoint (first: {unexpected[:5]})")
    model.eval()
    print(f"Loaded {checkpoint} (epoch {ckpt.get('epoch')}, iters {ckpt.get('iters')})")
    return model


class Ocelot3Source:
    """One Ocelot3ParquetDataset per calendar year (the dataset's date range must lie in one year)."""

    def __init__(self, params, times):
        from utils.dataloader_ocelot3_parquet import Ocelot3ParquetDataset

        by_year = {}
        for t in times:
            by_year.setdefault(t.year, []).append(t)
        self.datasets = {}
        for year, ts in by_year.items():
            self.datasets[year] = Ocelot3ParquetDataset(params, min(ts).strftime("%Y-%m-%d"),
                                                        max(ts).strftime("%Y-%m-%d"), train=False)

    def run(self, model, t, params, device, include_metar):
        dataset = self.datasets[t.year]
        idx = dataset.binned_samples.index(t.strftime("date=%Y-%m-%d_%H"))
        return run_model_inference_ocelot3(model, dataset, idx, params, device)


class NetcdfSource:
    def __init__(self, params, cfg):
        self.data_dir = cfg["data_dir"] or params.test_data_path
        self.stats_path = cfg["stats_path"]

    def run(self, model, t, params, device, include_metar):
        nc_path = os.path.join(self.data_dir, t.strftime("%Y-%m-%d_%H.nc"))
        if not os.path.isfile(nc_path):
            raise FileNotFoundError(nc_path)
        return run_model_inference(model, nc_path, params, self.stats_path, device, include_metar=include_metar)


def mode_settings(mode, cfg, base_ratio):
    """(params overrides, include_metar) for an obs mode -- as in evaluation.py."""
    if mode == "all_obs":
        return {"hold_out_obs": False}, True
    if mode == "heldout":
        ratio = cfg["heldout_ratio"] if cfg["heldout_ratio"] is not None else base_ratio
        return {"hold_out_obs": True, "hold_out_obs_ratio": float(ratio)}, True
    return {"hold_out_obs": True, "hold_out_obs_ratio": 1.0}, False  # no_obs

####################

def diff_stats(pred, ref, mask=None):
    valid = np.isfinite(pred) & np.isfinite(ref)
    if mask is not None:
        valid &= mask
    y = pred[valid].astype(np.float64)
    x = ref[valid].astype(np.float64)
    n = x.size
    if n == 0:
        return {"n": 0, "bias": np.nan, "rmse": np.nan, "mae": np.nan, "corr": np.nan}
    d = y - x
    corr = np.corrcoef(x, y)[0, 1] if n > 1 and np.std(x) > 0 and np.std(y) > 0 else np.nan
    return {"n": int(n), "bias": float(d.mean()), "rmse": float(np.sqrt(np.mean(d ** 2))),
            "mae": float(np.abs(d).mean()), "corr": float(corr)}


def metric_rows(t, mode, results, var_names):
    """Verification of the model analysis (and, for comparison, the background and the target
    analysis) against the target analysis on the grid and against the station obs."""
    heldout = np.asarray(results["heldout_mask"]) > 0
    sources = {"model": results["prediction_analysis_unnorm"],
               "ges": results["inp_pred_unnorm"],
               "anl": results["target_analysis_unnorm"]}
    rows = []
    for i, var in enumerate(var_names):
        obs = results["obs_unnorm"][i]
        checks = [("grid_vs_anl", results["target_analysis_unnorm"][i], None, ("model", "ges")),
                  ("obs", obs, None, ("model", "ges", "anl"))]
        if mode == "heldout":
            checks.append(("heldout_obs", obs, heldout, ("model", "ges", "anl")))
        for verif, ref, mask, source_names in checks:
            for source in source_names:
                rows.append({"time": t.strftime("%Y-%m-%d %H:00"), "mode": mode, "source": source,
                             "verif": verif, "var": var, **diff_stats(sources[source][i], ref, mask)})
    return rows


def summarize(df):
    """Pool per-time metrics over all times (n-weighted bias/MAE/MSE; corr is the mean over times)."""
    df = df[df["n"] > 0].copy()
    df["sum_d"] = df["bias"] * df["n"]
    df["sum_d2"] = df["rmse"] ** 2 * df["n"]
    df["sum_ad"] = df["mae"] * df["n"]
    g = df.groupby(["mode", "source", "verif", "var"], sort=False)
    out = g.agg(n_times=("time", "nunique"), n=("n", "sum"), sum_d=("sum_d", "sum"),
                sum_d2=("sum_d2", "sum"), sum_ad=("sum_ad", "sum"), corr_mean=("corr", "mean")).reset_index()
    out["bias"] = out["sum_d"] / out["n"]
    out["rmse"] = np.sqrt(out["sum_d2"] / out["n"])
    out["mae"] = out["sum_ad"] / out["n"]
    return out[["mode", "source", "verif", "var", "n_times", "n", "bias", "rmse", "mae", "corr_mean"]]

####################

def latlon_coords(lat, lon, H, W):
    lat, lon = np.asarray(lat), np.asarray(lon)
    if lat.ndim == 1 and lon.ndim == 1 and lat.size == H and lon.size == W:
        return {"lat": ("y", lat), "lon": ("x", lon)}
    if lat.shape == (H, W) and lon.shape == (H, W):
        return {"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)}
    return {}


def save_fields(path, t, results_by_mode, var_names, attrs):
    first = next(iter(results_by_mode.values()))
    _, H, W = first["prediction_analysis_unnorm"].shape
    dims = ("var", "y", "x")
    data_vars = {
        "background": (dims, first["inp_pred_unnorm"]),
        "target_analysis": (dims, first["target_analysis_unnorm"]),
        "obs": (dims, first["obs_unnorm"]),
    }
    for mode, res in results_by_mode.items():
        data_vars[f"analysis_{mode}"] = (dims, res["prediction_analysis_unnorm"])
        if mode == "heldout":
            data_vars["heldout_mask"] = (("y", "x"), (np.asarray(res["heldout_mask"]) > 0).astype(np.int8))

    ds = xr.Dataset(data_vars, coords={"var": var_names, **latlon_coords(first["lat"], first["lon"], H, W)},
                    attrs={**attrs, "analysis_time": t.strftime("%Y-%m-%dT%H:00"),
                           "units": ", ".join(f"{v}: {UNITS.get(v, '?')}" for v in var_names)})
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32" if name != "heldout_mask" else "int8"}
                for name in data_vars}
    tmp = path + ".tmp"
    ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, path)


def save_fields_pt(path, t, results_by_mode, var_names, attrs):
    """Same fields as save_fields, as a torch.save dict of CPU float32 tensors (var, y, x).
    Adds the model's raw normalized output per mode (residual if learn_residual).
    Load with torch.load(path)."""
    first = next(iter(results_by_mode.values()))
    as_tensor = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
    out = {
        "analysis_time": t.strftime("%Y-%m-%dT%H:00"),
        "var_names": list(var_names),
        "units": {v: UNITS.get(v, "?") for v in var_names},
        "attrs": dict(attrs),
        "lat": as_tensor(first["lat"]),
        "lon": as_tensor(first["lon"]),
        "background": as_tensor(first["inp_pred_unnorm"]),
        "target_analysis": as_tensor(first["target_analysis_unnorm"]),
        "obs": as_tensor(first["obs_unnorm"]),
        "analysis": {mode: as_tensor(res["prediction_analysis_unnorm"]) for mode, res in results_by_mode.items()},
        "prediction_norm": {mode: as_tensor(res["prediction_array_norm"]) for mode, res in results_by_mode.items()},
    }
    if "heldout" in results_by_mode:
        out["heldout_mask"] = torch.from_numpy(np.asarray(results_by_mode["heldout"]["heldout_mask"]) > 0)
    tmp = path + ".tmp"
    torch.save(out, tmp)
    os.replace(tmp, path)


def save_plots(plot_dir, t, results, var_names, channels, title_model):
    stamp = t.strftime("%Y-%m-%d_%H")
    when = t.strftime("%Y-%m-%d %H UTC")
    error = dict(results)
    error["prediction_analysis_unnorm"] = results["prediction_analysis_unnorm"] - results["target_analysis_unnorm"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # plt.show() on the Agg backend
        for var in channels:
            if var not in var_names:
                print(f"  plot_channels: unknown variable '{var}' (have {var_names})")
                continue
            channel = f"output_{var}"
            units = UNITS.get(var)
            plot_output_channel(error, channel, colorbar_scale_style="centered",
                                title_str=f"{var}, model minus anl, {when} ({title_model})",
                                colorbar_label=f"Error ({units})",
                                plot_savepath=os.path.join(plot_dir, f"error_{var}_{stamp}.png"))
            plot_scatter_pred_vs_target(results, channel, points="all", units=units,
                                        title_str=f"{var}, model vs anl, {when} ({title_model})",
                                        plot_savepath=os.path.join(plot_dir, f"scatter_{var}_{stamp}.png"))
            plt.close("all")

####################

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="predict YAML, e.g. configs/predict_example.yaml")
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE",
                        help="override an option (YAML value; dotted keys for nesting, e.g. params.seed=1)")
    args = parser.parse_args()

    cfg = load_predict_config(args.config, args.overrides)
    times = analysis_times(cfg)
    params = build_params(cfg)
    data_source = getattr(params, "data_source", "netcdf")
    base_ratio = params.hold_out_obs_ratio

    if cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg["device"])

    out_dir = cfg["output_dir"]
    fields_dir = os.path.join(out_dir, "fields")
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    if cfg["save_fields"]:
        os.makedirs(fields_dir, exist_ok=True)
    if cfg["plot_channels"]:
        os.makedirs(plot_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "metrics_per_time.csv")
    done = set()
    if cfg["skip_existing"] and os.path.isfile(metrics_path):
        done = set(pd.read_csv(metrics_path)["time"])
    elif os.path.isfile(metrics_path):
        os.remove(metrics_path)

    with open(os.path.join(out_dir, "predict_config_resolved.yaml"), "w") as f:
        YAML().dump({"predict": cfg, "model_params": dict(params.items())}, f)

    print(f"Predict run '{cfg['name']}': {len(times)} time(s) {times[0]:%Y-%m-%d %H} .. {times[-1]:%Y-%m-%d %H}, "
          f"modes {cfg['modes']}, data_source {data_source}, device {device}")
    print(f"Output: {out_dir}")

    model = load_model(params, cfg["checkpoint"], device)
    source = Ocelot3Source(params, times) if data_source == "ocelot3" else NetcdfSource(params, cfg)
    attrs = {"checkpoint": os.path.abspath(cfg["checkpoint"]), "config_filepath": cfg["config_filepath"],
             "data_source": data_source}
    title_model = cfg["name"]

    n_ok, failed = 0, []
    for k, t in enumerate(times, 1):
        stamp = t.strftime("%Y-%m-%d %H:00")
        if stamp in done:
            print(f"[{k}/{len(times)}] {stamp} already done, skipping")
            continue
        t0 = time.time()
        try:
            results_by_mode, rows = {}, []
            for mode in cfg["modes"]:
                overrides, include_metar = mode_settings(mode, cfg, base_ratio)
                for key, val in overrides.items():
                    params[key] = val
                results = source.run(model, t, params, device, include_metar)
                var_names = [n.split("output_", 1)[1] for n in results["output_channel_names"]]
                for key in ("prediction_tensor", "input_tensor"):  # free GPU memory
                    results.pop(key, None)
                results_by_mode[mode] = results
                rows += metric_rows(t, mode, results, var_names)
        except Exception as e:
            if not cfg["continue_on_error"]:
                raise
            failed.append(stamp)
            print(f"[{k}/{len(times)}] {stamp} FAILED: {type(e).__name__}: {e}")
            continue

        pd.DataFrame(rows).to_csv(metrics_path, mode="a", header=not os.path.isfile(metrics_path), index=False)
        if cfg["save_fields"]:
            base = os.path.join(fields_dir, t.strftime("%Y-%m-%d_%H"))
            if "nc" in cfg["field_formats"]:
                save_fields(base + ".nc", t, results_by_mode, var_names, attrs)
            if "pt" in cfg["field_formats"]:
                save_fields_pt(base + ".pt", t, results_by_mode, var_names, attrs)
        if cfg["plot_channels"]:
            plot_mode = "all_obs" if "all_obs" in results_by_mode else cfg["modes"][0]
            save_plots(plot_dir, t, results_by_mode[plot_mode], var_names, cfg["plot_channels"], title_model)

        n_ok += 1
        rmse = {r["var"]: r["rmse"] for r in rows
                if r["mode"] == cfg["modes"][0] and r["source"] == "model" and r["verif"] == "obs"}
        print(f"[{k}/{len(times)}] {stamp} done in {time.time() - t0:.1f}s | {cfg['modes'][0]} model-vs-obs RMSE "
              + " ".join(f"{v}={x:.3f}" for v, x in rmse.items()))

    if os.path.isfile(metrics_path):
        summary = summarize(pd.read_csv(metrics_path))
        summary.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
        with pd.option_context("display.max_rows", None, "display.width", 200, "display.precision", 4):
            print("\n" + summary.to_string(index=False))

    print(f"\n{n_ok} time(s) processed, {len(failed)} failed" + (f": {failed}" if failed else ""))
    sys.exit(1 if failed and n_ok == 0 else 0)


if __name__ == "__main__":
    main()
