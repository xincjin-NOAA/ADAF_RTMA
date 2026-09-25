"""Standalone smoke test for the Ocelot3 Parquet data source (docs/OCELOT3_ADAPTER.md).

Checks shapes against real data, not accuracy. Stages, in order, so a failure
tells you which one broke:
  1. infer_grid_shape() reads urma2p5_terrain.npy's native (H, W).
  2. Ocelot3ParquetDataset.__init__ opens a state/ges bin and reshapes its lat/lon.
  3. verify_grid_alignment() checks that reshape forms a row-major raster.
  4. dataset[0] pulls one full training sample (raw components).
  5. read_ocelot3_sample_for_inference() pulls one inference sample with hold-out.

Usage (no GPU needed):
    python test_ocelot3_pipeline.py \\
        --config_filepath ./config/params_lowres_ocelot3.yaml \\
        --date 2022-02-01
"""
import argparse

import numpy as np

from utils.YParams import YParams
from utils.dataloader_ocelot3_parquet import (
    Ocelot3ParquetDataset,
    infer_grid_shape,
    ocelot3_in_chans,
    ocelot3_pad_multiple,
    read_ocelot3_sample_for_inference,
)
from utils.ocelot3_grid_source import verify_grid_alignment


def print_arrays(named_arrays):
    for name, arr in named_arrays:
        arr = np.asarray(arr)
        nan_frac = np.isnan(arr.astype(float)).mean() if arr.size else 0.0
        print(f"   {name:17s} shape={arr.shape} dtype={arr.dtype} nan_frac={nan_frac:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filepath", default="./config/params_lowres_ocelot3.yaml")
    parser.add_argument("--data_dir", default=None, help="override ocelot3_data_dir")
    parser.add_argument("--static_data_dir", default=None, help="override ocelot3_static_data_dir")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD, used as both start and end date")
    args = parser.parse_args()

    params = YParams(args.config_filepath)
    if args.data_dir:
        params["ocelot3_data_dir"] = args.data_dir
    if args.static_data_dir:
        params["ocelot3_static_data_dir"] = args.static_data_dir

    print("=" * 60)
    print("1. infer_grid_shape()")
    pad_multiple = ocelot3_pad_multiple(params)
    native = infer_grid_shape(params.ocelot3_static_data_dir)
    params["img_size_y"], params["img_size_x"] = infer_grid_shape(params.ocelot3_static_data_dir, pad_multiple)
    params["in_chans"] = ocelot3_in_chans(params)
    print(f"   native (H, W) = {native}, model (H, W) = {(params.img_size_y, params.img_size_x)} "
          f"(pad_multiple={pad_multiple}), in_chans = {params.in_chans}")

    print("=" * 60)
    print("2. Ocelot3ParquetDataset.__init__")
    dataset = Ocelot3ParquetDataset(params, args.date, args.date, train=False)
    print(f"   {len(dataset)} bin(s) in range")
    if len(dataset) == 0:
        print("   No bins in range -- try another --date or confirm data exists for it.")
        return

    print("=" * 60)
    print("3. verify_grid_alignment()")
    verify_grid_alignment(dataset.grid_lat.ravel(), dataset.grid_lon.ravel(), dataset.grid_shape)
    print("   OK -- lat is row-constant, lon is column-constant after reshape.")

    print("=" * 60)
    print("4. dataset[0] -- raw components for Trainer.prepare_batch")
    sample = dataset[0]
    names = ["inp_pred", "inp_obs", "inp_sat", "topo", "field_tar", "obs_tar",
             "field_mask", "obs_tar_mask", "lat", "lon"]
    print_arrays(zip(names, sample))
    n_chans = sum(np.asarray(sample[i]).shape[0] for i in range(4))
    assert n_chans == params.in_chans, f"assembled input has {n_chans} channels, expected {params.in_chans}"
    print(f"   assembled input channels = {n_chans} (matches in_chans)")

    print("=" * 60)
    print("5. read_ocelot3_sample_for_inference(dataset, 0, ...)")
    inf = read_ocelot3_sample_for_inference(dataset, 0, hold_out_obs_ratio=0.2, seed=1)
    names = ["inp", "inp_pred", "field_tar", "hold_out_obs", "inp_obs_for_eval", "field_mask", "lat", "lon"]
    print_arrays(zip(names, inf))
    hold_out_obs, inp_obs_for_eval = inf[3], inf[4]
    print(f"   stations held out={int(np.count_nonzero(hold_out_obs[0]))}, "
          f"shown to model={int(np.count_nonzero(inp_obs_for_eval[0]))}")

    print("=" * 60)
    print("All checks passed.")


if __name__ == "__main__":
    main()
