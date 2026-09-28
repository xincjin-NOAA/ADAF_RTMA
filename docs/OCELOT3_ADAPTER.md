# Ocelot3 Parquet Data Source

ADAF-RTMA can train on Ocelot3's URMA Parquet data as well as on its usual per-timestep NetCDF files. The Parquet files are read with Ocelot3's own parsing, QC and normalization code, which comes from the standalone [`orca-common`](https://github.com/xincjin-NOAA/orca-common) package (`orca_common`).

This is a port of the adapter in the sibling `adaf` repo (its `docs/OCELOT3_ADAPTER.md`), adapted to this repo's training pipeline. The NetCDF path is unchanged. The Ocelot3 path is only used when `data_source: "ocelot3"`.

## Files

| File | Purpose |
|---|---|
| [utils/dataloader_ocelot3_parquet.py](../utils/dataloader_ocelot3_parquet.py) | `Ocelot3ParquetDataset`, `get_data_loader_ocelot3`, `read_ocelot3_sample_for_inference`, and grid/channel helpers |
| [utils/ocelot3_grid_source.py](../utils/ocelot3_grid_source.py) | Reshapes to the native grid, rasterizes point obs and pads (copied from adaf, plus a reusable KDTree) |
| [config/params_lowres_ocelot3.yaml](../config/params_lowres_ocelot3.yaml) | `params_lowres_ges_goes.yaml` with the data section swapped for Ocelot3 |
| [configs/train_ocelot3_example.yaml](../configs/train_ocelot3_example.yaml) | Single-run submit config for `submit_train.sh` ([SUBMITTING_JOBS.md](SUBMITTING_JOBS.md)); `experiment_configs.yaml` has `ocelot3_*` experiments too |
| [test_ocelot3_pipeline.py](../test_ocelot3_pipeline.py) | CPU smoke test against real data |

## Setup

Install `orca_common` into the training environment:

```bash
pip install -e /path/to/orca-common          # sibling checkout
# or
pip install git+https://github.com/xincjin-NOAA/orca-common
```

It also needs `scipy` (for the KDTree), which is already in `ADAF_environment.yml`.

## Usage

1. Edit the `ocelot3_*` keys in `config/params_lowres_ocelot3.yaml`:

   | Key | Meaning |
   |---|---|
   | `ocelot3_data_dir` | Root directory that contains the `*.parquet` tables |
   | `ocelot3_static_data_dir` | Directory with `urma2p5_terrain.npy` and `urma2p5_slmask_nolakes.npz` |
   | `ocelot3_train_start_date` / `_end_date` | Training date range (`YYYY-MM-DD`) |
   | `ocelot3_valid_start_date` / `_end_date` | Validation date range (`YYYY-MM-DD`) |
   | `ocelot3_delta_time` | Hours between bins (default 1) |
   | `ocelot3_pad_multiple` | Grid padding; `None` means automatic (see Padding under Design) |

   Each start/end pair must fall within a single year, because `orca_common`'s bin generator takes month/day plus one year.

2. Run the smoke test. It needs no GPU:

   ```bash
   python test_ocelot3_pipeline.py --config_filepath ./config/params_lowres_ocelot3.yaml --date 2022-02-01
   ```

3. Train:

   ```bash
   ./submit_train.sh configs/train_ocelot3_example.yaml
   # or several variants at once:
   ./submit_experiments_from_yaml.sh - ocelot3
   ```

   Every config key is also a command-line flag of `train_ges_goes.py`, for example `--ocelot3_train_start_date 2022-03-01`, and a key in the submit YAML. The flags are generated from the chosen config file, so use `params_lowres_ocelot3.yaml` rather than adding `data_source` to the GOES config (it has no such key).

When `data_source` is `ocelot3`, `train_ges_goes.py`:

- sets `img_size_y/x` from the static terrain grid **before** building the model;
- sets `in_chans` to `4 background + 4 × obs_time_window obs + 1 topography` (17 for a 3-hour window);
- builds the train and valid loaders from the date ranges instead of directories.

The rest of the Trainer is unchanged: GPU assembly, loss, bf16, compile and Local-SGD all work the same way.

## Design

**Carried over from adaf:**

- **No re-gridding.** The `ges` (background) and `anal` (target) fields are read on Ocelot3's native grid and only reshaped. `urma_ok` is 171 × 331.
- **Station obs are rasterized.** The conventional obs `diag_t/q/u/v` are placed in the nearest grid cell, and several obs in one cell are averaged.
- **Normalization is Ocelot3's own z-score** (`FEATURE_STATS`), not this repo's min-max. `learn_residual` is applied in that z-scored space. **Losses and metrics are not comparable** between the NetCDF and Ocelot3 data sources.
- **No satellite channels.** Ocelot3's observation config has an empty satellite block.
- **Obs hold-out** hides `hold_out_obs_ratio` of the stations from the model's input only, never from the target. It uses the same shuffle/split and `obs_mask_seed` rule as the NetCDF loader.
- **Schema.** This targets the aardvark-branch `obs_config_urma_ok.py` (`surface_obs_t/u/v/q` tables, `spfh_2maboveground`). That schema has been confirmed against the real `urma_ok` data. Ocelot3's main branch uses a different, incompatible schema.

**Adapted for this repo:**

- **Raw components.** `__getitem__` returns the same 10-tuple as `dataloader_multifiles_ges_goes.GetDataset`: `(inp_pred, inp_obs, inp_sat, topo, field_tar, obs_tar, field_mask, obs_tar_mask, lat, lon)`. `Trainer.prepare_batch` then builds `field_obs_tar`, applies the residual and concatenates the input on the GPU. `inp_sat` is an empty `(0, H, W)` array, so the concatenation needs no special case.
- **Channel order follows `target_vars`** (q, t, u10, v10 by default). adaf hard-codes (t, q, u10, v10) and has to relabel channels downstream.
- **Padding.** `LowResEncDec` (`arch: lowres`) already reflect-pads its input to a multiple of `downscale × window_size` and crops the output, so by default the data stays on the native grid. For the flat `EncDec`, the grid is padded to a multiple of `patch_size × window_size` plus one extra block, as in adaf. Padded cells are marked invalid in `field_mask` and `obs_tar_mask`, so they don't count in the loss. `ocelot3_pad_multiple` overrides either default.
- **Sampling** uses this repo's `FractionalDistributedSampler` (which honours `train_sample_fraction`) and seeded `RandomSampler`.
- **Speed.** Each Parquet bin is fetched once per hour, instead of once per variable per hour. The observation-to-grid KDTree is built once per dataset, not on every rasterization.

## Inference

`read_ocelot3_sample_for_inference(dataset, idx, hold_out_obs_ratio, seed)` returns one fully assembled, padded sample with its own hold-out split: `(inp, inp_pred, field_tar, hold_out_obs, inp_obs_for_eval, field_mask, lat, lon)`. Everything stays in Ocelot3's z-scored space.

[evaluation.py](../evaluation.py) evaluates Ocelot3-trained models when `data_source = "ocelot3"`. It uses `run_model_inference_ocelot3(model, dataset, idx, params, device)` in [utils/inference_functions.py](../utils/inference_functions.py), which returns the same results dictionary as the NetCDF `run_model_inference`, so the same plots work for both sources. Outputs are converted back to physical units with the `ges` z-score stats from `FEATURE_STATS` (`anal` shares them), and temperature is converted from K to C. Hold-out follows `hold_out_obs`, `hold_out_obs_ratio` and `obs_mask_seed`, the same as the NetCDF path. Cells where `field_mask` is false are set to NaN.

It is **not** wired into `apply_lowres_to_ges_goes.ipynb` yet.

## Open items

- Not yet verified with a real training run in this repo. The smoke test checks only the pipeline mechanics.
- Only `train_ges_goes.py` is wired up. `train.py` (the non-satellite copy) is not.
- `orca_common` is a manual snapshot of Ocelot3's code. Re-sync it if Ocelot3's QC, stats or instrument config change.
- If you point this at a Parquet dataset other than `urma_ok`, re-run the smoke test. Its grid-alignment check (stage 3) confirms that the Parquet rows are row-major over the terrain grid.
