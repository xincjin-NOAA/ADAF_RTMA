# Batch Inference (predict.py)

[predict.py](../predict.py) runs a trained checkpoint over a range of analysis times. It saves the analyses and computes verification metrics against the target analysis and the station obs. It works with both data sources: NetCDF+GOES and Ocelot3 Parquet.

[evaluation.py](../evaluation.py) is still the tool for looking closely at a single hour. `predict.py` is for many hours, run as a batch.

| Command | Use it for |
|---|---|
| `./submit_predict.sh <predict.yaml> [key=value ...]` | Submit a single-GPU SLURM job (run on a login node) |
| `python predict.py <predict.yaml> [key=value ...]` | Run directly, on a GPU node or interactively |

```bash
./submit_predict.sh configs/predict_example.yaml --dry-run          # preview the job script
./submit_predict.sh configs/predict_example.yaml end_time=2023-06-30T23:00 slurm.time=08:00:00
./submit_predict.sh configs/predict_ocelot3_example.yaml --debug    # 30-minute limit
python predict.py configs/predict_example.yaml start_time=2023-06-13T06:00 end_time=2023-06-13T06:00
```

Example configs: [configs/predict_example.yaml](../configs/predict_example.yaml) (NetCDF) and [configs/predict_ocelot3_example.yaml](../configs/predict_ocelot3_example.yaml).

`key=value` overrides are parsed as YAML. Dotted keys reach nested blocks, for example `params.hold_out_obs_ratio=0.2` or `slurm.time=04:00:00`.

## Submitter flags

| Flag | Effect |
|---|---|
| `--dry-run` | Write the job script and resolved config, but don't submit |
| `--debug` | Set the time limit to 00:30:00 |
| `--force` | Allow overwriting an existing `metrics_per_time.csv` in `output_dir` |

The submitter merges the overrides into the YAML and writes the result to `<output_dir>/predict_config.yaml`. The job runs `predict.py` on that copy, so a job can always be rerun from its output directory.

## Obs modes

Each analysis time is run once per mode in `modes`:

| Mode | Model input | What the `obs` verification means |
|---|---|---|
| `all_obs` | Every station | Fit to the obs the model was given |
| `heldout` | A seeded `heldout_ratio` of stations withheld (default: the config's `hold_out_obs_ratio`, 0.1) | Also adds `heldout_obs`: independent verification at the withheld stations |
| `no_obs` | No stations | Fully independent; shows what the model adds to the background from satellite and topography alone |

The withheld stations are chosen as in training: seeded by `obs_mask_seed` and the analysis time. The same stations are withheld on every run.

## Outputs (`output_dir`, default `predictions/<name>/`)

| File | Contents |
|---|---|
| `metrics_per_time.csv` | One row per time × mode × source × verification × variable, with `n`, `bias`, `rmse`, `mae` and `corr`. Written after each time, so a job that times out keeps its partial results. |
| `metrics_summary.csv` | The same metrics pooled over all times. `bias`, `rmse` and `mae` are weighted by `n`; `corr_mean` is the mean over times. |
| `fields/<YYYY-MM-DD_HH>.nc` | Written when `save_fields: true` and `field_formats` includes `nc`. Contains `background`, `target_analysis`, `obs` (NaN away from stations), `analysis_<mode>` for each mode, and `heldout_mask`, all in physical units and zlib-compressed. |
| `fields/<YYYY-MM-DD_HH>.pt` | Written when `field_formats` includes `pt`. A `torch.save` dict holding the same fields as CPU float32 tensors of shape (var, y, x): `background`, `target_analysis`, `obs`, `analysis[mode]`, `heldout_mask` (bool), `lat` and `lon`. It also has `prediction_norm[mode]`, the model's raw normalized output (a residual when `learn_residual`), plus `var_names`, `units`, `analysis_time` and `attrs`. Read it with `torch.load(path)`. It is uncompressed, so roughly 60 MB per field on the full NetCDF grid. |
| `plots/` | For each variable in `plot_channels`, an error map (model minus anl) and a scatter plot (model vs anl), using the `all_obs` run |
| `predict_config_resolved.yaml` | The options and model params actually used |

Metric columns:

- `source`: `model` (the model analysis), `ges` (the background) or `anl` (the target analysis: RTMA or URMA).
- `verif`: what the source is compared against.
  - `grid_vs_anl`: the target analysis, at every valid grid cell.
  - `obs`: the station obs at analysis time.
  - `heldout_obs`: the station obs at the withheld stations only.

To compare the model with the background and the operational analysis, read the rows with `verif = obs` or `heldout_obs` and compare the three sources.

Units are °C for `t`, kg/kg for `q` and m/s for `u10` and `v10`. On the Ocelot3 path, obs are converted back to physical units with their own conventional-instrument stats (`diag_t`, and so on), which differ from the ges stats.

## Option reference

| Key | Default | Meaning |
|---|---|---|
| `name` | file stem | Run name; used in `output_dir` and the job name |
| `description` | `""` | Free text |
| `config_filepath` | `./config/params_lowres_ges_goes.yaml` | Model/data config. Its `data_source` selects the NetCDF or Ocelot3 path |
| `checkpoint` | required | e.g. `training_runs/<name>/best_ckpt.tar` or `ckpt.tar` |
| `params` | `{}` | Overrides of `config_filepath` keys. Copy any architecture or data overrides the training run used, because the model is rebuilt from these. A mismatch fails with "model weights missing". |
| `start_time` / `end_time` | required / `start_time` | First and last analysis time, both inclusive, e.g. `2023-06-13T00:00` |
| `hour_step` | 1 | Hours between analysis times |
| `times` | `null` | An explicit list of times, used instead of start/end/step |
| `data_dir` | `test_data_path` | NetCDF only: the directory of `YYYY-MM-DD_HH.nc` files |
| `stats_path` | `./data_preparation_ges/stats_ges.csv` | NetCDF only: min-max stats for converting back to physical units |
| `modes` | `[all_obs, heldout, no_obs]` | See [Obs modes](#obs-modes) |
| `heldout_ratio` | `hold_out_obs_ratio` | Fraction of stations withheld in `heldout` mode |
| `device` | `auto` | `auto`, `cuda` or `cpu` |
| `output_dir` | `predictions/{name}` | `{name}` is replaced by `name` |
| `save_fields` | `true` | Write per-hour field files to `fields/`. For the full NetCDF grid, `.nc` is roughly 100–200 MB per hour. |
| `field_formats` | `[nc]` | Which field files to write: `nc`, `pt`, or both (`[nc, pt]`) |
| `plot_channels` | `[]` | Variables to plot, e.g. `[t, u10]` |
| `skip_existing` | `false` | Resume: skip times already in `metrics_per_time.csv`. When `false`, an existing file is replaced. |
| `continue_on_error` | `true` | Log and skip a time whose data is missing or unreadable |
| `slurm` | see below | Used only by `submit_predict.sh` |

`slurm` block: `account`, `partition`, `qos`, `cpus_per_task` (8), `mem` (`64G`), `time` (`02:00:00`), `extra_sbatch`, `env_setup`, `env_vars` and `python`. The job always uses one node and one GPU.

## Notes

- **Resuming after a timeout.** Resubmit with `skip_existing=true`, e.g. `./submit_predict.sh configs/predict_example.yaml skip_existing=true`.
- **Ocelot3 date ranges.** One dataset is built per calendar year, so a range can cross a year boundary. The obs window reads the `obs_time_window - 1` hours before each analysis time.
- **NetCDF obs.** `train_obs_source` from the config (default `metar`) also filters the obs at inference, so the `obs` verification uses only those stations.
