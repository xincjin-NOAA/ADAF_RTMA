# ADAF-RTMA vs. ADAF: Codebase Comparison

This document compares two forks of the ADAF model (https://github.com/microsoft/ADAF):

- **adaf_rtma** (this repo, `xincjin-NOAA/ADAF_RTMA`) changes the model itself. It has a new architecture that runs on the larger, higher-resolution RTMA grid and its own data pipeline.
- **adaf** (`xincjin-NOAA/adaf`) keeps the original Microsoft model and data unchanged. It adds a second data source (Ocelot3 Parquet), documentation and experiment tooling.

Snapshot compared: adaf_rtma at `b1d907c`, adaf at `67ca51f` (2026-09-25).

## 1. Purpose and domain

| | **adaf_rtma** | **adaf** |
|---|---|---|
| Goal | The full RTMA domain at 2.5 km | The original paper's grid (0.05°, CONUS) |
| Grid (x × y) | 2345 × 1597 | 1280 × 512 |
| Where it runs | Hard-coded for the Ursa supercomputer (`/scratch3`, `/scratch5` paths) | Anywhere; paths are relative (`./data/`) |
| Git remote | `xincjin-NOAA/ADAF_RTMA` | `xincjin-NOAA/adaf` |

## 2. Model architecture

This is the biggest difference between the two repos.

- **adaf** has only `models/encdec.py`, the original flat SwinIR-style `EncDec`. Every stage runs at full resolution.
- **adaf_rtma** adds [models/encdec_lowres.py](../models/encdec_lowres.py) with `LowResEncDec`. You select it through `build_model()` with `arch: lowres`:
  - Convolution layers shrink the input by 4× (`downscale: 4`) before the Swin attention, and PixelShuffle layers scale it back up afterwards. This makes attention affordable at 2.5 km and lets station observations influence a wider area.
  - A full-resolution skip connection: the head concatenates the stem features (`f0`) with the upsampled features.
  - `LayerScale2d` is a per-channel gate on the Swin body, initialised at `1e-4`. Without it, the model tends to collapse to copying the background through the skip connection.
  - Sizes from the config: `body_dim` 192, `depths` [4, 4], `num_heads` [6, 6], `window_size` 12, `stem_dims` [96, 128], `head_dims` [96, 96].
  - The old `pyramid` architecture (`MultiScaleEncDec`) was removed on purpose. `build_model()` raises an error saying it was the failure that `lowres` replaced.
- The two copies of `models/encdec.py` are the same apart from a few comments.

## 3. Inputs and data

| | **adaf_rtma** | **adaf** |
|---|---|---|
| Background | RTMA first guess (`rtma_ges_q/t/u10/v10`) | HRRR forecast (`hrrr_q/t/u_10/v_10`) |
| Observations | `sta_q/t/u10/v10` | `sta_q/t/u10/v10` |
| Satellite (4 GOES channels) | `goes_channel_02/07/10/14` | `CMI02/07/14/10` |
| Field target | `rtma_anl_*` | `rtma_*` |
| Data preparation | `data_preparation_ges/` (uses the RTMA 2.5 km terrain file, copied separately from Ursa) | `data_preparation/` (original) |
| Other data sources | None | Ocelot3 Parquet (see below) |

**Ocelot3 Parquet data source (adaf only):**

- `utils/data_loader_ocelot3_parquet.py` and `utils/ocelot3_grid_source.py`, backed by the external `orca_common` package.
- Smoke-tested end to end on the `urma_ok` dataset (a 171 × 331 grid) with `test_ocelot3_pipeline.py`. It has not yet been checked with a real training run.
- Documented in `docs/OCELOT3_ADAPTER.md`.

**Data loaders:**

- adaf_rtma adds a `FractionalDistributedSampler`, so each epoch can train on only part of the data (`train_sample_fraction`). It also adds seeded sampling and `prefetch_factor`.
- adaf uses a plain `DistributedSampler`.

## 4. Training script

Both scripts use the same three losses: the full field, the observations, and the field at observation points.

**adaf_rtma** ([train_ges_goes.py](../train_ges_goes.py)) is tuned for speed on HPC:

- Mixed precision in bfloat16 (`amp_dtype`), plus TF32.
- `torch.compile` (`compile_model`).
- `channels_last` memory layout.
- Losses are summed on the GPU and synced to the CPU once per epoch, instead of calling `.item()` at every step.
- Post-Local-SGD (`localsgd_h`, `localsgd_warmup`).
- Warmup (`warmup_iters`) and gradient clipping (`grad_clip`).
- Richer hold-out of observations:
  - a random hold-out ratio between 0.05 and 0.30;
  - holding out whole blocks (`hold_out_block_*`);
  - a held-out loss weight (`heldout_loss_weight`);
  - METAR-only training observations (`train_obs_source`).
- wandb is turned off.

**adaf** (`train.py`) stays closer to the original:

- wandb logging.
- A CSV loss log (`loss_history.csv`) that works without wandb.
- Plain AMP and DDP.
- A `--data_source ocelot3` branch.
- A `--learn_residual` override on the command line.

**Config loader (`utils/YParams.py`):**

- adaf_rtma:
  - allows `params.x` access;
  - overrides values from the command line (`override_from_cli`);
  - cleans values with `to_builtin` so saved checkpoints don't hit pickling problems.
- adaf: the original minimal loader, plus a `log()` method.

**Config layout:**

- adaf_rtma: one flat file, [config/params_lowres_ges_goes.yaml](../config/params_lowres_ges_goes.yaml).
- adaf: several named experiments that inherit through YAML anchors (`config/experiment.yaml`, `experiment_ocelot3.yaml`, `experiment_test.yaml`). It also has `experiment_configs.yaml` with `submit_experiments_from_yaml.sh`.

## 5. Inference

- **adaf_rtma:**
  - A short [inference.py](../inference.py) (161 lines) plus helpers in [utils/inference_functions.py](../utils/inference_functions.py).
  - The main workflow is [apply_lowres_to_ges_goes.ipynb](../apply_lowres_to_ges_goes.ipynb), which runs inference on one hour at a time.
- **adaf:** a full `inference.py` (866 lines) that supports:
  - Monte Carlo dropout;
  - Gaussian perturbation of inputs;
  - undoing the normalisation on outputs;
  - saving outputs;
  - an Ocelot3 test-date range.

  It also comes with a diagnostic tool, `utils/diagnose_boundary_error.py`.

## 6. Tooling and documentation

| | **adaf_rtma** | **adaf** |
|---|---|---|
| Environment | `ADAF_environment.yml` and `zz_cuda_headers.sh` (a CUDA setup workaround) | `environment.yml` and `requirements.txt` |
| Job scripts | sbatch launchers for new and resumed training runs | `submit_*.sh`, `training/*.sh`, `jobs/` |
| Documentation | `README.md` (setup on Ursa) | `docs/ARCHITECTURE.md`, `docs/OCELOT3_ADAPTER.md`, `docs/BOUNDARY_ERROR_DIAGNOSTIC.md` |
| Tests | None | `test_ocelot3_pipeline.py` |
| Other | | LICENSE, SECURITY.md |

adaf_rtma has two copies of its pipeline:

- a plain one: `train.py` + `utils/dataloader_multifiles.py`;
- a GOES one: `train_ges_goes.py` + `utils/dataloader_multifiles_ges_goes.py`.

They differ only in adding the satellite input and in the default config file.

## 7. Opportunities to cross-port

1. **Ocelot3 data into adaf_rtma.** *Done:* see [OCELOT3_ADAPTER.md](OCELOT3_ADAPTER.md). The Ocelot3 loader has been ported, and `train_ges_goes.py` can train `LowResEncDec` on Ocelot3 data (`data_source: "ocelot3"`).
2. **Speed work into adaf.** adaf_rtma's bf16, `torch.compile`, `channels_last`, fractional sampler and improved `YParams` would carry over cleanly.
3. **Merge adaf_rtma's two pipelines.** The satellite and non-satellite training scripts could become one script with a flag.
4. **Documentation into adaf_rtma.** An architecture document like adaf's `docs/ARCHITECTURE.md` would help. The README also notes that the file naming needs cleaning up.
