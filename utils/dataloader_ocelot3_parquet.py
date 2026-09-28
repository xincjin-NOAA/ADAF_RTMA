"""ADAF-RTMA Dataset that reads Ocelot3's URMA Parquet data via Ocelot3's own
parsing/QC/normalization code (installed from the standalone `orca-common`
package, https://github.com/xincjin-NOAA/orca-common), instead of the
per-timestep NetCDF files read by utils/dataloader_multifiles_ges_goes.py.

Ported from the adaf repo (utils/data_loader_ocelot3_parquet.py) and adapted
to this repo's conventions -- see docs/OCELOT3_ADAPTER.md. Differences from
the adaf version:
  * __getitem__ returns the *raw* 10-tuple this repo's Trainer.prepare_batch
    expects (gpu_assemble): (inp_pred, inp_obs, inp_sat, topo, field_tar,
    obs_tar, field_mask, obs_tar_mask, lat, lon). field_obs_tar, the
    residual and the input concat are done on the GPU by the Trainer, not here.
  * Channel order follows params.target_vars (q, t, u10, v10 by default)
    instead of a fixed (t, q, u10, v10), so per-channel loss logs line up.
  * inp_sat is an empty (0, H, W) array -- Ocelot3 has no satellite source
    (its observation_config "satellite" block is empty).
  * Padding: LowResEncDec (arch: lowres) reflect-pads its input internally,
    so by default this path stays on the native grid. For the flat EncDec the
    grid is padded up to a multiple of patch_size * window_size, as in adaf.
    Override with params.ocelot3_pad_multiple.
  * Sampling reuses this repo's FractionalDistributedSampler / seeded
    RandomSampler, same as get_data_loader for the NetCDF path.
  * Each Parquet bin is fetched once per hour (not once per variable), and
    the obs->grid KDTree is built once per dataset.

Design decisions carried over unchanged from adaf:
  * No re-gridding: ges/anal fields are read on Ocelot3's native grid and only
    reshaped (utils/ocelot3_grid_source.py). Point observations are rasterized
    (nearest cell) onto that grid.
  * Normalization is Ocelot3's own z-score (FEATURE_STATS), not this repo's
    min-max stats -- losses are not comparable across the two data sources.
  * Schema is the aardvark-branch obs_config_urma_ok.py (surface_obs_t/u/v/q,
    spfh_2maboveground), confirmed against the real urma_ok data.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler

try:
    from orca_common.dataset_timeseries import ParquetDataManager, generate_binned_timestamp_list
    from orca_common.observation_config import load_observation_config
except ModuleNotFoundError as e:
    if e.name != "orca_common":
        raise
    raise ModuleNotFoundError(
        "data_source=ocelot3 needs the orca_common package in this environment: "
        "pip install -e /path/to/orca-common  (or pip install git+https://github.com/xincjin-NOAA/orca-common). "
        "See docs/OCELOT3_ADAPTER.md."
    ) from e
from utils.dataloader_multifiles_ges_goes import FractionalDistributedSampler
from utils.ocelot3_grid_source import (
    build_grid_tree,
    flat_to_grid,
    load_static_grid,
    pad_shape,
    pad_to_shape,
    rasterize_points_to_grid,
    split_pad,
)

# ges_cfg["features"] names for each state variable, and the single-feature
# conventional instrument for each -- from orca_common's obs_config_urma_ok.py.
STATE_FEATURE_NAMES = {
    "t": "tmp_2maboveground",
    "q": "spfh_2maboveground",
    "u10": "ugrd_10maboveground",
    "v10": "vgrd_10maboveground",
}
CONVENTIONAL_INSTRUMENTS = {"t": "diag_t", "q": "diag_q", "u10": "diag_u", "v10": "diag_v"}

####################

def ocelot3_pad_multiple(params):
    """How far to pad the native grid. LowResEncDec pads internally, so 1 (no
    padding) there; the flat EncDec needs patch_size * window_size."""
    explicit = getattr(params, "ocelot3_pad_multiple", None)
    if explicit is not None:
        return int(explicit)
    if str(getattr(params, "arch", "") or "").lower() == "lowres":
        return 1
    return int(getattr(params, "patch_size", 1)) * int(getattr(params, "window_size", 1))


def infer_grid_shape(static_data_dir, pad_multiple=1):
    """(H, W) to set params.img_size_y/x to before the model is built."""
    _, _, (h, w) = load_static_grid(static_data_dir)
    return pad_shape(h, w, pad_multiple)


def ocelot3_in_chans(params):
    """Model input channels for this path: background + obs window + topography (no satellite)."""
    n_vars = len(params.target_vars)
    return n_vars * (1 + params.obs_time_window) + 1

####################

class Ocelot3ParquetDataset(Dataset):
    """Drop-in replacement for dataloader_multifiles_ges_goes.GetDataset
    that sources ges/anal/obs from Ocelot3 Parquet files."""

    def __init__(self, params, start_date, end_date, train):
        self.params = params
        self.train = train
        self.data_dir = params.ocelot3_data_dir
        self.static_data_dir = params.ocelot3_static_data_dir
        self.obs_time_window = params.obs_time_window
        self.delta_time = int(getattr(params, "ocelot3_delta_time", 1) or 1)

        self.state_vars = list(params.target_vars)
        unknown = [v for v in self.state_vars if v not in STATE_FEATURE_NAMES]
        if unknown:
            raise ValueError(
                f"target_vars {unknown} have no Ocelot3 mapping; "
                f"supported: {list(STATE_FEATURE_NAMES)}"
            )

        (
            self.observation_config,
            self.feature_stats,
            self.fill_values,
            _instrument_weights,
            _increment_stats,
        ) = load_observation_config(exp_type="rtma_ok", config_name="urma_ok")

        if "state" not in self.observation_config or "ges" not in self.observation_config["state"]:
            raise ValueError("observation_config has no state/ges instrument -- check orca_common's obs_config_urma_ok.py.")
        self.ges_cfg = self.observation_config["state"]["ges"]
        self.ges_feature_idx = {
            v: self.ges_cfg["features"].index(STATE_FEATURE_NAMES[v]) for v in self.state_vars
        }

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if start.year != end.year:
            # generate_binned_timestamp_list takes MM/DD + a single year
            raise ValueError(f"Ocelot3 date range must lie within one year; got {start_date} .. {end_date}")
        self.binned_samples = generate_binned_timestamp_list(
            start.strftime("%m/%d"), end.strftime("%m/%d"), start.year, self.delta_time
        )

        self.loader = ParquetDataManager(
            data_dir=self.data_dir,
            observation_config=self.observation_config,
            feature_stats=self.feature_stats,
            fill_values=self.fill_values,
            delta_time=self.delta_time,
        )
        # anal isn't a top-level obs_config key -- it's ges's own config pointed
        # at anal_zarr_name, sharing ges's normalization stats so anal and ges
        # are in the same z-scored space.
        anal_zarr_name = self.ges_cfg.get("anal_zarr_name", "anal_urma")
        anal_obs_config = {"state": {"anal": {**self.ges_cfg, "zarr_name": anal_zarr_name}}}
        anal_feature_stats = {**self.feature_stats, "anal": self.feature_stats.get("ges", {})}
        self.anal_loader = ParquetDataManager(
            data_dir=self.data_dir,
            observation_config=anal_obs_config,
            feature_stats=anal_feature_stats,
            fill_values=self.fill_values,
            delta_time=self.delta_time,
        )

        self.terrain, self.slmask, self.grid_shape = load_static_grid(self.static_data_dir)
        pad_multiple = ocelot3_pad_multiple(params)
        self.padded_shape = pad_shape(*self.grid_shape, pad_multiple)
        if (params.img_size_y, params.img_size_x) != self.padded_shape:
            raise ValueError(
                f"params.img_size_y/x = {(params.img_size_y, params.img_size_x)} does not match "
                f"{self.padded_shape} (native grid {self.grid_shape} padded to a multiple of "
                f"{pad_multiple}) -- set them from infer_grid_shape() before building the model."
            )

        probe = self.loader.get_data_for_bin(self.binned_samples[0])
        if "ges" not in probe.get("state", {}):
            raise KeyError(
                f"No 'ges' data returned for bin {self.binned_samples[0]!r} -- check that a matching "
                f"Hive partition (<file_base>_<year>.parquet/date=.../cycle=...) exists under "
                f"data_dir={self.data_dir!r}."
            )
        ges_probe = probe["state"]["ges"]
        self.grid_lat = flat_to_grid(ges_probe["lat_deg"], self.grid_shape, name="ges.lat_deg")
        self.grid_lon = flat_to_grid(ges_probe["lon_deg"], self.grid_shape, name="ges.lon_deg")
        self.grid_tree = build_grid_tree(self.grid_lat, self.grid_lon)

    ###

    def __len__(self):
        return len(self.binned_samples)

    ###

    def _state_grid_and_mask(self, loader, state_key, bin_name):
        inst = loader.get_data_for_bin(bin_name)["state"][state_key]
        feats = inst["features_norm"].numpy()           # (N, target_dim), z-scored
        valid = inst["features_valid_mask"].numpy()     # (N, target_dim), bool
        grids, masks = [], []
        for v in self.state_vars:
            col = self.ges_feature_idx[v]
            grids.append(flat_to_grid(feats[:, col], self.grid_shape, name=f"{state_key}.{v}"))
            masks.append(flat_to_grid(valid[:, col], self.grid_shape, name=f"{state_key}.{v}.valid"))
        return np.stack(grids, axis=0).astype(np.float32), np.stack(masks, axis=0)

    def _rasterize_obs(self, bin_name):
        """(n_vars, H, W) rasterized station obs for one hourly bin, 0 = no station."""
        conv = self.loader.get_data_for_bin(bin_name).get("conventional", {})
        out = np.zeros((len(self.state_vars),) + tuple(self.grid_shape), dtype=np.float32)
        for i, v in enumerate(self.state_vars):
            inst = conv.get(CONVENTIONAL_INSTRUMENTS[v])
            if inst is None:
                continue
            val = inst["features_norm"].numpy()[:, 0]  # single-feature instrument
            valid = inst["features_valid_mask"].numpy()[:, 0]
            out[i] = rasterize_points_to_grid(
                inst["lat_deg"][valid], inst["lon_deg"][valid], val[valid],
                self.grid_lat, self.grid_lon, self.grid_shape, tree=self.grid_tree,
            )
        return out

    @staticmethod
    def _bin_name_at_offset(bin_name, hours_back):
        # bin_name format: "date=YYYY-MM-DD_HH"
        date_part, hour = bin_name.split("=", 1)[1].rsplit("_", 1)
        dt = pd.to_datetime(date_part) + pd.Timedelta(hours=int(hour) - hours_back)
        return f"date={dt.strftime('%Y-%m-%d')}_{dt.strftime('%H')}"

    def _read_obs_window(self, bin_name):
        """(n_vars, obs_time_window, H, W), oldest hour first -- same layout as the NetCDF sta_* vars."""
        hours = [self._rasterize_obs(self._bin_name_at_offset(bin_name, self.obs_time_window - 1 - k))
                 for k in range(self.obs_time_window)]
        return np.stack(hours, axis=1)

    @staticmethod
    def _build_obs_mask(reference_grid, ratio, seed):
        """Boolean (H, W) mask of a random `ratio` fraction of nonzero (station) cells to hold out.
        Same shuffle/split as GetDataset.__getitem__ in dataloader_multifiles_ges_goes.py."""
        if seed != 0:  # 0 = random seed
            np.random.seed(seed)
        flat = reference_grid.reshape(-1)
        obs_indices = np.where(flat != 0)[0]
        hold_out_num = int(len(obs_indices) * ratio)
        np.random.shuffle(obs_indices)
        mask = np.zeros(flat.shape, dtype=bool)
        mask[obs_indices[:hold_out_num]] = True
        return mask.reshape(reference_grid.shape)

    def _padded_lat_lon(self):
        h, w = self.grid_shape
        top, bottom = split_pad(self.padded_shape[0] - h)
        left, right = split_pad(self.padded_shape[1] - w)
        lat = np.pad(self.grid_lat[:, 0], (top, bottom), mode="edge")
        lon = np.pad(self.grid_lon[0, :], (left, right), mode="edge")
        return lat, lon

    ###

    def __getitem__(self, idx):
        bin_name = self.binned_samples[idx]
        h, w = self.grid_shape

        inp_pred, field_mask = self._state_grid_and_mask(self.loader, "ges", bin_name)
        field_tar, _anal_mask = self._state_grid_and_mask(self.anal_loader, "anal", bin_name)

        obs = self._read_obs_window(bin_name)  # (n_vars, T, H, W)
        obs_tar = obs[:, -1]
        obs_tar_mask = obs_tar != 0

        # Hold out stations from the model's input only; the target keeps them.
        if self.params.hold_out_obs:
            obs_mask = self._build_obs_mask(obs[0, 0], self.params.hold_out_obs_ratio, self.params.obs_mask_seed)
            obs = obs * ~obs_mask
        inp_obs = obs.reshape((-1, h, w))

        topo = self.terrain[np.newaxis, :, :]
        inp_sat = np.zeros((0, h, w), dtype=np.float32)

        # Pad from the native grid up to params.img_size_y/x (no-op when equal).
        pad = lambda a: pad_to_shape(a, self.padded_shape)
        lat, lon = self._padded_lat_lon()

        return (pad(inp_pred), pad(inp_obs), pad(inp_sat), pad(topo), pad(field_tar), pad(obs_tar),
                pad(field_mask), pad(obs_tar_mask), lat, lon)

####################

def get_data_loader_ocelot3(params, date_range, distributed, train, fractional=False):
    """Same return contract as dataloader_multifiles_ges_goes.get_data_loader.

    date_range: (start_date, end_date) strings. fractional=True applies
    params.train_sample_fraction (the training split only).
    """
    start_date, end_date = date_range
    dataset = Ocelot3ParquetDataset(params, start_date, end_date, train)

    sample_fraction = float(getattr(params, "train_sample_fraction", 1.0)) if fractional else 1.0
    seed = int(getattr(params, "seed", 0))

    if distributed:
        if fractional:
            sampler = FractionalDistributedSampler(dataset, sample_fraction=sample_fraction, seed=seed, drop_last=True)
        else:
            sampler = DistributedSampler(dataset, shuffle=train, seed=seed)
    elif train:
        generator = torch.Generator()
        generator.manual_seed(seed)
        sampler = RandomSampler(dataset, replacement=False,
                                num_samples=max(1, int(len(dataset) * sample_fraction)),
                                generator=generator)
    else:
        sampler = None

    num_workers = params.num_data_workers
    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=num_workers,
        prefetch_factor=params.prefetch_factor if num_workers > 0 else None,
        shuffle=False,
        sampler=sampler if train else None,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset

####################

def read_ocelot3_sample_for_inference(dataset, idx, hold_out_obs_ratio, seed):
    """One sample for inference, with its own hold-out split (independent of
    training's obs_mask_seed).

    Returns (inp, inp_pred, field_tar, hold_out_obs, inp_obs_for_eval,
    field_mask, lat, lon), all padded to the model's img_size and in Ocelot3's
    z-scored space. inp is the full model input (background, obs, topography);
    field_tar is the residual vs. the background when params.learn_residual.
    """
    bin_name = dataset.binned_samples[idx]
    h, w = dataset.grid_shape

    inp_pred, field_mask = dataset._state_grid_and_mask(dataset.loader, "ges", bin_name)
    field_tar, _anal_mask = dataset._state_grid_and_mask(dataset.anal_loader, "anal", bin_name)

    obs = dataset._read_obs_window(bin_name)
    obs_mask = dataset._build_obs_mask(obs[0, 0], hold_out_obs_ratio, seed)

    obs_tar = obs[:, -1]                     # analysis-time, unmasked
    inp_obs_for_eval = obs_tar * ~obs_mask   # what the model was shown
    hold_out_obs = obs_tar * obs_mask        # scored against, never shown

    inp_obs = (obs * ~obs_mask).reshape((-1, h, w))
    topo = dataset.terrain[np.newaxis, :, :]

    if dataset.params.learn_residual:
        field_tar = field_tar - inp_pred

    inp = np.concatenate((inp_pred, inp_obs, topo), axis=0)

    pad = lambda a: pad_to_shape(a, dataset.padded_shape)
    lat, lon = dataset._padded_lat_lon()
    return (pad(inp), pad(inp_pred), pad(field_tar), pad(hold_out_obs), pad(inp_obs_for_eval),
            pad(field_mask), lat, lon)
