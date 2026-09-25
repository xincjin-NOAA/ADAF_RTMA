import os
import numpy as np
import pandas as pd
import torch
import xarray as xr
import matplotlib.pyplot as plt
import datetime as dt


from utils.misc_functions import *
from utils.inference_functions import *
from utils.YParams import *

data_source = "netcdf" # "netcdf" (RTMA ges/anl + GOES NetCDF) | "ocelot3" (URMA Parquet, docs/OCELOT3_ADAPTER.md)

model_number = 22046307
resume_number = False
model_name = "best_ckpt" #"best_ckpt"

base_dir = "/scratch3/NCEPDEV/da/Xin.C.Jin/git/adaf_rtma"

#ckpt_path = f"{base_dir}/training_runs/lowres_{model_number}/{model_name}.tar"
ckpt_path = f"{base_dir}/checkpoint/{model_name}.tar"
if resume_number:
    ckpt_path = f"{base_dir}/training_runs/lowres_{model_number}_resume_{resume_number}/{model_name}.tar"

# Change date as desired
analysis_time = dt.datetime(2023, 6, 13, 6) #(2022,10,13,4)

if data_source == "ocelot3":
    config_filepath = "./config/params_lowres_ocelot3.yaml"
    bg_name, anl_name = "URMA ges", "URMA anl"
else:
    config_filepath = "./config/params_lowres_ges_goes.yaml"
    bg_name, anl_name = "RTMA ges", "RTMA anl"

    data_dir = f"/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data_ges_goes/"
    stats_path = f"{base_dir}/data_preparation_ges/stats_ges.csv"

    # 2022
    # nc_path = f"{data_dir}/valid_data/{analysis_time.strftime('%Y-%m-%d_%H.nc')}"

    # 2023
    nc_path = f"{data_dir}/test_data/{analysis_time.strftime('%Y-%m-%d_%H.nc')}"


# Ensure model package is importable from notebook
if base_dir not in os.sys.path:
    os.sys.path.append(base_dir)

from models.encdec_lowres import LowResEncDec

params = YParams(config_filepath)

if data_source == "ocelot3":
    from utils.dataloader_ocelot3_parquet import (
        Ocelot3ParquetDataset, infer_grid_shape, ocelot3_in_chans, ocelot3_pad_multiple,
    )

    # The model is sized from img_size_x/y and in_chans, so set them from the Ocelot3 grid
    # before building it -- same as train_ges_goes.py.
    params["img_size_y"], params["img_size_x"] = infer_grid_shape(params.ocelot3_static_data_dir,
                                                                  pad_multiple=ocelot3_pad_multiple(params))
    params["in_chans"] = ocelot3_in_chans(params)

    # One-day dataset around analysis_time; obs from earlier hours are read by bin name, so the
    # obs window may reach into the previous day.
    day = analysis_time.strftime("%Y-%m-%d")
    ocelot3_dataset = Ocelot3ParquetDataset(params, day, day, train=False)
    ocelot3_idx = ocelot3_dataset.binned_samples.index(analysis_time.strftime("date=%Y-%m-%d_%H"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Data source:", data_source)
print("Checkpoint:", ckpt_path)
if data_source == "ocelot3":
    print("Ocelot3 data:", params.ocelot3_data_dir, ocelot3_dataset.binned_samples[ocelot3_idx])
else:
    print("NetCDF:", nc_path)
    print("Stats:", stats_path)

# Load model
model = LowResEncDec(params).to(device)
ckpt, missing, unexpected = load_checkpoint_weights(model, ckpt_path, device)
model.eval()

print("Checkpoint keys:", list(ckpt.keys()))
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))


def run_inference(include_metar=True):
    """Runs the model on analysis_time for the selected data_source with the current
    params hold-out settings. include_metar only applies to the NetCDF path."""
    if data_source == "ocelot3":
        return run_model_inference_ocelot3(model, ocelot3_dataset, ocelot3_idx, params, device)
    return run_model_inference(model, nc_path, params, stats_path, device, include_metar=include_metar)


## 1. Model given all obs
params["hold_out_obs"] = False #Don't hold out any obs

results_all_obs = run_inference(include_metar=True)


## 3. Model given no obs
params["hold_out_obs"] = True
params["hold_out_obs_ratio"] = 1 #Deny everything

results_no_obs = run_inference(include_metar=False)

## 4. Differences

unnorm_keys = [
    'prediction_residual_unnorm', 
    'target_residual_unnorm', 
    'inp_pred_unnorm', 
    'prediction_analysis_unnorm', 
    'target_analysis_unnorm'
]

results_all_obs_minus_no_obs = results_all_obs.copy()

for key in unnorm_keys:
    results_all_obs_minus_no_obs[key] = results_all_obs[key] - results_no_obs[key]

# Compare all obs output to RTMA
results_all_obs_minus_rtma = results_all_obs.copy()
results_all_obs_minus_rtma['prediction_analysis_unnorm'] = results_all_obs['prediction_analysis_unnorm'] - results_all_obs['target_analysis_unnorm']

#Compare HRRR to RTMA
results_hrrr_minus_rtma = results_all_obs.copy()
results_hrrr_minus_rtma['prediction_analysis_unnorm'] = results_all_obs['inp_pred_unnorm'] - results_all_obs['target_analysis_unnorm']

title_model = f"model {model_number}" if not resume_number else f"model {model_number}_resume_{resume_number}"

units_dict={'output_t':'C',
            'output_q':'g/kg',
            'output_u10':'m/s',
            'output_v10':'m/s'}

plot_dir = f"./figures"
os.makedirs(plot_dir, exist_ok=True)

channel = "output_t"

plot_output_channel(results_all_obs, 
                    f"{channel}", 
                    channel_to_select='prediction_analysis_unnorm',
                    title_str=f"{channel}, all obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})", 
                    # colorbar_scale_style='extreme',
                    # abs_min=-10, abs_max=45,
                    colorbar_label=f"Value ({units_dict[channel]})", 
                    # plot_savepath='./aa') 
                    plot_savepath=f"{plot_dir}/{data_source}_{channel}_model_{model_number}_{model_name}.png") 
                    # plot_savepath=f"{plot_dir}/ges/{data_source}_{channel}_model_{model_number}_{model_name}.png") 


plot_output_channel(results_all_obs, 
                    f"{channel}", 
                    title_str=f"{channel}, model innovation, all obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                    channel_to_select="prediction_residual_unnorm",
                    colorbar_scale_style='extreme',
                    abs_min=-5, abs_max=5,
                    colorbar_label=f"Value ({units_dict[channel]})", 
                    #plot_savepath='./bb')
                    plot_savepath=f"{plot_dir}/{data_source}_innovation_{channel}_model_{model_number}_{model_name}.png") 
                    # plot_savepath=f"{plot_dir}/ges/{data_source}_innovation_{channel}_model_{model_number}_{model_name}.png") 


plot_output_channel(results_all_obs_minus_rtma, 
                    f"{channel}", 
                    title_str=f"{channel}, model error (output minus {anl_name}), all obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                    channel_to_select="prediction_analysis_unnorm",
                    colorbar_scale_style='extreme',
                    abs_min=-5, abs_max=5,
                    colorbar_label=f"Value ({units_dict[channel]})", 
                    # plot_savepath='./cc')
                    plot_savepath=f"{plot_dir}/{data_source}_error_{channel}_model_{model_number}_{model_name}.png") 
                    # plot_savepath=f"{plot_dir}/ges/{data_source}_error_{channel}_model_{model_number}_{model_name}.png") 

plot_output_channel(results_hrrr_minus_rtma, 
                    f"{channel}", 
                    title_str=f"{channel}, {bg_name} error (ges minus anl), {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                    channel_to_select="prediction_analysis_unnorm",
                    colorbar_scale_style='extreme',
                    abs_min=-5, abs_max=5,
                    colorbar_label=f"Value ({units_dict[channel]})", 
                    #plot_savepath='./dd')
                    plot_savepath=f"{plot_dir}/{data_source}_error_hrrr_{channel}_tmp.png")
                    # plot_savepath=f"{plot_dir}/ges/{data_source}_error_ges_{channel}.png")

plot_output_channel(results_all_obs_minus_no_obs, 
                    f"{channel}", 
                    title_str=f"{channel} comparison, all obs minus no obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})", 
                    colorbar_scale_style='extreme',
                    abs_min=-4, abs_max=4,
                    colorbar_label=f"Difference ({units_dict[channel]})", 
                    #plot_savepath='./ee')
                    plot_savepath=f"{plot_dir}/{data_source}_difference_noobs_{channel}_model_{model_number}_{model_name}.png") 
                    # plot_savepath=f"{plot_dir}/ges/{data_source}_difference_noobs_{channel}_model_{model_number}_{model_name}.png") 
## 5. Scatter: predicted vs. target

# Model analysis vs. target analysis over the whole grid
plot_scatter_pred_vs_target(results_all_obs,
                            f"{channel}",
                            points="all",
                            title_str=f"{channel}, model vs {anl_name}, all obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                            units=units_dict[channel],
                            plot_savepath=f"{plot_dir}/{data_source}_scatter_{channel}_model_{model_number}_{model_name}.png")

# Background (ges) vs. analysis, for comparison
plot_scatter_pred_vs_target(results_all_obs,
                            f"{channel}",
                            pred_key="inp_pred_unnorm",
                            points="all",
                            title_str=f"{channel}, {bg_name} vs anl, {analysis_time.strftime('%Y-%m-%d %H')} UTC",
                            units=units_dict[channel],
                            plot_savepath=f"{plot_dir}/{data_source}_scatter_ges_{channel}.png")

# Only at station cells
plot_scatter_pred_vs_target(results_all_obs,
                            f"{channel}",
                            points="obs",
                            title_str=f"{channel}, model vs {anl_name} at station cells, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                            units=units_dict[channel],
                            plot_savepath=f"{plot_dir}/{data_source}_scatter_obs_{channel}_model_{model_number}_{model_name}.png")
# residual
plot_scatter_pred_vs_target(results_all_obs,
                            f"{channel}",
                            pred_key="prediction_residual_unnorm",
                            target_key="target_residual_unnorm",
                            points="all",
                            title_str=f"{channel}, model vs {anl_name} (residual), all obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                            units=units_dict[channel],
                            plot_savepath=f"{plot_dir}/{data_source}_scatter_{channel}_model_{model_number}_{model_name}_residual.png")


