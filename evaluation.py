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

model_number = 22046307  
resume_number = False 
model_name = "best_ckpt" #"best_ckpt"

base_dir = "/scratch3/NCEPDEV/da/Xin.C.Jin/git/adaf_rtma"
data_dir = f"/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data_ges_goes/" 
stats_path = f"{base_dir}/data_preparation_ges/stats_ges.csv"

#ckpt_path = f"{base_dir}/training_runs/lowres_{model_number}/{model_name}.tar"
ckpt_path = f"{base_dir}/checkpoint/{model_name}.tar"
if resume_number:
    ckpt_path = f"{base_dir}/training_runs/lowres_{model_number}_resume_{resume_number}/{model_name}.tar"
    

# Ensure model package is importable from notebook
if base_dir not in os.sys.path:
    os.sys.path.append(base_dir)

from models.encdec_lowres import LowResEncDec

config_filepath="./config/params_lowres_ges_goes.yaml"
params = YParams(config_filepath)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Checkpoint:", ckpt_path)
print("Stats:", stats_path)

# Load model
model = LowResEncDec(params).to(device)
ckpt, missing, unexpected = load_checkpoint_weights(model, ckpt_path, device)
model.eval()

print("Checkpoint keys:", list(ckpt.keys()))
print("Missing keys:", len(missing))
print("Unexpected keys:", len(unexpected))

# Change date as desired
analysis_time = dt.datetime(2023, 6, 13, 6) #(2022,10,13,4) 

# 2022
# nc_path = f"{data_dir}/valid_data/{analysis_time.strftime('%Y-%m-%d_%H.nc')}" 

# 2023
nc_path = f"{data_dir}/test_data/{analysis_time.strftime('%Y-%m-%d_%H.nc')}" 



## 1. Model given all obs
params["hold_out_obs"] = False #Don't hold out any obs

results_all_obs = run_model_inference(model, nc_path, params, stats_path, device, include_metar=True)


## 3. Model given no obs
params["hold_out_obs"] = True
params["hold_out_obs_ratio"] = 1 #Deny everything

results_no_obs = run_model_inference(model, nc_path, params, stats_path, device, include_metar=False)

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

channel = "output_t"

plot_output_channel(results_all_obs, 
                    f"{channel}", 
                    channel_to_select='prediction_analysis_unnorm',
                    title_str=f"{channel}, all obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})", 
                    # colorbar_scale_style='extreme',
                    # abs_min=-10, abs_max=45,
                    colorbar_label=f"Value ({units_dict[channel]})", 
                    # plot_savepath='./aa') 
                    plot_savepath=f"{plot_dir}/{channel}_model_{model_number}_{model_name}.png") 
                    # plot_savepath=f"{plot_dir}/ges/{channel}_model_{model_number}_{model_name}.png") 


plot_output_channel(results_all_obs, 
                    f"{channel}", 
                    title_str=f"{channel}, model innovation, all obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                    channel_to_select="prediction_residual_unnorm",
                    colorbar_scale_style='extreme',
                    abs_min=-5, abs_max=5,
                    colorbar_label=f"Value ({units_dict[channel]})", 
                    #plot_savepath='./bb')
                    plot_savepath=f"{plot_dir}/innovation_{channel}_model_{model_number}_{model_name}.png") 
                    # plot_savepath=f"{plot_dir}/ges/innovation_{channel}_model_{model_number}_{model_name}.png") 


plot_output_channel(results_all_obs_minus_rtma, 
                    f"{channel}", 
                    title_str=f"{channel}, model error (output minus RTMA anl), all obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                    channel_to_select="prediction_analysis_unnorm",
                    colorbar_scale_style='extreme',
                    abs_min=-5, abs_max=5,
                    colorbar_label=f"Value ({units_dict[channel]})", 
                    # plot_savepath='./cc')
                    plot_savepath=f"{plot_dir}/error_{channel}_model_{model_number}_{model_name}.png") 
                    # plot_savepath=f"{plot_dir}/ges/error_{channel}_model_{model_number}_{model_name}.png") 

plot_output_channel(results_hrrr_minus_rtma, 
                    f"{channel}", 
                    title_str=f"{channel}, RTMA ges error (ges minus anl), {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                    channel_to_select="prediction_analysis_unnorm",
                    colorbar_scale_style='extreme',
                    abs_min=-5, abs_max=5,
                    colorbar_label=f"Value ({units_dict[channel]})", 
                    #plot_savepath='./dd')
                    plot_savepath=f"{plot_dir}/error_hrrr_{channel}_tmp.png")
                    # plot_savepath=f"{plot_dir}/ges/error_ges_{channel}.png")

plot_output_channel(results_all_obs_minus_no_obs, 
                    f"{channel}", 
                    title_str=f"{channel} comparison, all obs minus no obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})", 
                    colorbar_scale_style='extreme',
                    abs_min=-4, abs_max=4,
                    colorbar_label=f"Difference ({units_dict[channel]})", 
                    #plot_savepath='./ee')
                    plot_savepath=f"{plot_dir}/difference_noobs_{channel}_model_{model_number}_{model_name}.png") 
                    # plot_savepath=f"{plot_dir}/ges/difference_noobs_{channel}_model_{model_number}_{model_name}.png") 
## 5. Scatter: predicted vs. target

# Model analysis vs. RTMA analysis over the whole grid
plot_scatter_pred_vs_target(results_all_obs,
                            f"{channel}",
                            points="all",
                            title_str=f"{channel}, model vs RTMA anl, all obs, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                            units=units_dict[channel],
                            plot_savepath=f"{plot_dir}/scatter_{channel}_model_{model_number}_{model_name}.png")

# Background (RTMA ges) vs. RTMA analysis, for comparison
plot_scatter_pred_vs_target(results_all_obs,
                            f"{channel}",
                            pred_key="inp_pred_unnorm",
                            points="all",
                            title_str=f"{channel}, RTMA ges vs anl, {analysis_time.strftime('%Y-%m-%d %H')} UTC",
                            units=units_dict[channel],
                            plot_savepath=f"{plot_dir}/scatter_ges_{channel}.png")

# Only at station cells
plot_scatter_pred_vs_target(results_all_obs,
                            f"{channel}",
                            points="obs",
                            title_str=f"{channel}, model vs RTMA anl at station cells, {analysis_time.strftime('%Y-%m-%d %H')} UTC ({title_model})",
                            units=units_dict[channel],
                            plot_savepath=f"{plot_dir}/scatter_obs_{channel}_model_{model_number}_{model_name}.png")
