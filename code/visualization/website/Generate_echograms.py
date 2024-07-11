import echopype as ep
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cv2
import os
import random
import pandas as pd
import shutil
from pathlib import Path
from scipy.signal import medfilt 

from processing import extract_meta_data, remove_vertical_lines, clean_times
from find_bottom import get_beam_dead_zone, find_bottom
from find_fish import find_fish_median, medianfun
from find_waves import find_waves, find_layer
from visualization import data_to_images
from export_data import save_data, shorten_list

from matplotlib.colors import Normalize

def process_data(data_path, cal_params, env_params):
    raw_echodata = ep.open_raw(data_path, sonar_model="EK80")

    ds_Sv_raw = ep.calibrate.compute_Sv(
        raw_echodata,
        env_params=env_params,
        cal_params=cal_params,
        waveform_mode='BB',
        encode_mode="complex",
    )

    ds_MVBS = ep.commongrid.compute_MVBS(
        ds_Sv_raw,
        range_meter_bin=0.1,
        ping_time_bin='2S'
    )

    ds_MVBS = ds_MVBS.pipe(swap_chan)
    selected_data = ds_MVBS.Sv.to_numpy()[0]
    return selected_data

def swap_chan(ds: xr.Dataset) -> xr.Dataset:
    return (
        ds.set_coords("frequency_nominal")
        .swap_dims({"channel": "frequency_nominal"})
        .reset_coords("channel")
    )


def create_normal_echogram(data, filepath='echogram'):
    np_data = np.nan_to_num(data, copy=True)
    np_data[np_data < -90] = -90
    np_data = (np_data - np_data.min()) / (np_data.max() - np_data.min())
    np_data = np_data * 256

    flip_np_data = np.rot90(np_data, 3)

    name = path.name
    cv2.imwrite(f'dump/{name}_greyscale.png', flip_np_data)

    image = cv2.imread(f'dump/{name}_greyscale.png', 0)
    colormap = plt.get_cmap('viridis')
    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'dump/{name}_original.png', heatmap)


def create_echogram(data, cmap = "gnuplot", upper = -80, lower = -95, vmin = 0, vmax = 255, blur_size=5, blur_type="Kernels", kernel_first=5, kernel_second=200):
    np_data = np.nan_to_num(data, copy=True)
    np_data[np_data < lower] = lower
    np_data[np_data > upper] = upper
    np_data = (np_data - np_data.min()) / (np_data.max() - np_data.min())
    np_data = np_data * 256

    flip_np_data = np.rot90(np_data, 3)

    name = filename.name
    cv2.imwrite(f'dump/{name}_greyscale.png', flip_np_data)

    image = cv2.imread(f'dump/{name}_greyscale.png', 0)

    if blur_size and blur_type:
        if blur_type == "Median":
            image = medfilt(image, [5, 31])
        elif blur_type == "Kernels" and kernel_first and kernel_second:
            kernel = np.ones((kernel_first, kernel_second), np.float32) / (kernel_first * kernel_second)
            image = cv2.filter2D(image, -1, kernel)

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    heatmap = (colormap(norm(image)) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'dump/{name}_blur.png', heatmap)
    return(np_data)



# Run 


envs = {'temperature': 16, 'salinity': 9, 'pressure': 10.1325}
cals = {'gain_correction': 28.49, 'equivalent_beam_angle': -21}

raw_path = Path("/Users/jonas/Desktop/testdata_small/")
files = raw_path.glob("*.raw")
for filename in files: 
    data = process_data(filename, cals, envs)
    dd = create_echogram(data, lower = -90, upper = -78, vmin = 0, vmax = 255, cmap = "plasma", blur_type = "Median")

