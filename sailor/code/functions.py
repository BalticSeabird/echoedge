import xarray as xr
import pandas as pd
import numpy as np
import echopype as ep
import matplotlib.pyplot as plt
import cv2
import os 
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Process data 
def swap_chan(ds: xr.Dataset) -> xr.Dataset:
    """Function to replace the channel dimension and coordinate with the
    frequency_nominal variable containing actual frequency values.
    Note that this step is possible only because there are
    no duplicated frequencies present.
    """
    return (
        ds.set_coords("frequency_nominal")
        .swap_dims({"channel": "frequency_nominal"})
        .reset_coords("channel")
    )


def process_data(path, env_params, cal_params, bin_size, ping_time_bin='2S', original_resolution=False):
    """
    Env_params : dictionary with water temperature in degree C, salinity, pressure in dBar
    Cal_params : dictionary with gain correction (middle value with 0.6.4 version), equivalent beam angle
    Function to load raw data to ep format, calibrate ep data, 
    compute MVBS (mean volume backscattering strength) and
    run swap_chan. Returns NetCDF object (xarray.core.dataset.Dataset). 
    """

    if '.raw' in path:
        raw_echodata = ep.open_raw(path, sonar_model="EK80")

    else:
        raw_data_path = Path.cwd().joinpath(path)
        for fp in raw_data_path.iterdir():
            if fp.suffix == ".raw":
                raw_echodata = ep.open_raw(fp, sonar_model='EK80')
                break
            else:
                print("File not working, please provide a .raw file.")
    
    ds_Sv_raw = ep.calibrate.compute_Sv(
        raw_echodata,
        env_params = env_params,
        cal_params = cal_params,
        waveform_mode="CW",
        encode_mode="complex",
    )

    ds_MVBS = ep.commongrid.compute_MVBS(
        ds_Sv_raw, # calibrated Sv dataset
        range_meter_bin=bin_size, # bin size to average along range in meters
        ping_time_bin=ping_time_bin # bin size to average along ping_time in seconds
    )

    if original_resolution == False: 
        ds_MVBS = ds_MVBS.pipe(swap_chan)
        ping_times = ds_MVBS.Sv.ping_time.values
        return ds_MVBS, ping_times
    else:
        ping_times = ds_Sv_raw.Sv.ping_time.values
        return ds_Sv_raw, ping_times


def remove_vertical_lines(echodata):
    # Hitta indexen för arrayerna som bara innehåller NaN
    nan_indices = np.isnan(echodata).all(axis=1)
    indices_to_remove = np.where(nan_indices)[0]

    # Ta bort arrayerna med NaN-värden från din ursprungliga array
    echodata = echodata[~nan_indices]

    return echodata, indices_to_remove


def clean_times(ping_times, nan_indicies):
    mask = np.ones(ping_times.shape, dtype=bool)
    mask[nan_indicies] = False

    # Använd masken för att ta bort värden från ping_times
    ping_times = ping_times[mask]

    return ping_times


# Plot and Visualize data
def data_to_images(data, filepath='', make_wider=False):
    """
    Function to create images (greyscale & viridis) from np_array with high resolution.
    """

    np_data = np.nan_to_num(data, copy=True)
    np_data[np_data<-90]= -90
    np_data = (np_data - np_data.min())/(np_data.max() - np_data.min())
    np_data = np_data*256
    
    # flip_np_data = cv2.flip(np_data, 1) # flip the image to display it correctly

    cv2.imwrite(f'{filepath}_greyscale.png', np_data) # greyscale image

    image = cv2.imread(f'{filepath}_greyscale.png', 0)
    colormap = plt.get_cmap('viridis') # 
    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    if make_wider == True:
        height, width = np_data.shape
        width = int(width * 5)
        height = int(height / 3)
        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(f'{filepath}.png', heatmap) 



def moving_average(data, window_size):
    series = pd.Series(data)
    moving_averages = series.rolling(window=window_size, center=True, min_periods=1).mean()
    return  moving_averages.tolist()

def interpolate_nan(lst, depth_if_all_nan):

    arr = np.array(lst)
    nan_indices = np.isnan(arr)
    non_nan_indices = np.arange(len(arr))[~nan_indices]

    if len(non_nan_indices) == 0:
        return [depth_if_all_nan] * len(arr)

    # Interpolate NaN values using linear interpolation
    arr[nan_indices] = np.interp(np.arange(len(arr))[nan_indices], non_nan_indices, arr[non_nan_indices])

    return arr.tolist()

def replace_outliers_with_nan(data):

    def detect_outliers(data, threshold=3):
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = [(x - mean) / std_dev for x in data]
        return np.abs(z_scores) > threshold

    outliers_mask = detect_outliers(data)
    data = np.asarray(data, dtype=float)
    data[outliers_mask] = np.nan

    return data


def get_beam_dead_zone(echodata):
    echodata[np.isnan(echodata)] = 0
    row_sums = np.mean(echodata, axis=1).tolist()
    for i, row in enumerate(row_sums):
        if row == row:
            if row < (-50):
                return i

def find_bottom(echodata, window_size):
    echodata_original = echodata.copy()
    #Get dead zone and slice it out from echodata
    dead_zone = get_beam_dead_zone(echodata) 
    echodata = echodata[dead_zone:, :] 

    #Finds the maxecho and depth
    depth = np.argmax(echodata, axis=0) 
    hardness = echodata[depth, np.arange(echodata.shape[1])]

    #Finding weak pings
    weak_ping_mask = np.isnan(np.where(hardness < -30, np.nan, depth))
    #Findind outliers and set them to nan
    depth = replace_outliers_with_nan(np.where(hardness < -30, echodata.shape[0], depth) )
    #Setting weak pings as nan as well
    depth[weak_ping_mask] = np.nan

    #calculating roughness on the values that aren't nan, if there aren't a bottom, depth roughness will be 0
    non_nan_depth = depth[~np.isnan(depth)]
    if len(non_nan_depth) == 0:
        depth_roughness = 0
    else:
        depth_roughness = np.round(np.median(np.abs(np.diff(non_nan_depth))), 2)

    #interpolating nan values and smoothing
    depth = interpolate_nan(depth, echodata.shape[0])
    depth = moving_average(depth , window_size)

    #Taking upper deadzone that was sliced to account and adding bottom deadzone to depth
    depth = [int(item + dead_zone) for item in depth]
    depth = [item - 30 if item != echodata_original.shape[0] else item for item in depth]

    #Removing the bottom
    for i in range(0, len(depth)):
        echodata_original[depth[i]:,(i)] = 0

    return depth, hardness, depth_roughness, echodata_original

# Find and detect waves
def find_layer(echodata, beam_dead_zone, in_a_row_thresh, layer_quantile, layer_strength_thresh, layer_size_thresh):

    echodata[np.isnan(echodata)] = 0
    echodata = echodata[beam_dead_zone:]
    in_a_row = 0

    for n, row in enumerate(echodata):
        row = row[~np.isnan(row)]
        avg_val = np.quantile(row, layer_quantile)

        if avg_val < layer_strength_thresh:
            in_a_row += 1

        if in_a_row == in_a_row_thresh:
            break

    if n > layer_size_thresh:
        layer = n + beam_dead_zone
        return layer
    else:
        return False
        

def find_wave_smoothness(waves_list):
    wave_difs = [abs(j-i) for i, j in zip(waves_list[:-1], waves_list[1:])]
    wave_smoothness = sum(wave_difs) / len(waves_list)
    return wave_smoothness

def find_waves(echodata, wave_thresh, in_a_row_waves, depth):

    echodata[np.isnan(echodata)] = 0

    dead_zone = get_beam_dead_zone(echodata)

    line = []

    for i, ping in enumerate(echodata.T):

        in_a_row = 0
        found_limit = False

        #Removing upper deadzone and bottom from echodata
        ping = ping[dead_zone:int(depth[i])]

        for i, value in enumerate(ping):
            if value < wave_thresh:
                in_a_row += 1
            else:
                in_a_row = 0 
            if in_a_row == in_a_row_waves:
                found_limit = True 
                line.append(i-in_a_row+dead_zone)
                break
        if not found_limit:
            line.append(dead_zone)

    line = [wave + 1 for wave in line]
    for ping in range(echodata.shape[1]):
        echodata[:line[ping], ping] = 0

    wave_avg = sum(line) / len(line)
    wave_smoothness = find_wave_smoothness(line)
    
    return echodata, line, wave_avg, wave_smoothness

# Find fish volume - NEW JONAS VERSION
def find_fish_median(echodata, waves, ground):
    """
    Function to calc the cumulative sum of fish for each ping.

    Args: 
        data (numpy.ndarray): The sonar data (dB)
        waves: list of indices where the waves are ending in each ping
        ground: list of indices where ground starts for each ping

    Returns: 
        sum (numpy.ndarray): a sum for each ping (NASC).
    """
 
    for i, ping in enumerate(echodata):
        wave_limit = waves[i]
        ground_limit = ground[i]

        ping[(ground_limit):] = np.nan # Also masking dead zone
        ping[:(wave_limit)] = np.nan # lab with different + n values here

    # calc NASC (Nautical Area Scattering Coefficient - m2nmi-2)1
    nasc = 4 * np.pi * (1852**2) * (10**(echodata/10)) * 0.1

    # nan to zero
    where_are_nans = np.isnan(nasc)
    nasc[where_are_nans] = 0
    
    return nasc


def medianfun(nasc, start, stop):
    """
    Function to calculate the median of cumulative sums for each list in the input list.
    It uses nasc outputted from the find_fish_median2 function
    """
    nascx, fish_depth = [], []

    nasc_copy = nasc.copy()

    for ping in nasc_copy:
        ping[0:(start*10)] = 0 
        ping[(stop*10):1000] = 0
        cumsum = np.cumsum(ping)
        totnasc = sum(ping)
        medval = totnasc/2
        fishdepth = np.argmax(cumsum>medval)/10
        nascx.append(totnasc)
        fish_depth.append(fishdepth)
 
    return nascx, fish_depth

def save_data(data, filename, save_path, txt_path=False):

    df = pd.DataFrame(data)
    df.to_csv(f'{save_path}/{filename}')

    if txt_path:
        with open(txt_path, 'a') as txt_doc:
            txt_doc.write(f'{filename}\n')

