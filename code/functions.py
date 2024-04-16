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
        waveform_mode="BB",
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



# Find bottom
def maxecho(x, start, end, i):
    x = x[start:end,(i)]
    maxval = max(x)
    echodepth = (np.argmax(x==maxval)+start)
    return echodepth

def maxval(x, start, end, i):
    x = x[start:end,(i)]
    maxval = max(x)
    return maxval

def move_fun(x, window_size): 
    # Padd the list with values 
    padding = window_size // 2 if window_size % 2 == 1 else window_size // 2 - 1
    x = x[:padding] + x + x[-padding:]

    i = 0
    moving_averages = [] 
    
    #moving averages by window size
    while i < len(x) - window_size + 1: 
        
        window = x[i : i + window_size] 
        window_average = np.median(window) 
        moving_averages.append(window_average) 
        i += 1
    
    return(moving_averages)

def find_dead_zone(echodata, depth):
    echodata = echodata.T
    echodata_flipped = np.fliplr(echodata)  
    depth = [int(1000-x) for x in depth]
    depth = [x+10 for x in depth]
    dead_zone = []
    
    for i, ping in enumerate(echodata_flipped):

        #Remove bottom and leaving dead zone
        depth_i = depth[i]
        ping[:depth[i]] = np.nan

        in_a_row = 0
        found_limit = False

        for i, value in enumerate(ping):
            if value < -75:
                in_a_row += 1
            else:
                in_a_row = 0 

            if in_a_row == 3:
                found_limit = True 
                if i-depth_i < 25:
                    dead_zone.append(i-in_a_row)
                else:
                    dead_zone.append(25)
                break

        if not found_limit:
            dead_zone.append(20)

    dead_zone = move_fun(dead_zone, 15)
    dead_zone = [int(1000-x)-5 for x in dead_zone]

    return dead_zone

def find_bottom(echodata, window_size, dead_zone, bottom_roughness_thresh, bottom_hardness_thresh):

    bottom_remove = True
    
    depth = []
    for i in range(0, echodata.shape[1]):
        temp = maxecho(echodata, dead_zone, echodata.shape[0] - dead_zone, i)
        depth.append(temp)
        
    # Smoothed and average bottom depth
    depth_smooth = move_fun(depth, window_size)
    depth_roughness = np.round(np.median(abs(np.diff(depth))), 2)

    # Bottom hardness 
    hardness = []
    for i in range(0, echodata.shape[1]):
        temp = maxval(echodata, dead_zone, echodata.shape[0] - dead_zone, i)
        hardness.append(temp)

    # Smoothed and average bottom hardness        
    hardness_smooth = move_fun(hardness, window_size)
    hardness_mean = np.round(np.nanmean(hardness), 2)

    if depth_roughness > bottom_roughness_thresh or hardness_mean < bottom_hardness_thresh:
        bottom_remove = False
        for item in range(len(depth_smooth)):
            if hardness_smooth[item] < -25 :
                depth_smooth[item] = 1000

    if bottom_remove: 

        dead_zone = find_dead_zone(echodata, depth_smooth)
        
        # Remove points under sea floor
        int_list = [int(item) for item in dead_zone]
        for i in range(0, len(dead_zone)):
            echodata[int_list[i]:,(i)] = 0

    return depth_smooth, hardness, depth_roughness, echodata

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

def find_waves(echodata, wave_thresh, in_a_row_waves, beam_dead_zone, depth):

    echodata[np.isnan(echodata)] = 0

    line = []

    for i, ping in enumerate(echodata.T):

        in_a_row = 0
        found_limit = False

        ping_depth = int(depth[i])
        ping = ping[:ping_depth]

        for i, value in enumerate(ping):
            if value < wave_thresh:
                in_a_row += 1
            else:
                in_a_row = 0 
            if in_a_row == in_a_row_waves:
                found_limit = True 
                line.append(i-in_a_row)
                break
        if not found_limit:
            line.append(beam_dead_zone)


    for ping in range(echodata.shape[1]):
        echodata[:line[ping], ping] = 0

    wave_avg = sum(line) / len(line)
    wave_smoothness = find_wave_smoothness(line)
    
    return echodata, line, wave_avg, wave_smoothness

# Find fish volume - NEW JONAS VERSION
def find_fish_median(echodata, waves, ground, dead_zone):
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

        ping[(ground_limit-dead_zone):] = np.nan # Also masking dead zone
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

    for ping in nasc:
        ping[0:(start*10)] = 0 
        ping[(stop*10):1000] = 0
        cumsum = np.cumsum(ping)
        totnasc = sum(ping)
        medval = totnasc/2
        fishdepth = np.argmax(cumsum>medval)/10
        nascx.append(totnasc)
        fish_depth.append(fishdepth)
 
    return nascx, fish_depth

def save_data(data, filename, save_path, txt_path):

    df = pd.DataFrame(data)
    df.to_csv(f'{save_path}/{filename}')

    with open(txt_path, 'a') as txt_doc:
        txt_doc.write(f'{filename}\n')

