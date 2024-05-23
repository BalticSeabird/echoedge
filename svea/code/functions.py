import xarray as xr
import pandas as pd
import numpy as np
import echopype as ep
import matplotlib.pyplot as plt
import cv2
import os 
import warnings


from scipy.ndimage import median_filter
from scipy.ndimage import maximum_filter
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, maximum_filter
from datetime import datetime
from pathlib import Path



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


def extract_meta_data(path):

    if '.raw' in path:
        raw_echodata = ep.open_raw(path, sonar_model="EK80", use_swap=True)

    else:
        raw_data_path = Path.cwd().joinpath(path)
        for fp in raw_data_path.iterdir():
            if fp.suffix == ".raw":
                raw_echodata = ep.open_raw(fp, sonar_model='EK80')
                break
            else:
                print("File not working, please provide a .raw file.")
    
    channels = raw_echodata.platform.channel.to_numpy()
    longitude = raw_echodata.platform.longitude.to_numpy()
    latitude = raw_echodata.platform.latitude.to_numpy()
    transmit_type = raw_echodata.beam.transmit_type.to_numpy()

    return raw_echodata, channels, longitude, latitude, transmit_type



def process_data(path, env_params, cal_params, bin_size, waveform, ping_time_bin='2S'):
    """
    Env_params : dictionary with water temperature in degree C, salinity, pressure in dBar
    Cal_params : dictionary with gain correction (middle value with 0.6.4 version), equivalent beam angle
    Function to load raw data to ep format, calibrate ep data, 
    compute MVBS (mean volume backscattering strength) and
    run swap_chan. Returns NetCDF object (xarray.core.dataset.Dataset). 
    """

    raw_echodata = ep.open_raw(path, sonar_model="EK80", use_swap=True)

    ds_Sv_raw = ep.calibrate.compute_Sv(
        raw_echodata,
        env_params = env_params,
        cal_params = cal_params,
        waveform_mode=waveform,
        encode_mode="complex",
    )

    ds_MVBS = ep.commongrid.compute_MVBS(
        ds_Sv_raw, # calibrated Sv dataset
        range_meter_bin=bin_size, # bin size to average along range in meters
        ping_time_bin=ping_time_bin # bin size to average along ping_time in seconds
    )

    ds_MVBS = ds_MVBS.pipe(swap_chan)
    ping_times = ds_MVBS.Sv.ping_time.values
    return ds_MVBS, ping_times



# Plot and Visualize data
def data_to_images(data, filepath=''):
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

    cv2.imwrite(f'{filepath}.png', heatmap) 



def get_beam_dead_zone(echodata): 

    echodata[np.isnan(echodata)] = 0
    row_sums = np.mean(echodata, axis=1).tolist()
    for i, row in enumerate(row_sums):
        if row == row:
            if row < (-50):
                return i


def detect_outliers(data, threshold=3):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    return np.abs(z_scores) > threshold

# def find_half_bottom(echodata):
#     echodata_original = echodata.copy()
#     echodata = echodata.T
#     echodata_flipped = np.fliplr(echodata) 
#     dead_zone = []
    
#     for i, ping in enumerate(echodata_flipped):
#         in_a_row = 0
#         found_limit = False

#         for i, value in enumerate(ping):
#             if value < -70:
#                 in_a_row += 1
#             else:
#                 in_a_row = 0 

#             if in_a_row == 3:
#                 found_limit = True 
#                 dead_zone.append(i-in_a_row)
#                 break


#         if not found_limit:
#             dead_zone.append(0)

#     dead_zone = move_fun(dead_zone, 15)
#     dead_zone = [int(echodata.shape[1]-x) for x in dead_zone]

#     for i in range(0, len(dead_zone )):
#         echodata_original[(dead_zone[i]):,(i)] = 0 

#     return echodata_original


def find_bottom_for_svea(echodata, wave_line):
    
    echodata_original = echodata.copy()
    echodata = echodata[max(wave_line):, :]
    echodata =  median_filter(echodata, size=(30,5))

    echodata = np.where(echodata < -50, -90, 0)

    # Hittar alla max indices
    max_indices = np.argmax(echodata, axis=0)
    max_indices[max_indices == 0] = echodata_original.shape[0]

    #Interpolerar alla outliers 
    outlier_mask = detect_outliers(max_indices)
    data_with_nans = [x if not outlier else np.nan for x, outlier in zip(max_indices, outlier_mask)]
    non_outlier_indices = np.where(~outlier_mask)[0]
    interp_func_outliers = interp1d(non_outlier_indices, np.array(data_with_nans)[non_outlier_indices], kind='linear', fill_value='extrapolate')
    interpolated_outliers = [interp_func_outliers(i) if np.isnan(x) else x for i, x in enumerate(data_with_nans)]


    non_nan_indices = np.where(max_indices != echodata_original.shape[0])[0]
    non_nan_values  = max_indices[non_nan_indices]

    #Om alla pings är tomma efter threhoslding returnerar man nan
    if non_nan_indices.size == 0:
        max_indices = [350]*echodata_original.shape[1]
    else:
        #Interpolerar alla tomma pings som ett resultat efter thresholding
        interp_func_zeros = interp1d(non_nan_indices, non_nan_values, kind='linear', fill_value='extrapolate')
        max_indices = [interp_func_zeros(i) if x == echodata_original.shape[0] else x for i, x in enumerate(interpolated_outliers)]

        #Tar bort botten 
        max_indices = [int(i) for i in max_indices]
        for i in range(0, len(max_indices )):
            echodata_original[(max_indices[i]+max(wave_line)):,(i)] = 0 

    return echodata_original, max_indices 


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
        try: 
            layer = n + beam_dead_zone
            return layer
        
        except:
            return False
    else:
        return False
        

def find_wave_smoothness(waves_list):
    wave_difs = [abs(j-i) for i, j in zip(waves_list[:-1], waves_list[1:])]
    wave_smoothness = sum(wave_difs) / len(waves_list)
    return wave_smoothness

def find_waves(echodata, wave_thresh, in_a_row_waves, beam_dead_zone):

    echodata[np.isnan(echodata)] = 0

    line = []

    for i, ping in enumerate(echodata.T):

        in_a_row = 0
        found_limit = False

        # if depth[i] == depth[i]:
        #     ping_depth = int(depth[i])
        #     ping = ping[:ping_depth]

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
        echodata[:(line[ping]), ping] = 0

    wave_avg = sum(line) / len(line)
    wave_smoothness = find_wave_smoothness(line)
    
    return echodata, line, wave_avg, wave_smoothness


def save_data(data, filename, save_path, txt_path=False):

    df = pd.DataFrame(data)
    df.to_csv(f'{save_path}/{filename}')

    if txt_path:
        with open(txt_path, 'a') as txt_doc:
            txt_doc.write(f'{filename}\n')


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

        #checking if the value is nan
        if ground[i] == ground[i]:
            ground_limit = ground[i]
            ping[(ground_limit-5):] = np.nan # Also masking dead zone
        ping[:(wave_limit+3)] = np.nan # lab with different + n values here

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







def find_wave_smoothness(waves_list):
    wave_difs = [abs(j-i) for i, j in zip(waves_list[:-1], waves_list[1:])]
    wave_smoothness = sum(wave_difs) / len(waves_list)
    return wave_smoothness

def find_waves(echodata, wave_thresh, in_a_row_waves, beam_dead_zone):

    echodata[np.isnan(echodata)] = 0

    line = []

    for i, ping in enumerate(echodata.T):

        in_a_row = 0
        found_limit = False

        # if depth[i] == depth[i]:
        #     ping_depth = int(depth[i])
        #     ping = ping[:ping_depth]

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
        echodata[:(line[ping]), ping] = 0

    wave_avg = sum(line) / len(line)
    wave_smoothness = find_wave_smoothness(line)
    
    return echodata, line, wave_avg, wave_smoothness


def save_data(data, filename, save_path, txt_path=False):

    df = pd.DataFrame(data)
    df.to_csv(f'{save_path}/{filename}')

    if txt_path:
        with open(txt_path, 'a') as txt_doc:
            txt_doc.write(f'{filename}\n')

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


def shorten_list(original_list, target_length):
    if target_length >= len(original_list):
        return original_list

    step_size = (len(original_list) - 1) / (target_length - 1)
    shortened_list = [original_list[int(round(i * step_size))] for i in range(target_length)]
    return shortened_list
