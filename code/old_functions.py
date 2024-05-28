import xarray as xr
import numpy as np
import echopype as ep
import matplotlib.pyplot as plt
import cv2
import pandas as pd

from pathlib import Path



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


def process_data(path, env_params, cal_params, bin_size):
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
        ping_time_bin='2S' # bin size to average along ping_time in seconds
    )
    

    ds_MVBS = ds_MVBS.pipe(swap_chan)
    ping_times = ds_MVBS.Sv.ping_time.values

    return ds_MVBS, ping_times


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
def data_to_images(data, filepath=''):
    """
    Function to create images (greyscale & viridis) from np_array with high resolution.
    """

    np_data = np.nan_to_num(data, copy=True)
    np_data = np_data[:, 4:-4]
    #First method normalization of the data
    np_data[np_data<-90]= -90
    np_data = (np_data - np_data.min())/(np_data.max() - np_data.min())
    # np_data = np.rot90(np_data, k=3) # rotate the np array to rotate the image
    np_data = np_data*256
    
    flip_np_data = cv2.flip(np_data, 1) # flip the image to display it correctly

    cv2.imwrite(f'{filepath}_greyscale.png', flip_np_data) # greyscale image

    image = cv2.imread(f'{filepath}_greyscale.png', 0)
    colormap = plt.get_cmap('viridis') # 
    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'{filepath}.png', heatmap) 


# Find bottom
def maxecho(x, start, end, i):
    x = x[start:end,(i)]
    maxval = max(x)
    echodepth = (np.argmax(x==maxval)+start)/10
    return echodepth

def maxval(x, start, end, i):
    x = x[start:end,(i)]
    maxval = max(x)
    return maxval

def move_fun(x, window_size): 
    
    i = 0
    # Initialize an empty list to store moving averages 
    moving_averages = [] 
    
    # Loop through the array to consider 
    # every window of size 3 
    while i < len(x) - window_size + 1: 
        
        # Store elements from i to i+window_size 
        # in list to get the current window 
        window = x[i : i + window_size] 
    
        # Calculate the average of current window 
        window_average = np.median(window) 
        
        # Store the average of current 
        # window in moving average list 
        moving_averages.append(window_average) 
        
        # Shift window to right by one position 
        i += 1
    
    return(moving_averages)

def find_bottom(echodata, window_size, surf_offset, bottom_offset, bottom_roughness_thresh, bottom_hardness_thresh):
    
    depth = []
    for i in range(0, echodata.shape[1]):
        temp = maxecho(echodata, surf_offset, 979, i)
        depth.append(temp)

    depth = [x*10 for x in depth]
    
    # Smoothed and average bottom depth
    depth_smooth = move_fun(depth, window_size)
    depth_roughness = np.round(np.mean(abs(np.diff(depth))), 2)

    # Bottom hardness 
    hardness = []
    for i in range(0, echodata.shape[1]):
        temp = maxval(echodata, surf_offset, 979, i)
        hardness.append(temp)

    # Smoothed and average bottom hardness        
    hardness_smooth = move_fun(hardness, window_size)
    hardness_mean = np.round(np.nanmean(hardness), 2)

    # If bottom is weak, change to 97 m 
    if depth_roughness > bottom_roughness_thresh or hardness_mean < bottom_hardness_thresh:
        for item in range(len(depth_smooth)):
            #print(len(hardness_smooth))
            #print(depth_smooth[item])
            if hardness_smooth[item] < -25 :
                depth_smooth[item] = 970 


    # Remove Bottom echoes above strongest echo
    # 20 pixels above bottom removed because of echo-effect near bottom 
    depth_smooth = [x-bottom_offset for x in depth_smooth]
    
    # Remove 4 first and four last pings (columns) 
    echodata[:, 0:4] = 0 
    echodata[:, -4:] = 0  

    bottom_remove = True

    if bottom_remove: 

        # Remove points under sea floor
        int_list = [int(item) for item in depth_smooth]
        for i in range(0, len(depth_smooth)):
            echodata[int_list[i]:, 4+(i)] = 0

    return depth_smooth, hardness, depth_roughness, echodata



# Find and detect waves
def find_layer(echodata, beam_dead_zone, in_a_row_thresh, layer_quantile, layer_strength_thresh, layer_size_thresh):

    echodata = echodata[beam_dead_zone:]
    in_a_row = 0

    for n, row in enumerate(echodata):
        row = row[4:-4]
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

def find_waves(echodata, wave_thresh, in_a_row_waves, beam_dead_zone):
    echodata = np.transpose(echodata)

    echodata_crop = echodata[:, beam_dead_zone:]

    datas = []
    line = []

    for i, ping in enumerate(echodata_crop):
        in_a_row = 0
        found_limit = False
        data = []

        for n, val in enumerate(ping):
            if val < wave_thresh: # Denna ska vi testa andra värden på
                in_a_row += 1
            else:
                in_a_row = 0

            if in_a_row == in_a_row_waves: # även här kan vi testa andra värden
                found_limit = True
                line.append(n-in_a_row_waves)
                for x in range(n-in_a_row_waves):
                    data.append(100)
                break

        if not found_limit:
            line.append(len(ping)-2)
            for x in range(len(ping)-2):
                data.append(100)

        datas.append(data)

    line = [i+beam_dead_zone for i in line]


    for i in range(echodata.shape[0]):
        echodata[i, :line[i]] = 0

    echodata = np.transpose(echodata)
    echodata = echodata[4:-4]
    
    line = line[4:-4]
    wave_avg = sum(line) / len(line)
    wave_smoothness = find_wave_smoothness(line)

    return echodata, line, wave_avg, wave_smoothness




# Find fish volume - NEW JONAS VERSION
def find_fish_median(data, waves, ground):
    """
    Function to calc the cumulative sum of fish for each ping.

    Args: 
        data (numpy.ndarray): The sonar data (dB)
        waves: list of indices where the waves are ending in each ping
        ground: list of indices where ground starts for each ping

    Returns: 
        sum (numpy.ndarray): a sum for each ping (NASC).
    """
    np_data = data[4:-4]
    for i, ping in enumerate(np_data):
        wave_limit = waves[i]
        ground_limit = ground[i]

        ping[(ground_limit-16):] = np.nan # lab with - n here
        ping[:(wave_limit+5)] = np.nan # lab with different + n values here

    # calc NASC (Nautical Area Scattering Coefficient - m2nmi-2)1
    nasc = 4 * np.pi * (1852**2) * (10**(np_data/10)) * 0.1

    # nan to zero
    where_are_nans = np.isnan(nasc)
    nasc[where_are_nans] = 0
    
    # find the total (last value) of the cumulative sum and calc index of mean val
    return nasc


def medianfun(x, start, stop):
    """
    Function to calculate the median of cumulative sums for each list in the input list.
    It uses nasc outputted from the find_fish_median2 function
    """
    nascx, fish_depth = [], []
    temp = x.copy()

    for col in temp:
        col[0:(start*10)] = 0 
        col[(stop*10):1000] = 0
        cumsum = np.cumsum(col)
        totnasc = sum(col)
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