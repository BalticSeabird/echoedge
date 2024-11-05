import xarray as xr
import pandas as pd
import numpy as np
import echopype as ep
import matplotlib.pyplot as plt
import cv2
import os 
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageFilter
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
from skimage.transform import rescale
import warnings

warnings.filterwarnings("ignore")

def get_gps(path, survey = 'Karlso-2023'):
    """
    Get original coordinates without interpolation 

    Args: 
        path (str): The path to gps.csv file  
        survey (str) : The survey to filter year

    Returns: 
        pd.DataFrame: A DataFrame with coordinates and time
    """
    df_gps = pd.read_csv(path)
    df_gps = df_gps[df_gps['Survey'] == survey]
    df_gps['datetime'] = pd.to_datetime(df_gps['GPS_date'] + ' ' + df_gps['GPS_time'])
    df_gps = df_gps.drop(['GPS_date', 'GPS_time'], axis=1)
    return df_gps

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


def process_data(path, env_params, cal_params, range_meter_bin=0.1, ping_time_bin='2S', original_resolution=False):
    """
    Env_params : dictionary with water temperature in degree C, salinity, pressure in dBar
    Cal_params : dictionary with gain correction (middle value with 0.6.4 version), equivalent beam angle
    Function to load raw data to ep format, calibrate ep data, 
    compute MVBS (mean volume backscattering strength) and
    run swap_chan. Returns NetCDF object (xarray.core.dataset.Dataset). 
    """

    if '.raw' or '.RAW' in path:
        raw_echodata = ep.open_raw(path, sonar_model="EK80")

    else:
        raw_data_path = Path.cwd().joinpath(path)
        for fp in raw_data_path.iterdir():
            if fp.suffix == ".raw" or fp.suffix == ".RAW":
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
        range_meter_bin=range_meter_bin, # bin size to average along range in meters
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
    # if not filepath.endswith('complete') and not os.path.exists(f'{filepath}.npy'):
    #     np.save(f'{filepath}', data)
    # else:
    #     print("exist !")
    if filepath.endswith('complete') and not os.path.exists(f'{filepath}.npy'):
        np.save(f'{filepath}', data)
      
    #print(data.shape)
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

def remove_greyscale_files(directory):

    files = os.listdir(directory)
    
    for file in files:

        if file.endswith('_greyscale.png'):
            file_path = os.path.join(directory, file)
            os.remove(file_path)


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



def find_bottom(echodata, surf_offset=20, window_size=9):

    bottom_remove = True

    depth = []
    for i in range(0, echodata.shape[1]):
        temp = maxecho(echodata, surf_offset, echodata.shape[0] - surf_offset, i)
        depth.append(temp)
        
    # Smoothed and average bottom depth
        
    depth_smooth = move_fun(depth, window_size)
    depth_roughness = np.round(np.mean(abs(np.diff(depth))), 2)

    # Bottom hardness 
    hardness = []
    for i in range(0, echodata.shape[1]):
        temp = maxval(echodata, surf_offset, echodata.shape[0] - surf_offset, i)
        hardness.append(temp)

    # Smoothed and average bottom hardness        
    hardness_smooth = move_fun(hardness, window_size)
    hardness_mean = np.round(np.nanmean(hardness), 2)
    # If bottom is weak, change to 97 m 
    if hardness_mean < -20:
        bottom_remove = False
    if depth_roughness > 30 or hardness_mean < -20:
        for item in range(len(depth_smooth)):
            if hardness_smooth[item] < -25 :
                depth_smooth[item] = 970 


    depth_smooth = [x-surf_offset for x in depth_smooth]

    if bottom_remove: 

        # Remove points under sea floor
        int_list = [int(item) for item in depth_smooth]
        for i in range(0, len(depth_smooth)):
            echodata[int_list[i]:,(i)] = 0

    return depth_smooth, hardness, depth_roughness, echodata

def find_bottom_orignal_resolution(echodata):
    depth = []
    for i in range(0, echodata.shape[1]):
        temp = maxecho(echodata, 0, echodata.shape[0], i)
        depth.append(temp)

    # Smoothed and average bottom depth
    #Has to be uneven number 
    depth_smooth = move_fun(depth, 51)
    depth_smooth = [x-340 for x in depth_smooth]

    # Remove points under sea floor
    int_list = [int(item) for item in depth_smooth]
    for i in range(0, len(depth_smooth)):
        echodata[int_list[i]:,(i)] = 0

    return depth_smooth, echodata

# Find and detect waves
def find_layer(echodata):

    echodata = echodata[20:]
    in_a_row = 0

    for n, row in enumerate(echodata):
        #row = row[4:-4] #FYRA BORTTAGEN
        row = row[~np.isnan(row)]
        avg_val = np.quantile(row, .7)
        # print(f'{n}: {avg_val}')
        if avg_val < -80:
            in_a_row += 1

        if in_a_row == 3:
            break

    if n > 15:
        layer = n + 20
        return layer
    else:
        return False
        



def find_wave_smoothness(waves_list):
    wave_difs = [abs(j-i) for i, j in zip(waves_list[:-1], waves_list[1:])]
    wave_smoothness = sum(wave_difs) / len(waves_list)
    return wave_smoothness

def find_waves(echodata, limit):
    echodata = np.transpose(echodata)

    echodata_crop = echodata[:, 15:]

    datas = []
    line = []

    for i, ping in enumerate(echodata_crop):
        in_a_row = 0
        found_limit = False
        data = []

        for n, val in enumerate(ping):
            if val < -limit: # Denna ska vi testa andra värden på
                in_a_row += 1
            else:
                in_a_row = 0

            if in_a_row == 3: # även här kan vi testa andra värden
                found_limit = True
                line.append(n-5)
                for x in range(n-5):
                    data.append(100)
                break

        if not found_limit:
            line.append(len(ping)-2)
            for x in range(len(ping)-2):
                data.append(100)

        datas.append(data)

    line = [i+16 for i in line]


    for i in range(echodata.shape[0]):
        echodata[i, :line[i]] = 0

    echodata = np.transpose(echodata)
    #echodata = echodata[4:-4] #FYRA BORTTAGEN
    
    #line = line[4:-4] #FYRA BORTTAGEN
    wave_avg = sum(line) / len(line)
    wave_smoothness = find_wave_smoothness(line)

    return echodata, line, wave_avg, wave_smoothness




# Find fish volume 
def find_fish_median(data, waves, ground):
    """
    Function to calc the cumulative sum of fish for each ping.

    Args: 
        data (numpy.ndarray): The sonar data (dB)
        waves: list of indices where the waves are ending in each ping
        ground: list of indices where ground starts for each ping

    Returns: 
        cumsum (numpy.ndarray): a cumulative sum for each ping.
        mean_indexes: median point of fish for each ping.
    """
    #np_data = data[4:-4] #FYRA BORTTAGEN
    np_data = data
    for i, ping in enumerate(np_data):
        wave_limit = waves[i]
        ground_limit = ground[i]

        ping[(ground_limit-16):] = np.nan # lab with - n here
        ping[:(wave_limit+5)] = np.nan # lab with different + n values here

    # calc NASC (Nautical Area Scattering Coefficient - m2nmi-2)1
    new_np_data = 4 * np.pi * (1852**2) * (10**(np_data/10)) * 0.1

    # nan to zero
    where_are_nans = np.isnan(new_np_data)
    new_np_data[where_are_nans] = 0
    cumsum = np.cumsum(new_np_data, axis=1)

    # find the total (last value) of the cumulative sum and calc index of mean val
    max_vals, mean_indexes = [], []
    for arr in cumsum:
        max_val = arr[-1]
        mean_index = min(range(len(arr)), key=lambda i: abs(arr[i]-(max_val/2)))
        max_vals.append(max_val)
        mean_indexes.append(mean_index)

    return cumsum, mean_indexes


def medianfun(list_of_lists):
    """
    Function to calculate the median of cumulative sums for each list in the input list.
    """
    fish_inds, fish_depths = [], []

    for x in list_of_lists:
        x = np.array(x)
        fishind = x[-1]
        medval = fishind/2
        fishdepth = np.argmax(x>medval)/10
        fish_inds.append(fishind)
        fish_depths.append(fishdepth)

    return fish_inds, fish_depths


# Find fish volume - NEW JONAS VERSION
def find_fish_median2(data, waves, ground):
    """
    Function to calc the cumulative sum of fish for each ping.

    Args: 
        data (numpy.ndarray): The sonar data (dB)
        waves: list of indices where the waves are ending in each ping
        ground: list of indices where ground starts for each ping

    Returns: 
        sum (numpy.ndarray): a sum for each ping (NASC).
    """
    #np_data = data[4:-4] # FYRA BORTTAGEN
    np_data = data
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


def medianfun2(x, start, stop):
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


def sort_files_by_time(filenames):
    """
    Args: 
        file_names: List of file names

    Returns: 
        list: List of file names sorted by time
    """
    def extract_date_and_time(filename):
        datetime_str = filename.split('-')[2] + filename.split('-')[3]
        file_datetime = datetime.strptime(datetime_str, "D%Y%m%dT%H%M%S")
        return file_datetime

    # Sort filenames based on extracted date and time
    sorted_filenames = sorted(filenames, key=extract_date_and_time)
    return sorted_filenames


def get_raw_files(start=None, end=None, path = '/mnt/pr_SLU/pr_SLU/Sailor/raw/2023/'):
    """
    Selects raw files within specific time range and sorts the files by time. If no start and end is provided, all files is provided.
    
    Args: 
        start (tuple) : A tuple representing the start date and time (year, month, day, hour, minute, second)
        end (int) : A tuple representing the end date and time (year, month, day, hour, minute, second).
        path (str): The path to rawfiles

    Returns: 
        selected_files: List of filenames within time range 
        raw_files_path: List of paths to rawfiles within time range
    """

    if start is None or end is None:
        file_names = [file for file in os.listdir(path)]
        file_paths = [path + file for file in file_names if file.endswith('.raw')]
    else:
        raw_files = [file for file in os.listdir(path)]
        selected_files = []

        start = datetime(*start)
        end = datetime(*end)

        for file in raw_files:
            datetime_str = file.split('-')[2] + file.split('-')[3]
            file_datetime = datetime.strptime(datetime_str, "D%Y%m%dT%H%M%S")
    
            if start <= file_datetime <= end:
                selected_files.append(file)

        file_names = sort_files_by_time(selected_files)
        file_paths = [path + file for file in file_names if file.endswith('.raw')]

    return file_names, file_paths



def get_echodata(raw_file_path, env_params, cal_params, range_meter_bin=0.1, ping_time_bin='2s', clean=False):
    """
    Get echodataset and list of ping_times. If remove_bottom_and_waves is set to True it also returns list of wave_line and depth.

    Args: 
        raw_file_path (str): Path to the raw file
        range_meter_bin (int, optional): Bin size along range in meters, default to 0.1
        ping_time_bin (str, optional) = Bin size along ping_time, default to 2S 
        clean(boolean, optional) = Removes bottom and waves, default False

    Returns: 
        numpy.ndarray: 2D array containing echodata 
        list: time for every ping 
        list: wave line index
        list: bottom depth index
    """

    echodata, ping_times = process_data(raw_file_path, env_params, cal_params, range_meter_bin, ping_time_bin)
    echodata = echodata.Sv.to_numpy()[0]
    echodata, nan_indicies = remove_vertical_lines(echodata)
    echodata = np.swapaxes(echodata, 0, 1)

    if clean == True:
        depth, hardness, depth_roughness, echodata = find_bottom(echodata)
        depth = [int(d) for d in depth]

        echodatax = echodata.copy()
        layer = find_layer(echodatax)
        if layer:
            echodata, wave_line, wave_avg, wave_smoothness = find_waves(echodata, 68)
        else:
            echodata, wave_line, wave_avg, wave_smoothness = find_waves(echodata, 77)
            if wave_smoothness == 0:
                wave_dif = 0 
            else:
                wave_dif = wave_avg / wave_smoothness

            if wave_avg > 70 or (wave_avg > 34 and wave_smoothness < 8): 
                echodata, wave_line, wave_avg, wave_smoothness = find_waves(echodatax, 65)
                return echodata, wave_line, depth, ping_times
    
        return echodata, ping_times, wave_line, depth
    else:
        return echodata

def get_echodata_original(raw_file_path, env_params, cal_params, clean=False):
   
    echodata, ping_times = process_data(raw_file_path, env_params, cal_params, original_resolution=True)
    echodata = echodata.Sv.to_numpy()[0]
    echodata, nan_indicies = remove_vertical_lines(echodata)
    echodata = np.swapaxes(echodata, 0, 1)
    echodata = echodata[~np.isnan(echodata).any(axis=1)]

    if clean == True:
        depth, hardness, depth_roughness, echodata,  = find_bottom(echodata, surf_offset=280, window_size=20)
        echodatax = echodata.copy()
        layer = find_layer(echodatax)
        if layer:
            echodata, wave_line, wave_avg, wave_smoothness = find_waves(echodata, 68)
        else:
            echodata, wave_line, wave_avg, wave_smoothness = find_waves(echodata, 77)
            if wave_smoothness == 0:
                wave_dif = 0 
            else:
                wave_dif = wave_avg / wave_smoothness

            if wave_avg > 70 or (wave_avg > 34 and wave_smoothness < 8): 
                new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(echodatax, 65)
        
        return echodata, ping_times, wave_line, depth
    else:
        return echodata

def get_interpolated_gps(path, frequency=2):
    """
    Interpolates coordinates

    Args: 
        path (str): The path to gps.csv file  
        frequency (int) : The frequency of interpolation in seconds

    Returns: 
        pd.DataFrame: A DataFrame with interpolated coordinates and time
    """
    
    df = pd.read_csv(path)
    
    print('Loading and interpolating coordinates...')
    df['Time'] = pd.to_datetime(df['GPS_date'] + ' ' + df['GPS_time'])

    # drop rows with repeated time
    df = df.drop(columns=['GPS_date', 'GPS_time'])
    df = df[~df.Time.duplicated()]

    # new range (2 seconds), resample and interpolate
    new_range = pd.date_range(df.Time[0], df.Time.values[-1], freq=str(frequency)+'S')
    print("new range:",df.Time[0])
    print("new range:",df.Time.values[-1])
    interpolated_df = df.set_index('Time').reindex(new_range).interpolate().reset_index()

    interpolated_df.rename(columns= {'index' : 'time'}, inplace=True)
    interpolated_df['Longitude'] = interpolated_df['Longitude'].apply(lambda x: round(x, 5))
    interpolated_df['Latitude'] = interpolated_df['Latitude'].apply(lambda x: round(x, 5))

    # Get the velocity values corresponding to the continuous time interval ]T1,T2]
    present_df = interpolated_df[interpolated_df['time'].isin(df['Time'])]
    result_df = pd.merge_asof(interpolated_df, present_df, on='time', direction='forward')
    result_df = result_df[['time', 'Longitude_x','Latitude_x', 'Velocity_y']]
    result_df.rename(columns= {'Velocity_y' : 'Velocity','Longitude_x' : 'Longitude','Latitude_x' : 'Latitude'}, inplace=True)
    print('Coords loaded successfully!')

    return result_df

def get_gps(path, survey = 'Karlso-2023'):
    """
    Get original coordinates without interpolation 

    Args: 
        path (str): The path to gps.csv file  
        survey (str) : The survey to filter year

    Returns: 
        pd.DataFrame: A DataFrame with coordinates and time
    """
    df_gps = pd.read_csv(path)
    df_gps = df_gps[df_gps['Survey'] == survey]
    df_gps['datetime'] = pd.to_datetime(df_gps['GPS_date'] + ' ' + df_gps['GPS_time'])
    df_gps = df_gps.drop(['GPS_date', 'GPS_time'], axis=1)
    return df_gps


def download_posi(df1,df2):
    """
    Args : 
        df1 : the output dataframe without the postion infos of longitude and latitude
        df2 : the csv file contains the postion infos of longitude and latitude
    Returns:
        df1 after having position infos added
    """
    df2['time'] = pd.to_datetime(df2['time'])   # importante change dtype

    # Merge DataFrames on the 'time' column
    merged_df = pd.merge(df1, df2[['time', 'Longitude', 'Latitude','Velocity']], on='time', how='left')

    # Update the 'Longitude' and 'Latitude' columns in df1
    merged_df['Latitude'] = np.where(pd.isna(merged_df['Latitude_x']), merged_df['Latitude_y'], merged_df['Latitude_x'])
    merged_df['Longitude'] = np.where(pd.isna(merged_df['Longitude_x']), merged_df['Longitude_y'], merged_df['Longitude_x'])
    merged_df['Velocity'] = np.where(pd.isna(merged_df['Velocity_x']), merged_df['Velocity_y'], merged_df['Velocity_x'])
    df1.loc[:, 'Longitude'] = merged_df['Longitude']
    df1.loc[:,'Latitude'] = merged_df['Latitude']
    df1.loc[:,'Velocity'] = merged_df['Velocity']


def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # 6371.0 Radius of the Earth in kilometers (you can adjust this value)
    distance = 6371.0 * c  

    return distance


def cut_zone(time):
    """
    Args time: 
        a str of a date time consisted by 'the time part'  +  '+' + additional part
        
    Returns:
        cutoff the "+11:00" in time formate.

    """
    # split the time part + additional part
    time_str = str(time)
    parts = time_str.split('+')
    [d_time,d_plus] = parts
    d_utc = pd.to_datetime(d_time)
    
    return pd.to_datetime(d_utc)


def inter_positions_new(arr):
    segments = []
    arr_unique = [arr[0]]
    for i in range(1,len(arr)):
        if arr[i] != arr[i-1]:
            segments.append(i)
            arr_unique.append(arr[i])

    segments.insert(0, 0)
    result_dict = {
        "segments": segments,
        "unique_values": arr_unique
    }
    return result_dict


def resize_image(img_path,velocity,meanvelocity):
    """
    img_path : complete path for a image file
    velocity : the number of lines = the pixel count vertically in the image
    """
    # premiere etape : read the image 
    img = cv2.imread(img_path)

    # seconde : get the division positions and factors
    div = inter_positions_new(velocity)
    div_pix = div['segments']
    div_fac = div['unique_values']

    # Initialize an empty image with the same dimensions as the input image
    img_groupe = img[:, :0, :].copy()
    
    for i in range(len(div_pix)):
        # get the division position of pixels
        p_start = div_pix[i]
        p_end = div_pix[i + 1] if i + 1 < len(div_pix) else img.shape[1]
       
        # get the zoom factor fx, (fy=1)
        f_x = (div_fac[i] / meanvelocity)

        # chop the image divided by pixels
        left_img = img[:,p_start:p_end,:]
        interp_method = cv2.INTER_CUBIC if f_x > 1 else cv2.INTER_AREA
        img_chop = cv2.resize(left_img,None,fx = f_x, fy = 1,interpolation = interp_method)

        # Group the image with the images gotten before
        img_groupe = cv2.hconcat([img_groupe, img_chop])
    
    saveimg = img_path.replace('.png', '_new.png')
    cv2.imwrite(saveimg, img_groupe)


def resize_matrix(matrix,velocity,meanvelocity):
    """
    matrice : matrix for a data.npy file
    velocity : the number of lines = the pixel count vertically in the matrice
    """

    # get the division positions and factors
    div = inter_positions_new(velocity)
    div_pix = div['segments']
    div_fac = div['unique_values']

    # Initialize an empty image with the same dimensions as the input image
    matrix_groupe = matrix[:, :0].copy()

    for i in range(len(div_pix)):
        # get the division position of pixels
        p_start = div_pix[i]
        p_end = div_pix[i + 1] if i + 1 < len(div_pix) else matrix.shape[1]
        
        # get the zoom factor fx, (fy=1)
        f_x = (div_fac[i] / meanvelocity)

        # chop the image divided by pixels
        left_matrix = matrix[:,p_start:p_end]

        # if f_x > 1:
        #     matrix_chop =  rescale(left_matrix, (1,f_x), anti_aliasing=False)
        # else:
        #     matrix_chop =  rescale(left_matrix, (1,f_x), anti_aliasing=True, anti_aliasing_sigma=1)
        matrix_chop =  rescale(left_matrix, (1,f_x), anti_aliasing=True)

        # Group the image with the images gotten before
        matrix_groupe = np.concatenate((matrix_groupe, matrix_chop), axis=1)

    return(matrix_groupe)


def resize_matrix_31052024(matrix,velocity,meanvelocity):
    """
    matrice : matrix for a data.npy file
    velocity : the number of lines = the pixel count vertically in the matrice
    """

    # get the division positions and factors
    div = inter_positions_new(velocity)
    div_pix = div['segments']
    div_fac = div['unique_values']

    # Initialize an empty image with the same dimensions as the input image
    matrix_groupe = matrix[:, :0].copy()
    # matrix_groupe = np.zeros((matrix.shape[0], 0), dtype=matrix.dtype)
    mapping = []
    for i in range(len(div_pix)):
        # get the division position of pixels
        p_start = div_pix[i]
        p_end = div_pix[i + 1] if i + 1 < len(div_pix) else matrix.shape[1]
        
        # get the zoom factor fx, (fy=1)
        f_x = (div_fac[i] / meanvelocity)

        # chop the image divided by pixels
        left_matrix = matrix[:,p_start:p_end]
        if f_x > 1:
            matrix_chop =  rescale(left_matrix, (1,f_x), anti_aliasing=True, anti_aliasing_sigma=1)
        else:
            matrix_chop =  rescale(left_matrix, (1,f_x), anti_aliasing=True, anti_aliasing_sigma=1)
      
        mapping.append((p_start, p_end, matrix_groupe.shape[1],matrix_groupe.shape[1]+matrix_chop.shape[1]))
        # Group the image with the images gotten before
        matrix_groupe = np.concatenate((matrix_groupe, matrix_chop), axis=1)
    return matrix_groupe, mapping