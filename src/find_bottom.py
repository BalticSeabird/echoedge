import numpy as np

from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


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

    # If all pings are empty, we are returning 350 as a representation of NaN (no bottom found)
    if non_nan_indices.size == 0:
        max_indices = [350]*echodata_original.shape[1]
    else:
        # Interpolate all empty pings after interpolation
        interp_func_zeros = interp1d(non_nan_indices, non_nan_values, kind='linear', fill_value='extrapolate')
        max_indices = [interp_func_zeros(i) if x == echodata_original.shape[0] else x for i, x in enumerate(interpolated_outliers)]

        # Remove bottom
        max_indices = [int(i) for i in max_indices]
        for i in range(0, len(max_indices )):
            echodata_original[(max_indices[i]+max(wave_line)):,(i)] = 0 

    return echodata_original, max_indices 



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
    else:
        dead_zone = []
        for i in range(len(depth)):
            dead_zone.append(1000)


    return depth_smooth, hardness, depth_roughness, echodata, dead_zone
