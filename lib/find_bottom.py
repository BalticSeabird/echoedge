import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from scipy.ndimage import median_filter


# Svea
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


# Sailor
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


def find_bottom(echodata, window_size, hardness_thresh):
    echodata_original = echodata.copy()
    #Get dead zone and slice it out from echodata
    dead_zone = get_beam_dead_zone(echodata) 
    echodata = echodata[dead_zone:, :] 

    #Finds the maxecho and depth
    depth = np.argmax(echodata, axis=0) 
    hardness = echodata[depth, np.arange(echodata.shape[1])]

    #Finding weak pings
    weak_ping_mask = np.isnan(np.where(hardness < hardness_thresh, np.nan, depth))
    #Findind outliers and set them to nan
    depth = replace_outliers_with_nan(np.where(hardness < hardness_thresh, echodata.shape[0], depth) )
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