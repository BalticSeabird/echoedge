import numpy as np
import echopype as ep
import pandas as pd
import glob
import os
import sys
import warnings
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
import yaml
import traceback
from scipy.ndimage import median_filter
import numpy.ma as ma
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib'))

from yaml.loader import SafeLoader
from find_bottom import find_bottom
from export_data import save_data
from find_fish import find_fish_median, medianfun
from visualization import data_to_images
from processing import process_data, clean_times, remove_vertical_lines
from find_waves import find_waves, find_layer

warnings.filterwarnings("ignore")

def load_params(file_path):
    """
    Loads parameters from a YAML file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        list: A list of YAML documents loaded from the file.
    """
    with open(file_path, 'r') as f:
        return list(yaml.load_all(f, Loader=SafeLoader))

def load_gps_data(gps_path):
    """
    Reads GPS data from a CSV file and returns a DataFrame with a Datetime column.

    Args:
        gps_path (str): File path to the GPS data CSV file.

    Returns:
        pandas.DataFrame: Processed DataFrame containing GPS data with a Datetime column.

    Notes:
        The input CSV file is expected to have 'GPS_date' and 'GPS_time' columns which might needs to be changed. 
    """
    df_gps = pd.read_csv(gps_path)
    df_gps['Datetime'] = pd.to_datetime(df_gps['GPS_date'] + ' ' + df_gps['GPS_time'])
    return df_gps.drop(['GPS_date', 'GPS_time'], axis=1)

def get_files_to_process(path, completed_files_path):
    """
    Retrieves files in a directory that is not already processed.

    Args:
        path (str): Directory path containing files to process.
        completed_files_path (str): File path to a text file listing completed file names.

    Returns:
        list: List of file names that have not been marked as completed.

    """
    files = os.listdir(path)
    with open(completed_files_path, 'r') as completed_txt_file:
        completed_files = [line.strip() for line in completed_txt_file.readlines()]
    return [f for f in files if f not in completed_files]

def process_file(file, path, params, img_path, gps_data, ping_times, results, csv_path, completed_files_path):
    """
    Processed the raw files in echopype and returns a numpy array

    Args:
        file (str): The filename to prcess
        path (str) : The path to the raw files 
        params (dict): A dictionary containing parameters for processing the file.
        img_path (str): Path to save images 
        gps_data (list): GPS data corresponding to each ping
        ping_times (list): Times corresponding to each ping 
        results (dict): A list to store results from processing.
        csv_path (str): Path to store CSV files.
        completed_files_path (str): Path to completed_files.txt


    Returns:
        None

    """
    try:
        with open(completed_files_path, 'a') as txt_doc:
            txt_doc.write(f'{file}\n')

        filepath = os.path.join(path, file)
        new_file_name = os.path.splitext(file)[0]

        echodata, ping_times = process_data(filepath, params['env_params'], params['cal_params'], params['bin_size'], 'BB')
        echodata = echodata.Sv.to_numpy()[0]
        echodata, nan_indicies = remove_vertical_lines(echodata)
        echodata_swap = np.swapaxes(echodata, 0, 1)

        data_to_images(echodata_swap, f'{img_path}/{new_file_name}') # save img without ground
        os.remove(f'{img_path}/{new_file_name}_greyscale.png')

        depth, hardness, depth_roughness, new_echodata = find_bottom(echodata_swap, params['move_avg_windowsize'])
        bottom_depth = depth 

        new_echodatax = new_echodata.copy()
        layer = find_layer(new_echodatax, params['beam_dead_zone'], params['layer_in_a_row'], params['layer_quantile'], params['layer_strength_thresh'], params['layer_size_thresh'])
        if layer:
            new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodata, params['wave_thresh_layer'], params['in_a_row_waves'], params['beam_dead_zone'])
        else:
            new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodata, params['wave_thresh'], params['in_a_row_waves'], params['beam_dead_zone'])
            if wave_avg > params['extreme_wave_size']: 
                new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodatax, params['wave_thresh_layer'], params['in_a_row_waves'], params['beam_dead_zone'])


        data_to_images(new_echodata, f'{img_path}/{new_file_name}_complete')
        os.remove(f'{img_path}/{new_file_name}_complete_greyscale.png')

        find_contours(new_echodata, new_echodata.copy(), gps_data, ping_times, new_file_name, img_path, bottom_depth, results)

    except KeyboardInterrupt:
        save_results(results, csv_path)
    except Exception as error:
        traceback.print_exc()
   

def find_contours(new_echodata, original_echodata, gps_data, ping_times, new_file_name, img_path, bottom_depth, results):
    """

    Finds contours in processed echodata 

    Args:
        new_echodata (numpy.ndarray): Numpy array with proccessed echodata
        original_echodata (numpy.ndarray): Original echodata array before processing.
        gps_data (list): List of GPS data corresponding to each ping.
        ping_times (list): List of times corresponding to each ping.
        new_file_name (str): Name of the processed file.
        img_path (str): Path to save images or results.
        bottom_depth (float): Depth of the bottom detected in the echodata.
        results (dict): Dictionary or structure to store extracted results.

    Returns:
        None

    """
    #Create binary image for contour detection
    binary_echodata = np.where((new_echodata > -70) & (new_echodata < 0), 1, 0)
    image = (binary_echodata * 255).astype(np.uint8)
    _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    
    #Blur the binary image
    blur = cv2.blur(binary_image, (5,10))

    #Find contours and their convex hulss
    contours, _ = cv2.findContours(blur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convex_hulls = [cv2.convexHull(c) for c in contours]
    
    for contour, hull in zip(contours, convex_hulls):
        filter_and_extract_contours(hull, contour, original_echodata, binary_image, gps_data, ping_times, new_file_name, img_path, bottom_depth, results)


def filter_and_extract_contours(hull, contour, original_echodata, binary_image, gps_data, ping_times, new_file_name, img_path, bottom_depth, results):
    """
    Filters and extracts information from a contour 

    Args:
        hull (numpy.ndarray): Contour hull obtained from find contours
        original_echodata (numpy.ndarray): Original echodata array before processing.
        binary_image (numpy.ndarray): Echodata after thresholding 
        gps_data (pd.DataFrame): DataFrame containing GPS data corresponding to each ping.
        ping_times (list): List of times corresponding to each ping.
        new_file_name (str): Name of the processed file being analyzed.
        img_path (str): Path to save images or results.
        bottom_depth (float): Depth of the bottom detected in the echodata.
        results (list): List to store dictionaries with extracted information from contours.

    Returns:
        None

    Notes:
        - This function filters contours based on area, intensity, and density criteria.
        - Extracts information such as centroid coordinates, depth, GPS coordinates, dimensions, and acoustic metrics.
        - Draws filtered contours on the original echodata and saves images.
        - Stores extracted information in the `results` list as dictionaries.
    """

    # Calculate the area of the contour and filter out small contours
    area = cv2.contourArea(contour)
    if area < 200:
        return

    # Create and apply a mask to isolate the fish school within the original echodata
    mask = np.zeros_like(original_echodata, dtype=np.uint8) # Initialize a mask of the same shape as original_echodata with zeros
    cv2.fillPoly(mask, [contour], 255) # Fill the contour in the mask with 255
    targetROI = np.where(mask == 255, original_echodata, np.nan) # Apply mask to orignal echodata. TargetROI will have orignal echodata values inside the contour and NaN outside

    # Calculate the mean intensity ignoring NaN values and filter out low intensity contours
    intensity = np.nanmean(targetROI)
    
    # if intensity < -80:
    #     return
   
    # Apply the mask to the binary image and calculate the density of white pixels within the contour and filter out low density contours
    targetROI_binary = cv2.bitwise_and(binary_image, mask) # Apply the mask to the binary image 
    white_pixels_count = np.count_nonzero(targetROI_binary)# Count the number of white pixels
    density = white_pixels_count / area 
    if density < 0.35:
        return

    # Draw the filtered contour on the original echodata for visualization
    #cv2.drawContours(original_echodata, [hull], 0, (0, 255, 0), 1)
    cv2.drawContours(original_echodata, [contour], 0, (0, 255, 0), 1)

    # Calculate the centroid of the contour using moments
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # Retrieve the corresponding time and GPS coordinates for the centroid of the contour
    time = ping_times[cX]
    depth = cY * 0.1
    index_of_coordinate = np.argmin(np.abs(gps_data['Datetime'].values - time))
    latitude = gps_data['Latitude'][index_of_coordinate]
    longitude = gps_data['Longitude'][index_of_coordinate]

    # Calculate the bounding box dimensions of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Calculate NASC metrics within the contour
    targetROI = ma.masked_equal(targetROI, 0)
    nasc = 4 * np.pi * (1852 ** 2) * (10 ** (targetROI / 10)) * 0.1
    nasc_total = np.sum(nasc)
    nasc_mean = np.mean(nasc)

    # Append the extracted information to the results list and saves the image 
    results.append({
        'time': time,
        'longitude': longitude,
        'latitude': latitude,
        'depth': depth,
        'width': w,
        'height': h,
        'area': area,
        'nasc_mean': nasc_mean,
        'nasc_total': nasc_total,
        'intensity': intensity,
        'density': density,
        'bottom_depth': np.median(bottom_depth) * 0.1
    })

    data_to_images(original_echodata, f'{img_path}/{new_file_name}_clusters_on_original')
    os.remove(f'{img_path}/{new_file_name}_clusters_on_original_greyscale.png')

def save_results(results, csv_path):
    """
    Saves results to a CSV file.

    Args:
        results (list of dict): List of dictionaries containing results to be saved.
        csv_path (str): File path where the CSV file will be saved.

    Returns:
        None
    """
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(df.head())


def main():
    """
    Main function for processing raw files using specified parameters and saving results.

    This function reads command-line arguments for file paths and parameters,
    loads necessary data, processes files, and saves results to a CSV file.

    Command-line Arguments:
        sys.argv[1] (str): Path to the directory containing raw files to process.
        sys.argv[2] (str): Path to the file containing completed file names.
        sys.argv[3] (str): Path to create an empty file to track newly processed files.
        sys.argv[4] (str): Path to save the CSV file containing results.
        sys.argv[5] (str): Path to the YAML file containing processing parameters.
        sys.argv[6] (str): Path to save images generated during processing.
        sys.argv[7] (str): Path to the CSV file containing GPS data.
    """
    try: 
        if len(sys.argv) < 8:
            raise ValueError("Insufficient command-line arguments provided.")

        params_path = sys.argv[5]
        path = sys.argv[1]
        completed_files_path = sys.argv[2]
        new_processed_files_path = sys.argv[3]
        csv_path = sys.argv[4]
        img_path = sys.argv[6]
        gps_path = sys.argv[7]

        params = load_params(params_path)
        sonar_depth = params[0]['sonar_depth']
        gps_data = load_gps_data(gps_path)

        
        open(new_processed_files_path, "w").close()
        files = get_files_to_process(path, completed_files_path)

        results = []
        if files:
            for file in files:
                print(f"Processing: {file}")
                if file.endswith('.raw'):
                    process_file(file, path, params[0], img_path, gps_data, [], results, csv_path, completed_files_path)

        save_results(results, csv_path)

    except Exception as e:
        print(f"Error in main(): {e}")

if __name__ == "__main__":
    main()