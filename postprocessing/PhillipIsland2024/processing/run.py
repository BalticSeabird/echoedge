import numpy as np
import pandas as pd
import warnings
import sys
import yaml
import os
import traceback
import tqdm
import datetime
from dateutil import tz
from math import radians, sin, cos, sqrt, atan2
 
from yaml.loader import SafeLoader

from processing import process_data, extract_meta_data, remove_vertical_lines, clean_times, get_interpolated_gps2
from find_bottom import get_beam_dead_zone, find_bottom
from find_fish import find_fish_median, medianfun
from find_waves import find_waves, find_layer
from visualization import data_to_images
from export_data import save_data, shorten_list

warnings.filterwarnings("ignore")


# Load all params from yaml-file
with open('../PI_parameters.yaml', 'r') as f:
    params = list(yaml.load_all(f, Loader=SafeLoader))

csv_path = "D:/SURVEY2023/PREPROCESS_DATA/Csv"
img_path = "D:/SURVEY2023/PREPROCESS_DATA/Img"
npy_path = "D:/SURVEY2023/PREPROCESS_DATA/npy"
files_path = "D:/SURVEY2023/SURVEY RAW DATA/Files" #2023 path
# files_p = pd.read_csv('../../../test/Rerun/Rerun_files.csv')
# files = files_p['raw_list']

files = []
names = []
for path, subdirs, files_name in os.walk(files_path):
    for name in files_name:
        files.append(os.path.join(path, name))
        names.append(name)

######################################################################
##########       The interpolation of the GPS position      ##########
gps_files_30 = '../../../test/Data_GPS/Interpolated_30sec'
gps_files_2 = '../../../test/Data_GPS/Interpolated_2sec'
frequency_value = 2
mean_velocity = 0.63   # to decide the velocity here by histogram output

interpolated_df = pd.DataFrame()
for gps_file in os.listdir(gps_files_30):
    file_path_gps = os.path.join(gps_files_2,gps_file)
    new_file_name_gps = file_path_gps.replace('_aussi_time.gps.csv', '_interpolated.csv')
    if os.path.exists(new_file_name_gps):
        print(f'The file {new_file_name_gps} exists.')
        interpolated = pd.read_csv(new_file_name_gps)
        interpolated_df=pd.concat([interpolated_df, interpolated])
    else:
        file_path_gps_30 = os.path.join(gps_files_30,gps_file)
        interpolated = get_interpolated_gps2(file_path_gps_30,frequency=frequency_value, ltz = params[0]['ltz'])
        interpolated.to_csv(new_file_name_gps)
        interpolated_df=pd.concat([interpolated_df, interpolated])

interpolated_df = interpolated_df.reset_index(drop=True)

upper = -30
lower = -95
#index i

# Run 
for file in tqdm.tqdm(files[:]):
    
    try:
        if '.raw' or '.RAW' in file:
            filepath = file
            name = file[-50:]
            new_file_name =  name.replace(name[-4:], '')

            # Load and process the raw data files
            echodata, ping_times = process_data(filepath, params[0]['env_params'], params[0]['cal_params'], params[0]['bin_size'], 'BB')
            echodata = echodata.Sv.to_numpy()[0]

            #UTC Time
            ping_times_series = pd.to_datetime(ping_times)

            # Correction for January files
            if ping_times_series[0] > pd.to_datetime("2023-12-17 05:00:00") :
                echodata = echodata + 22.451
            
            echodata, nan_indicies = remove_vertical_lines(echodata)
            echodata_swap = np.swapaxes(echodata, 0, 1)
            complete_echodata = echodata.copy() # create a copy to analyse bottom signal
            
                    
            data_to_images(echodata_swap, f'{img_path}/{new_file_name}',f'{npy_path}/{new_file_name}', upper = upper, lower = lower) # save img with ground
            #os.remove(f'{img_path}/{new_file_name}_greyscale.png')


            # Detect bottom algorithms
            depth, hardness, depth_roughness, new_echodata = find_bottom(echodata_swap, params[0]['move_avg_windowsize'], params[0]['bottom_hardness_thresh'])

            # Find, measure and remove waves in echodata
            new_echodatax = new_echodata.copy()
            layer = find_layer(new_echodatax, params[0]['beam_dead_zone'], params[0]['layer_in_a_row'], params[0]['layer_quantile'], params[0]['layer_strength_thresh'], params[0]['layer_size_thresh'])
            if layer:
                new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodata, params[0]['wave_thresh_layer'], params[0]['in_a_row_waves'], params[0]['beam_dead_zone'])
            else:
                new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodata, params[0]['wave_thresh'], params[0]['in_a_row_waves'], params[0]['beam_dead_zone'])

                if wave_avg > params[0]['extreme_wave_size']: 
                    new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodatax, params[0]['wave_thresh_layer'], params[0]['in_a_row_waves'], params[0]['beam_dead_zone'])

            data_to_images(new_echodata, f'{img_path}/{new_file_name}_complete',f'{npy_path}/{new_file_name}', upper = upper, lower = lower) # save img without ground and waves
            os.remove(f'{img_path}/{new_file_name}_complete_greyscale.png')

            # Find fish cumsum, median depth and inds
            depth = [int(d) for d in depth]
            
            nasc = find_fish_median(echodata, wave_line, depth) 
            nasc0, fish_depth0 = medianfun(nasc, params[0]['fish_layer0_start'], params[0]['fish_layer0_end'])
            nasc1, fish_depth1 = medianfun(nasc, params[0]['fish_layer1_start'], params[0]['fish_layer1_end'])
            nasc2, fish_depth2 = medianfun(nasc, params[0]['fish_layer2_start'], params[0]['fish_layer2_end'])
            nasc3, fish_depth3 = medianfun(nasc, params[0]['fish_layer3_start'], params[0]['fish_layer3_end'])
            
            #change from dm to meters 
            depth = [i*0.1 for i in depth if i != 0]
            depth_roughness = round(0.1 * depth_roughness if depth_roughness != 0 else 0, 2)
            wave_line = [i*0.1 for i in wave_line if i != 0]
         
            #adding sonar depth to depth variables 
            for i in range(len(depth)):
                if depth[i] != 100:
                    depth[i] += params[0]['sonar_depth']
                    
            for depth_list in [wave_line, fish_depth0, fish_depth1, fish_depth2, fish_depth3]:
                for i in range(len(depth_list)):
                    if sum(depth_list)/len(depth_list) != 0:
                        depth_list[i] += params[0]['sonar_depth']

            #round values to two decimal places
            for list in [depth, hardness, wave_line, nasc0, nasc1, nasc2, nasc3, fish_depth0, fish_depth1, fish_depth2, fish_depth3]:
                list[:] = [round(x, 2) for x in list]

            if nan_indicies.size != 0:
                ping_times_series = clean_times(ping_times_series, nan_indicies)
            
            # Get lat and long to match with ping times 
            Datetime_UTC = pd.Series(ping_times_series, name = 'Datetime_UTC')
            interpolated_df['Datetime_UTC'] = pd.to_datetime(interpolated_df['Datetime_UTC'])
            
            LatLong = interpolated_df.merge(Datetime_UTC, on = "Datetime_UTC", how = "inner")
            
            # Define the output
            Lat = LatLong["Latitude"]
            Long = LatLong["Longitude"]
            UTC_time = LatLong['Datetime_UTC']   
            Velocity = LatLong['Velocity']
            Datetime_local = LatLong['Datetime_local']
            

            # Save all results in dict
            data_dict = {
                'UTC_time': UTC_time,
                'Local_time' : Datetime_local,
                'Lat': Lat,
                'Long': Long,
                'Velocity' : Velocity,
                'bottom_hardness': hardness,
                'bottom_roughness': depth_roughness,
                'wave_depth': wave_line,
                'depth': depth,
                'nasc0': nasc0,
                'fish_depth0': fish_depth0, 
                'nasc1': nasc1,
                'fish_depth1': fish_depth1, 
                'nasc2': nasc2,
                'fish_depth2': fish_depth2, 
                'nasc3': nasc3,
                'fish_depth3': fish_depth3, 
            }
            df = pd.DataFrame(data_dict)

            save_data(data_dict, f'{new_file_name}.csv', csv_path)
            
    except Exception as e:
         print(f'Error in file {file}')
         print(e)
    continue