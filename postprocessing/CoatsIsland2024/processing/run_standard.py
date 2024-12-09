import numpy as np
import warnings
import yaml
import os
import tqdm
from pathlib import Path
import pandas as pd

from yaml.loader import SafeLoader

from processing import process_data, extract_meta_data, remove_vertical_lines, clean_times
from find_bottom import get_beam_dead_zone, find_bottom
from find_fish import find_fish_median, medianfun
from find_waves import find_waves, find_layer
from visualization import data_to_images
from export_data import save_data, shorten_list

warnings.filterwarnings("ignore")


csv_path = '../../../../../../../../mnt/BSP_NAS2_work/Acoustics_output_data/Echopype_results/Hudson2024/run9dec24/csv'
img_path = '../../../../../../../../mnt/BSP_NAS2_work/Acoustics_output_data/Echopype_results/Hudson2024/run9dec24/img'
npy_path = '../../../../../../../../mnt/BSP_NAS2_work/Acoustics_output_data/Echopype_results/Hudson2024/run9dec24/npy'
file_path = '../../../../../../../../mnt/BSP_NAS2/Acoustics/VOTO_Sailbuoy/HudsonBay_2024/Raw_files/'


files = Path(file_path).glob("*-0.raw")


# GPS Coordinates 
interpolated_df = pd.read_csv('coords_data/interpolated_coords_coats24.csv')

# Load all params from yaml-file
with open('params_coats24.yaml', 'r') as f:
    params = list(yaml.load_all(f, Loader=SafeLoader))

# Plot thresholds 
upper = -30
lower = -100


# Loop through all files
for file in tqdm.tqdm(files):
    try:
        new_file_name = file.stem

        # Load and process the raw data files
        echodata, ping_times = process_data(file, params[0]['env_params'], params[0]['cal_params'], params[0]['bin_size'], 'BB')
        echodata = echodata.Sv.to_numpy()[0]
        echodata, nan_indicies = remove_vertical_lines(echodata)
        echodata_swap = np.swapaxes(echodata, 0, 1)

        data_to_images(echodata_swap, f'{img_path}/{new_file_name}', upper = upper, lower = lower) # save img without ground

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

        data_to_images(new_echodata, f'{img_path}/{new_file_name}_complete', upper = upper, lower = lower) # save img without ground and waves

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
            if depth[i] != 150:
                depth[i] += params[0]['sonar_depth']
                
        for depth_list in [wave_line, fish_depth0, fish_depth1, fish_depth2, fish_depth3]:
            for i in range(len(depth_list)):
                if sum(depth_list)/len(depth_list) != 0:
                    depth_list[i] += params[0]['sonar_depth']

        #round values to two decimal places
        for list in [depth, hardness, wave_line, nasc0, nasc1, nasc2, nasc3, fish_depth0, fish_depth1, fish_depth2, fish_depth3]:
            list[:] = [round(x, 2) for x in list]

        if nan_indicies.size != 0:
            ping_times = clean_times(ping_times, nan_indicies)


        # Link to GPS coordinates
        Datetime_UTC = pd.Series(ping_times, name = 'Datetime_UTC')
        interpolated_df['Datetime_UTC'] = pd.to_datetime(interpolated_df['Datetime'])
        LatLong = interpolated_df.merge(Datetime_UTC, on = "Datetime_UTC", how = "inner")
        
        Lat = LatLong["Lat"]
        Long = LatLong["Long"]
        UTC_time = pd.to_datetime(LatLong['Datetime'], utc = True)   
        #Velocity = LatLong['Velocity']
        Datetime_local = UTC_time.dt.tz_convert("US/Eastern")

        # Save all results in dict
        data_dict = {
            'UTC_time': UTC_time,
            'Local_time' : Datetime_local,
            'Lat': Lat,
            'Long': Long,
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



        save_data(data_dict, f'{new_file_name}.csv', csv_path)
    except Exception as e:
        print(f'Error in file {file}')
        print(e)
        continue