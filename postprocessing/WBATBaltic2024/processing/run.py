import numpy as np
import pandas as pd
import warnings
import sys
import yaml
import os
import tqdm
from pathlib import Path

from yaml.loader import SafeLoader

from processing import process_data, extract_meta_data, remove_vertical_lines, clean_times
from find_bottom import get_beam_dead_zone, find_bottom
from find_fish import find_fish_median, medianfun
from find_waves import find_waves, find_layer
from visualization import data_to_images
from export_data import save_data, shorten_list

warnings.filterwarnings("ignore")


# Load all params from yaml-file
with open('postprocessing/SailorBaltic2024/params_Baltic2024.yaml', 'r') as f:
    params = list(yaml.load_all(f, Loader=SafeLoader))

csv_path = 'out/WBAT_Karlso2024/csv'
img_path = 'out/WBAT_Karlso2024/img'
#file_path = "../../../../../Volumes/JHS-SSD2/WBAT_St_Karlso"
file_path = Path("../../../../../../mnt/BSP_NAS2/Sailor/WBAT/Karlso2024/Raw_files")

files = file_path.rglob("*.raw")
#files = [file for file in files if file.startswith('WBAT-Phase0-')]

# Set parameter values for echogram normalization
upper = -30
lower = -95


# Test 
file = list(files)[10]
raw_echodata, channels, longitude, latitude, transmit_types = extract_meta_data(file)
import echopype as ep
raw_echodata = ep.open_raw(file, sonar_model='EK80')


# Run 
for file in tqdm.tqdm(files):
    try:
        new_file_name = file.stem

        # Load and process the raw data files
        echodata, ping_times = process_data(file, params[0]['env_params'], params[0]['cal_params'], params[0]['bin_size'], "BB")
        echodata = echodata.Sv.to_numpy()[0]
        echodata, nan_indicies = remove_vertical_lines(echodata)
        echodata_swap = np.swapaxes(echodata, 0, 1)

        data_to_images(echodata_swap, f'{img_path}/{new_file_name}', upper = upper, lower = lower) # save img without ground
        #os.remove(f'{img_path}/{new_file_name}_greyscale.png')


        # Detect bottom algorithms
        depth, hardness, depth_roughness, new_echodata = find_bottom(echodata_swap, params[0]['move_avg_windowsize'])

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
            ping_times = clean_times(ping_times, nan_indicies)

        # Get lat and long to match with ping times 
        Datetime = pd.Series(ping_times, name = "Datetime")

        # Save all results in dict
        data_dict = {
            'time': ping_times,
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


        save_data(data_dict, file.replace('.raw', '.csv'), csv_path)
    except Exception as e:
        print(f'Error in file {file}')
        print(e)
        continue