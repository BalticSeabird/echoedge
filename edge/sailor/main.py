import numpy as np
import warnings
import sys
import yaml
import os
import traceback

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from yaml.loader import SafeLoader
from find_bottom import find_bottom, find_dead_zone
from export_data import save_data
from find_fish import find_fish_median, medianfun
from visualization import data_to_images
from processing import process_data, clean_times, remove_vertical_lines
from find_waves import find_waves, find_layer

warnings.filterwarnings("ignore")


# Load all params from yaml-file
with open(sys.argv[5], 'r') as f:
    params = list(yaml.load_all(f, Loader=SafeLoader))


# Remove already processed files  
path = sys.argv[1]
completed_files_path = sys.argv[2]
new_processed_files_path = sys.argv[3]
csv_path = sys.argv[4]
img_path = sys.argv[6]


files = os.listdir(path)

completed_txt_file = open(completed_files_path, 'r')
completed_files = [line for line in completed_txt_file.readlines()]
completed_files = [file.replace('\n', '') for file in completed_files]

files = [f for f in files if f not in completed_files]

open(new_processed_files_path, "w").close()

if files:
    for file in files:
        print(file)
        if '.raw' in file:
            try: 
                with open(completed_files_path, 'a') as txt_doc:
                    txt_doc.write(f'{file}\n')

                filepath = f'{path}/{file}'
                new_file_name = filepath.split('/')[-1].replace('.raw', '')

                # Load and process the raw data files
                echodata, ping_times = process_data(filepath, params[0]['env_params'], params[0]['cal_params'], params[0]['bin_size'], 'BB')
                echodata = echodata.Sv.to_numpy()[0]
                echodata, nan_indicies = remove_vertical_lines(echodata)
                echodata_swap = np.swapaxes(echodata, 0, 1)

                data_to_images(echodata_swap, f'{img_path}/{new_file_name}') # save img without ground
                os.remove(f'{img_path}/{new_file_name}_greyscale.png')

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
 
                data_to_images(new_echodata, f'{img_path}/{new_file_name}_complete') # save img without ground and waves
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
        

                save_data(data_dict, file.replace('.raw', '.csv'), csv_path, new_processed_files_path)
             
            except Exception as error:
                traceback.print_exc()
                print(f'Problems with {file}')
                continue

else:
    print('All exising files already processed and analyzed.')