import numpy as np
import warnings
import sys
import yaml
import os
import traceback
import tqdm
import datetime

from yaml.loader import SafeLoader

from processing import process_data, extract_meta_data
from find_bottom import get_beam_dead_zone, find_bottom_for_svea
from find_fish import find_fish_median, medianfun
from find_waves import find_waves, find_layer
from visualization import data_to_images
from export_data import save_data, shorten_list

warnings.filterwarnings("ignore")


# Load all params from yaml-file
with open(sys.argv[5], 'r') as f:
    params = list(yaml.load_all(f, Loader=SafeLoader))

# Load external variables
path = sys.argv[1]
completed_files_path = sys.argv[2]
new_processed_files_path = sys.argv[3]
csv_path = sys.argv[4]
img_path = sys.argv[6]
sonar_depth = 7.6 # Is this correct?

# Load files
completed_txt_file = open(completed_files_path, 'r')
completed_files = [line for line in completed_txt_file.readlines()]
completed_files = [file.replace('\n', '') for file in completed_files]

files = os.listdir(path)
files = [file for file in files if '.raw' in file]
files = [f for f in files if f not in completed_files]
files.reverse()
open(new_processed_files_path, "w").close()

if files:
    for file in tqdm.tqdm((files[14:15])): # reversed to run the opposite direction
        try: 
            print(file)
            
            filepath = f'{path}/{file}'

            with open(completed_files_path, 'a') as txt_doc:
                txt_doc.write(f'{file}\n')

            raw_echodata, channels, longitude, latitude, transmit_types = extract_meta_data(filepath)
            del raw_echodata

            for i, (channel, transmit) in enumerate(zip(channels, transmit_types)):
                if 'ES38' in channel:
                    channel = channel.replace(" ", "_")
                    new_file_name = filepath.split('/')[-1].replace('.raw', '_') + channel

                    if all(x == transmit[0] for x in transmit):
                        transmit_type = transmit[0]
                    else: 
                        print(f'Error with {file}: Different transmit types in same channel')
                        break
                    
                    x = filepath.split('/')[-1].replace('.raw', '_') + channel

                    # Load and process the raw data files
                    echodata, ping_times = process_data(filepath, params[0]['env_params'], params[0]['cal_params'], params[0]['bin_size'], transmit_type)
                    echodata = echodata.Sv.to_numpy()[i]
                    echodata_swap = np.swapaxes(echodata, 0, 1)

                    data_to_images(echodata_swap, f'{img_path}/{new_file_name}') # save img without ground

                    # Find beam dead zone
                    beam_dead_zone = get_beam_dead_zone(echodata_swap)


                    # Find, measure and remove waves in echodata
                    new_echodatax = echodata_swap.copy()
                    layer = find_layer(new_echodatax, beam_dead_zone, params[0]['layer_in_a_row'], params[0]['layer_quantile'], params[0]['layer_strength_thresh'], params[0]['layer_size_thresh'])
                    if layer:
                        new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(echodata_swap, params[0]['wave_thresh_layer'], params[0]['in_a_row_waves'], params[0]['beam_dead_zone'])
                    else:
                        new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(echodata_swap, params[0]['wave_thresh'], params[0]['in_a_row_waves'], params[0]['beam_dead_zone'])

                        if wave_avg > params[0]['extreme_wave_size']: 
                            new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(echodata_swap, params[0]['wave_thresh_layer'], params[0]['in_a_row_waves'], params[0]['beam_dead_zone'])

                    # Find bottom
                    new_echodata, depth = find_bottom_for_svea(echodata_swap, wave_line)
                    

                    data_to_images(new_echodata, f'{img_path}/{new_file_name}_complete') # save img without ground

                    os.remove(f'{img_path}/{new_file_name}_greyscale.png')
                    os.remove(f'{img_path}/{new_file_name}_complete_greyscale.png')


                    # Fish calculations
                    nasc = find_fish_median(echodata, wave_line, depth)
                    nasc0, fish_depth0 = medianfun(nasc, 0, 150)
                    nasc1, fish_depth1 = medianfun(nasc, 0, 50)
                    nasc2, fish_depth2 = medianfun(nasc, 50, 100)
                    nasc3, fish_depth3 = medianfun(nasc, 100, 150)

                    for list in [nasc0, nasc1, nasc2, nasc3, fish_depth0, fish_depth1, fish_depth2, fish_depth3]:
                        list[:] = [round(x, 2) for x in list]


                    # Save data to csv
                    data_dict = {
                        'time': ping_times,
                        'latitude': shorten_list(latitude, len(ping_times)),
                        'longitude': shorten_list(longitude, len(ping_times)),
                        'depth': [d/10 for d in depth],
                        'wave_depth': [w/10 for w in wave_line],
                        'nasc0': nasc0,
                        'fish_depth0': fish_depth0,
                        'nasc1': nasc1,
                        'fish_depth1': fish_depth1,
                        'nasc2': nasc2,
                        'fish_depth2': fish_depth2,
                        'nasc3': nasc3,
                        'fish_depth3': fish_depth3,
                        'transmit_type': [transmit[0]] * len(ping_times),
                        'file': file,
                        'upload_time': [datetime.datetime.now() for i in range(len(ping_times))]
                    }

                    # Save data to csv
                    save_data(data_dict, f'{new_file_name}.csv', csv_path, new_processed_files_path)

        except Exception as e:
            print(f'Errors with {new_file_name}: {e}')


else:
    print('All exising files already processed and analyzed.')