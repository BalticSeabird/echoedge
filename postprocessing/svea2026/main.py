
# Postprocessing pipeline for Svea 2026. 

# RUN EXAMPLE: 

"""
python3 postprocessing/svea2026/main.py \
    postprocessing/svea2026/params.yaml \
    /../../../../../../Volumes/JHS-SSD2/RUTSPRAS_2026/raw/ \
    /../../../../../../Volumes/JHS-SSD2/RUTSPRAS_2026/csv/ \
    /../../../../../../Volumes/JHS-SSD2/RUTSPRAS_2026/img/
"""

import sys
import os

# Ensure the project's `lib` directory is on sys.path (works from any cwd)
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'lib'))
sys.path.insert(0, lib_path)

import numpy as np
import warnings
import yaml
import os
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
with open(sys.argv[1], 'r') as f:
    params = list(yaml.load_all(f, Loader=SafeLoader))

# Load external variables
raw_path = sys.argv[2]
csv_path = sys.argv[3]
img_path = sys.argv[4]
new_processed_data = False

# Load files

files = os.listdir(raw_path)
files = [file for file in files if '.raw' in file]
files.reverse()

if files:
    for file in tqdm.tqdm((files)): # reversed to run the opposite direction
        try: 
            print(file)
            
            filepath = f'{raw_path}/{file}'

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

                    data_to_images(
                        echodata_swap,
                        f'{img_path}/{new_file_name}',
                        upper=params[0]['image_upper'],
                        lower=params[0]['image_lower'],
                    ) # save img without ground

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
                    

                    data_to_images(
                        new_echodata,
                        f'{img_path}/{new_file_name}_complete',
                        upper=params[0]['image_upper'],
                        lower=params[0]['image_lower'],
                    ) # save img without ground

                    os.remove(f'{img_path}/{new_file_name}_greyscale.png')
                    os.remove(f'{img_path}/{new_file_name}_complete_greyscale.png')


                    # Fish calculations
                    nasc = find_fish_median(echodata, wave_line, depth)
                    max_depth = params[0]['no_bottom_default'] - 1
                    nasc0, fish_depth0 = medianfun(nasc, 0, 150, max_depth)
                    nasc1, fish_depth1 = medianfun(nasc, 0, 50, max_depth)
                    nasc2, fish_depth2 = medianfun(nasc, 50, 100, max_depth)
                    nasc3, fish_depth3 = medianfun(nasc, 100, 150, max_depth)

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
                    save_data(data_dict, f'{new_file_name}.csv', csv_path)

        except Exception as e:
            print(f'Errors with {new_file_name}: {e}')


else:
    print('All exising files already processed and analyzed.')