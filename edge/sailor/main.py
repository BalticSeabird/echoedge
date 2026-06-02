import numpy as np
import warnings
import sys
import yaml
import os
import sqlite3
import traceback

from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../../lib'))

from yaml.loader import SafeLoader
from find_bottom import find_bottom
from export_data import save_data
from find_fish import find_fish_median, medianfun
from visualization import data_to_images
from processing import process_data, clean_times, remove_vertical_lines
from find_waves import find_waves, find_layer

warnings.filterwarnings("ignore")

METADATA_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../metadatabase/sailbuoy_metadatabase.db'))
PROCESSING_METADATA_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_processing_metadata (
    id INTEGER PRIMARY KEY,
    platform TEXT NOT NULL,
    file_name TEXT NOT NULL,
    first_timestamp_utc TEXT,
    last_timestamp_utc TEXT,
    processed_at_utc TEXT NOT NULL,
    UNIQUE(platform, file_name)
)
"""


def _log_unreadable(completed_files_path, filename, reason):
    try:
        base_dir = os.path.dirname(os.path.abspath(completed_files_path))
        log_path = os.path.join(base_dir, 'unreadable_files.txt')
        with open(log_path, 'a') as f:
            f.write(f"{filename}\t{reason}\n")
    except Exception:
        pass


def _ensure_file(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, 'a').close()


def _ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def _utc_now_string():
    return pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')


def _format_timestamp_utc(value):
    if value is None:
        return None

    timestamp = pd.to_datetime(value, utc=True, errors='coerce')
    if pd.isna(timestamp):
        return None

    return timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')


def _get_platform_name(params_path):
    if isinstance(params_path, str) and params_path:
        return Path(params_path).stem

    return 'unknown'


def _create_metadata_database(db_path):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.execute(PROCESSING_METADATA_SCHEMA)
    connection.commit()
    return connection


def _save_processing_metadata(connection, platform, file_name, first_timestamp, last_timestamp):
    connection.execute(
        """
        INSERT OR REPLACE INTO file_processing_metadata (
            platform,
            file_name,
            first_timestamp_utc,
            last_timestamp_utc,
            processed_at_utc
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (
            platform,
            file_name,
            _format_timestamp_utc(first_timestamp),
            _format_timestamp_utc(last_timestamp),
            _utc_now_string(),
        ),
    )
    connection.commit()


# Load all params from yaml-file
with open(sys.argv[5], 'r') as f:
    params = list(yaml.load_all(f, Loader=SafeLoader))

platform_name = sys.argv[6] if len(sys.argv) > 6 else _get_platform_name(sys.argv[5])
metadata_connection = _create_metadata_database(METADATA_DB_PATH)


# Remove already processed files  
path = sys.argv[1]
completed_files_path = sys.argv[2]
new_processed_files_path = sys.argv[3]
output_path = sys.argv[4]
csv_path = output_path+'/csv'
img_path = output_path+'/img'

_ensure_directory(csv_path)
_ensure_directory(img_path)


files = os.listdir(path)

_ensure_file(completed_files_path)
_ensure_file(new_processed_files_path)

with open(completed_files_path, 'r') as completed_txt_file:
    completed_files = [line for line in completed_txt_file.readlines()]
completed_files = [file.replace('\n', '') for file in completed_files]

files = [f for f in files if f not in completed_files]

open(new_processed_files_path, "w").close()

if files:
    for file in files:
        print(file)
        if '.raw' in file:
            try: 
                # Build absolute path and pre-check read access before processing
                filepath = os.path.abspath(os.path.join(path, file))
                new_file_name = os.path.splitext(os.path.basename(filepath))[0]
                if not os.path.isfile(filepath) or not os.access(filepath, os.R_OK):
                    print(f'Skipping {file}: no read permission on {filepath}')
                    _log_unreadable(completed_files_path, file, f'No read permission on {filepath}')
                    continue

                # Load and process the raw data files
                echodata, ping_times = process_data(filepath, params[0]['env_params'], params[0]['cal_params'], params[0]['bin_size'], 'BB')
                echodata = echodata.Sv.to_numpy()[0]
                echodata, nan_indicies = remove_vertical_lines(echodata)
                echodata_swap = np.swapaxes(echodata, 0, 1)
                
                data_to_images(echodata_swap, f'{img_path}/{new_file_name}', upper=params[0]['image_upper'], lower=params[0]['image_lower']) # save img without ground
                os.remove(f'{img_path}/{new_file_name}_greyscale.png')

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
 
                data_to_images(new_echodata, f'{img_path}/{new_file_name}_complete', upper=params[0]['image_upper'], lower=params[0]['image_lower']) # save img without ground and waves
                os.remove(f'{img_path}/{new_file_name}_complete_greyscale.png')
                os.remove(f'{img_path}/{new_file_name}_complete.png')

                # Find fish cumsum, median depth and inds
                depth = [int(d) for d in depth]
                
                nasc = find_fish_median(echodata, wave_line, depth) 
                nasc0, fish_depth0 = medianfun(nasc, params[0]['fish_layer0_start'], params[0]['fish_layer0_end'], params[0]['no_bottom_default']-1)
                nasc1, fish_depth1 = medianfun(nasc, params[0]['fish_layer1_start'], params[0]['fish_layer1_end'], params[0]['no_bottom_default']-1)
                nasc2, fish_depth2 = medianfun(nasc, params[0]['fish_layer2_start'], params[0]['fish_layer2_end'], params[0]['no_bottom_default']-1)
                nasc3, fish_depth3 = medianfun(nasc, params[0]['fish_layer3_start'], params[0]['fish_layer3_end'], params[0]['no_bottom_default']-1)

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

                first_timestamp = ping_times[0] if len(ping_times) else None
                last_timestamp = ping_times[-1] if len(ping_times) else None

        
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
                _save_processing_metadata(metadata_connection, platform_name, file, first_timestamp, last_timestamp)

                # Mark as completed only after successful save
                with open(completed_files_path, 'a') as txt_doc:
                    txt_doc.write(f'{file}\n')

            except PermissionError as error:
                print(f'Skipping {file}: Permission denied -> {error}')
                _log_unreadable(completed_files_path, file, f'Permission denied on {filepath}')
                continue
            except Exception as error:
                traceback.print_exc()
                print(f'Problems with {file}')
                _log_unreadable(completed_files_path, file, f'Processing failed: {error.__class__.__name__}: {error}')
                continue

else:
    print('All exising files already processed and analyzed.')

metadata_connection.close()

# Run example main 
""" 
    python3 edge/sailor/main.py \
    ../../../../../../mnt/BSP_NAS2/Acoustics/SLU_Sailor2/Raw_data/2026 \
    edge/sailor/completed_files.txt \
    edge/sailor/new_processed_files.txt \
    ../../../../../../mnt/BSP_NAS2_work/Acoustics_output_data/Echopype_results/Baltic2026 \
    edge/sailor/paramsSillenBaltic2026.yaml \
    SAILOR2  """