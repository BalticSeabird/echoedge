from processing import process_data, extract_meta_data, remove_vertical_lines, clean_times
from visualization import data_to_images
import yaml 
from yaml.loader import SafeLoader
import numpy as np
from pathlib import Path

file = Path('../../../../../../../../mnt/BSP_NAS2/Acoustics/VOTO_Sailbuoy/HudsonBay_2024/Raw_files/WBAT-Phase0-D20240628-T160112-0.raw')


with open('params_Baltic2024.yaml', 'r') as f:
    params = list(yaml.load_all(f, Loader=SafeLoader))

# Loop through all files
new_file_name = file.stem

# Load and process the raw data files
echodata, ping_times = process_data(file, params[0]['env_params'], params[0]['cal_params'], params[0]['bin_size'], 'BB')
echodata = echodata.Sv.to_numpy()[0]
echodata, nan_indicies = remove_vertical_lines(echodata)
echodata_swap = np.swapaxes(echodata, 0, 1)

data_to_images(echodata_swap, '../../out/img2/', normalization = False, upper = -30, lower = -90) # save img without ground
