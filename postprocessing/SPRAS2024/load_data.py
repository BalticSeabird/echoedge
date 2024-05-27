import numpy as np
import warnings
import sys
import yaml
import os
import traceback
import tqdm
import datetime

from yaml.loader import SafeLoader

from processing import extract_meta_data


warnings.filterwarnings("ignore")

# Load external variables
path = 'data'


files = os.listdir(path)
files = [file for file in files if '.raw' in file]

if files:
    for file in tqdm.tqdm((files[:1])): # reversed to run the opposite direction
        print(file)
        
        filepath = f'{path}/{file}'

        raw_echodata, channels, longitude, latitude, transmit_types = extract_meta_data(filepath)
        print(raw_echodata)
        