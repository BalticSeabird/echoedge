import echopype as ep
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import tqdm
import cv2

from matplotlib.colors import Normalize



def swap_chan(ds: xr.Dataset) -> xr.Dataset:
    """Function to replace the channel dimension and coordinate with the
    frequency_nominal variable containing actual frequency values.
    Note that this step is possible only because there are
    no duplicated frequencies present.
    """
    return (
        ds.set_coords("frequency_nominal")
        .swap_dims({"channel": "frequency_nominal"})
        .reset_coords("channel")
    )


# Funktion för att ladda in och processera data från en raw-fil
def process_data(data_path, cal_params, env_params):
    raw_echodata = ep.open_raw(data_path, sonar_model="EK80")

    ds_Sv_raw = ep.calibrate.compute_Sv(
        raw_echodata,
        env_params=env_params,
        cal_params=cal_params,
        waveform_mode='BB',
        encode_mode="complex",
    )

    ds_MVBS = ep.commongrid.compute_MVBS(
        ds_Sv_raw, # calibrated Sv dataset
        range_meter_bin=0.1, # bin size to average along range in meters
        ping_time_bin='2S' # bin size to average along ping_time in seconds
    )

    ds_MVBS = ds_MVBS.pipe(swap_chan)

    selected_data = ds_MVBS.Sv.to_numpy()[0]  # ping 35 includes a lot of interesting stuff
    return selected_data

def create_normal_echogram(data, filepath='echogram'):
    np_data = np.nan_to_num(data, copy=True)
    np_data[np_data<-90]= -90
    np_data = (np_data - np_data.min())/(np_data.max() - np_data.min())
    np_data = np_data*256
    
    flip_np_data = np.rot90(np_data, 3) # flip the image to display it correctly

    cv2.imwrite(f'{filepath}_greyscale.png', flip_np_data) # greyscale image

    image = cv2.imread(f'{filepath}_greyscale.png', 0)
    colormap = plt.get_cmap('viridis') # 
    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'{filepath}.png', heatmap) 


def create_echogram(data, cmap, vmin, vmax, filepath='echogram'):
    np_data = np.nan_to_num(data, copy=True)
    np_data[np_data<-90]= -90
    np_data = (np_data - np_data.min())/(np_data.max() - np_data.min())
    np_data = np_data*256
    
    flip_np_data = np.rot90(np_data, 3) # flip the image to display it correctly

    cv2.imwrite(f'{filepath}_greyscale.png', flip_np_data) # greyscale image

    image = cv2.imread(f'{filepath}_greyscale.png', 0)
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap) # 
    heatmap = (colormap(norm(image)) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'{filepath}.png', heatmap) 


if __name__ == "__main__":

    envs = {
        'temperature': 16,
        'salinity': 9,
        'pressure': 10.1325
    }

    cals = {
        'gain_correction': 28.49,
        'equivalent_beam_angle': -21
    }

    path = '../../Raw_data/SLUAquaSailor2020V2-Phase0-D20200627-T023149-0.raw'
    # path = '../../Calibration 20210517/SLUAquaSailor2020CALIBRATION-Phase0-D20210517-T090725-0.raw'
    
    data = process_data(path, cals, envs)

    create_normal_echogram(data, 'echogram_normal')

    data[data < -90] = 0
    data[data > -84] = 0

    cmap = 'viridis'
    cmaps = plt.colormaps()
    cmaps = ['Accent', 'autumn_r', 'brg_r', 'CMRmap_r', 'gist_ncar_r', 'BrBG', 'cividis', 'viridis', 'Dark2', 'Flag']

    vmins = [(i*25+15) for i in range(9)]
    vmaxs = [(i*25+25) for i in range(10)]

    vmins = [0]
    vmaxs = [70]
    # Definiera nya gränsvärden för colormapen

    for cmap in tqdm.tqdm(cmaps):
        for vmax in vmaxs:
            for vmin in vmins:
                try:
                    create_echogram(data, cmap, vmin, vmax, f'echogram_{cmap}_{vmin}_{vmax}')
                except:
                    print('Error')
                    continue

