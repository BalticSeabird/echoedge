import echopype as ep
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import streamlit as st
import cv2
import os
import random
import pandas as pd
import shutil

from matplotlib.colors import Normalize

def swap_chan(ds: xr.Dataset) -> xr.Dataset:
    return (
        ds.set_coords("frequency_nominal")
        .swap_dims({"channel": "frequency_nominal"})
        .reset_coords("channel")
    )

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
        ds_Sv_raw,
        range_meter_bin=0.1,
        ping_time_bin='2S'
    )

    ds_MVBS = ds_MVBS.pipe(swap_chan)
    selected_data = ds_MVBS.Sv.to_numpy()[0]
    return selected_data

def create_normal_echogram(data, filepath='echogram'):
    np_data = np.nan_to_num(data, copy=True)
    np_data[np_data < -90] = -90
    np_data = (np_data - np_data.min()) / (np_data.max() - np_data.min())
    np_data = np_data * 256

    flip_np_data = np.rot90(np_data, 3)

    cv2.imwrite(f'{filepath}_greyscale.png', flip_np_data)

    image = cv2.imread(f'{filepath}_greyscale.png', 0)
    colormap = plt.get_cmap('viridis')
    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite('file_original.png', heatmap)

def create_echogram(data, cmap, vmin, vmax, blur_size=None, blur_type=None, kernel_first=None, kernel_second=None):
    np_data = np.nan_to_num(data, copy=True)
    np_data[np_data < -90] = -90
    np_data = (np_data - np_data.min()) / (np_data.max() - np_data.min())
    np_data = np_data * 256

    flip_np_data = np.rot90(np_data, 3)

    cv2.imwrite('file_greyscale.png', flip_np_data)

    image = cv2.imread('file_greyscale.png', 0)

    if blur_size and blur_type:
        if blur_type == "Median":
            image = cv2.medianBlur(image, blur_size)
        elif blur_type == "Kernels" and kernel_first and kernel_second:
            kernel = np.ones((kernel_first, kernel_second), np.float32) / (kernel_first * kernel_second)
            image = cv2.filter2D(image, -1, kernel)

    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    heatmap = (colormap(norm(image)) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite('file.png', heatmap)

def get_unique_folder_name(base_path):
    counter = 1
    new_path = base_path
    while os.path.exists(new_path):
        new_path = f"{base_path}_{counter}"
        counter += 1
    return new_path

def save_current_stats(filename, stats_dict):
    filename = filename.replace('.raw', '')
    base_destination_path = f'saved_results/{filename}'
    
    # Få en unik mappväg om mappen redan finns
    destination_file_path = get_unique_folder_name(base_destination_path)
    
    # Skapa målplatsen
    os.makedirs(destination_file_path, exist_ok=True)
    
    # Flytta filerna
    shutil.copy('file.png', os.path.join(destination_file_path, 'file.png'))
    shutil.copy('file_original.png', os.path.join(destination_file_path, 'file_original.png'))
    shutil.copy('file_greyscale.png', os.path.join(destination_file_path, 'file_greyscale.png'))
    
    # Spara Excel-filen
    print(stats_dict)
    df = pd.DataFrame(stats_dict)
    df.to_excel(f'{destination_file_path}/data_settings.xlsx', index=False)
    print(f"Filerna har sparats i {destination_file_path}.")



if __name__ == "__main__":
    st.title('Visualization lab of echograms')

    envs = {'temperature': 16, 'salinity': 9, 'pressure': 10.1325}
    cals = {'gain_correction': 28.49, 'equivalent_beam_angle': -21}

    kernels = {
        'Original': np.array([[0]]),
        'Blur': np.ones((10, 10), np.float32) / 100,
        'Edge Detection': np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]]),
        'Sharpen': np.array([[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]]),
        'Emboss': np.array([[-2, -1,  0], [-1,  1,  1], [ 0,  1,  2]])
    }

    cmaps = plt.colormaps()
    raw_path = '/mnt/BSP_NAS2/Sailor/Raw_data/'

    st.subheader("Select file")
    year = st.selectbox("Which year will you work with?", (2018, 2019, 2020, 2021, 2022, 2023, 2024), index=2)
    fileslist = os.listdir(f'{raw_path}/{year}')
    fileslist = [file for file in fileslist if file.endswith('-0.raw')]
    
    if 'selected_file_index' not in st.session_state:
        st.session_state.selected_file_index = random.randint(0, len(fileslist) - 1)
    
    if st.button("Randomize file"):
        st.session_state.selected_file_index = random.randint(0, len(fileslist) - 1)
    
    filename = st.selectbox("Filename", fileslist, index=st.session_state.selected_file_index)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.subheader("Adjust parameters")
        lower_limit = st.slider("Lower limit", -120, 0, value=-60)
        upper_limit = st.slider("Upper limit", -120, 0, value=-50)
        vmin = st.slider("Vmin", 0, 255, value=0)
        vmax = st.slider("Vmax", 0, 255, value=50)
        cmap = st.selectbox("Colormap", cmaps)

    path = f'{raw_path}/{year}/{filename}'
    data = process_data(path, cals, envs)
    create_normal_echogram(data)

    data[data < lower_limit] = 0
    data[data > upper_limit] = 0


    with col2:
        st.subheader("Blurring")
        blur = st.checkbox("Do you want to use blurring?")
        if blur:
            blur_type = st.radio("Which kind of blurring would you like to use?", ["Median", "Kernels"])
            if blur_type == "Median":
                blur_size = st.slider("Size of square in median blurring", 1, 199, value=3, step=2)
                create_echogram(data, cmap, vmin, vmax, blur_size=blur_size, blur_type=blur_type)
                if st.button("Save current stats"):
                    stats_dict = {
                        'lower_limit': [lower_limit],
                        'upper_limit': [upper_limit],
                        'vmin': [vmin],
                        'vmax': [vmax],
                        'cmap': [cmap],
                        'blur_type': [blur_type],
                        'blur_size': [blur_size]
                    }
                    save_current_stats(filename, stats_dict)


            elif blur_type == "Kernels":
                st.text("Size of kernel")
                height = st.slider("Height", 1, 500, value=2)
                width = st.slider("Width", 1, 500, value=10)
                create_echogram(data, cmap, vmin, vmax, blur_size=True, blur_type=blur_type, kernel_first=height, kernel_second=width)
                if st.button("Save current stats"):
                    stats_dict = {
                        'lower_limit': [lower_limit],
                        'upper_limit': [upper_limit],
                        'vmin': [vmin],
                        'vmax': [vmax],
                        'cmap': [cmap],
                        'blur_type': [blur_type],
                        'kernel_first': [height],
                        'kernel_second': [width]
                    }
                    save_current_stats(filename, stats_dict)

        else:
            create_echogram(data, cmap, vmin, vmax)
            if st.button("Save current stats"):
                stats_dict = {
                    'lower_limit': [lower_limit],
                    'upper_limit': [upper_limit],
                    'vmin': [vmin],
                    'vmax': [vmax],
                    'cmap': [cmap],
                    'blur': [False] 
                }
                save_current_stats(filename, stats_dict)


    with col3:
        st.image('file_original.png')

    with col4:
        st.image('file_greyscale.png')

    with col5:
        st.image('file.png')
