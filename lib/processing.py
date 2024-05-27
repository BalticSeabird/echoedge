import xarray as xr
import echopype as ep
import numpy as np
import warnings


from pathlib import Path



warnings.filterwarnings("ignore")

# Process data 
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


def extract_meta_data(path):

    if '.raw' in path:
        raw_echodata = ep.open_raw(path, sonar_model="EK80", use_swap=True)

    else:
        raw_data_path = Path.cwd().joinpath(path)
        for fp in raw_data_path.iterdir():
            if fp.suffix == ".raw":
                raw_echodata = ep.open_raw(fp, sonar_model='EK80')
                break
            else:
                print("File not working, please provide a .raw file.")
    
    channels = raw_echodata.platform.channel.to_numpy()
    longitude = raw_echodata.platform.longitude.to_numpy()
    latitude = raw_echodata.platform.latitude.to_numpy()
    transmit_type = raw_echodata.beam.transmit_type.to_numpy()

    return raw_echodata, channels, longitude, latitude, transmit_type



def process_data(path, env_params, cal_params, bin_size, waveform, ping_time_bin='2S'):
    """
    Env_params : dictionary with water temperature in degree C, salinity, pressure in dBar
    Cal_params : dictionary with gain correction (middle value with 0.6.4 version), equivalent beam angle
    Function to load raw data to ep format, calibrate ep data, 
    compute MVBS (mean volume backscattering strength) and
    run swap_chan. Returns NetCDF object (xarray.core.dataset.Dataset). 
    """

    raw_echodata = ep.open_raw(path, sonar_model="EK80", use_swap=True)

    ds_Sv_raw = ep.calibrate.compute_Sv(
        raw_echodata,
        env_params = env_params,
        cal_params = cal_params,
        waveform_mode=waveform,
        encode_mode="complex",
    )

    ds_MVBS = ep.commongrid.compute_MVBS(
        ds_Sv_raw, # calibrated Sv dataset
        range_meter_bin=bin_size, # bin size to average along range in meters
        ping_time_bin=ping_time_bin # bin size to average along ping_time in seconds
    )

    ds_MVBS = ds_MVBS.pipe(swap_chan)
    ping_times = ds_MVBS.Sv.ping_time.values
    return ds_MVBS, ping_times


def clean_times(ping_times, nan_indicies):
    mask = np.ones(ping_times.shape, dtype=bool)
    mask[nan_indicies] = False

    # Använd masken för att ta bort värden från ping_times
    ping_times = ping_times[mask]

    return ping_times


def remove_vertical_lines(echodata):
    # Hitta indexen för arrayerna som bara innehåller NaN
    nan_indices = np.isnan(echodata).all(axis=1)
    indices_to_remove = np.where(nan_indices)[0]

    # Ta bort arrayerna med NaN-värden från din ursprungliga array
    echodata = echodata[~nan_indices]

    return echodata, indices_to_remove