import xarray as xr
import echopype as ep
import numpy as np
import warnings
import pandas as pd 
import datetime
from dateutil import tz

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

    if '.raw' or '.RAW' in path:
        raw_echodata = ep.open_raw(path, sonar_model="EK80")
    
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

def get_interpolated_gps(path, frequency=1):
    """
    Interpolates coordinates

    Args: 
        path (str): The path to gps.csv file  
        frequency (int) : The frequency of interpolation in seconds

    Returns: 
        pd.DataFrame: A DataFrame with interpolated coordinates and time
    """
    
    print('Loading and interpolating coordinates...')
    df = pd.read_csv(path)
    df['Time'] = df['GPS_date'] + ' ' + df['GPS_time']

    # drop rows with undesirable years
    df['GPS_date'] = df['GPS_date'].astype('datetime64[ns]')
    df['Year'] = df['GPS_date'].dt.year
    df = df[df.Year == 2023]
    df.reset_index(inplace=True)
    df = df.drop(columns=['GPS_date', 'GPS_time', 'Year', 'index', 'Survey'])
    df = df[~df.Time.duplicated()]
    print(df)

    # convert the times column
    df['Time'] = pd.to_datetime(df['Time'])
    df['Time'] = pd.to_datetime((df['Time'].astype(np.int64)).astype('datetime64[ns]'))

    # new range (once a second), resample and interpolate
    new_range = pd.date_range(df.Time[0], df.Time.values[-1], freq=str(frequency)+'S')
    interpolated_df = df.set_index('Time').reindex(new_range).interpolate().reset_index()
    interpolated_df.rename(columns= {'index' : 'Datetime'}, inplace=True)
    interpolated_df['Longitude'] = interpolated_df['Longitude'].apply(lambda x: round(x, 5))
    interpolated_df['Latitude'] = interpolated_df['Latitude'].apply(lambda x: round(x, 5))

    print('Coords loaded successfully!')

    return interpolated_df

def get_interpolated_gps2(path, ltz, frequency=2):
    """
    Interpolates coordinates
    Jiao version
    Args: 
        path (str): The path to gps.csv file  
        frequency (int) : The frequency of interpolation in seconds
        local time zone (ltz) : time zone of the sampling site
    Returns: 
        pd.DataFrame: A DataFrame with interpolated coordinates and time
    """
    
    df = pd.read_csv(path)
    
    print('Loading and interpolating coordinates...')
    df['Datetime_UTC'] = pd.to_datetime(df['Datetime_UTC'])

    # drop rows with repeated time
    df = df.drop(columns=['Datetime_local'])
    df = df[~df.Datetime_UTC.duplicated()]

    # new range (2 seconds), resample and interpolate
    new_range = pd.date_range(df.Datetime_UTC[0], df.Datetime_UTC.values[-1], freq=str(frequency)+'S')
    print("new range:",df.Datetime_UTC[0])
    print("new range:",df.Datetime_UTC.values[-1])
    interpolated_df = df.set_index('Datetime_UTC').reindex(new_range).interpolate().reset_index()

    interpolated_df.rename(columns= {'index' : 'Datetime_UTC'}, inplace=True)
    interpolated_df['Longitude'] = interpolated_df['Longitude'].apply(lambda x: round(x, 5))
    interpolated_df['Latitude'] = interpolated_df['Latitude'].apply(lambda x: round(x, 5))

    # Get the velocity values corresponding to the continuous time interval ]T1,T2]
    present_df = interpolated_df[interpolated_df['Datetime_UTC'].isin(df['Datetime_UTC'])]
    result_df = pd.merge_asof(interpolated_df, present_df, on='Datetime_UTC', direction='forward')
    result_df = result_df[['Datetime_UTC', 'Longitude_x','Latitude_x', 'Velocity_y']]
    result_df.rename(columns= {'Velocity_y' : 'Velocity','Longitude_x' : 'Longitude','Latitude_x' : 'Latitude'}, inplace=True)

    #Add one column with the local time 
    Datetime = pd.DatetimeIndex(pd.to_datetime(result_df['Datetime_UTC']))
    to_zone =  tz.gettz(ltz)
    Datetime_local = Datetime.tz_localize('UTC').tz_convert(to_zone)
    Datetime_local = pd.DataFrame(Datetime_local)
    Datetime_local.rename(columns = {'Datetime_UTC' : 'Datetime_local'} , inplace = True)
    result_df = pd.concat([result_df, Datetime_local], axis =1)
    
    print('Coords loaded successfully!')

    return result_df


def download_posi(df1,df2):
    """
    Args : 
        df1 : the output dataframe without the postion infos of longitude and latitude
        df2 : the csv file contains the position infos of longitude and latitude
    Returns:
        df1 after having position infos added
    """
    df2['time'] = pd.to_datetime(df2['time'])   # importante change dtype

    # Merge DataFrames on the 'time' column
    merged_df = pd.merge(df1, df2[['time', 'Longitude', 'Latitude','Velocity']], on='time', how='left')

    # Update the 'Longitude' and 'Latitude' columns in df1
    merged_df['Latitude'] = np.where(pd.isna(merged_df['Latitude_x']), merged_df['Latitude_y'], merged_df['Latitude_x'])
    merged_df['Longitude'] = np.where(pd.isna(merged_df['Longitude_x']), merged_df['Longitude_y'], merged_df['Longitude_x'])
    merged_df['Velocity'] = np.where(pd.isna(merged_df['Velocity_x']), merged_df['Velocity_y'], merged_df['Velocity_x'])
    df1.loc[:, 'Longitude'] = merged_df['Longitude']
    df1.loc[:,'Latitude'] = merged_df['Latitude']
    df1.loc[:,'Velocity'] = merged_df['Velocity']


def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # 6371.0 Radius of the Earth in kilometers (you can adjust this value)
    distance = 6371.0 * c  

    return distance


def cut_zone(time):
    """
    Args time: 
        a str of a date time consisted by 'the time part'  +  '+' + additional part
        
    Returns:
        cutoff the "+11:00" in time formate.

    """
    # split the time part + additional part
    time_str = str(time)
    parts = time_str.split('+')
    [d_time,d_plus] = parts
    d_utc = pd.to_datetime(d_time)
    
    return pd.to_datetime(d_utc)
