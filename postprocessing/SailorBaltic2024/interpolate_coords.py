import pandas as pd
import numpy as np

def load_coordinates(path, frequency=1):
    """
    Function to load and interpolate coordinates
    Note that the output frequency after the interpolation will be 1S
    no matter of input frequencey, to change this pass frequency argument (int).
    """
    
    print('Loading and interpolating coordinates...')
    df = pd.read_csv(path)

    # convert the times column
    df['Time'] = pd.to_datetime(df['GPS_date']+df["GPS_time"], format = "%Y-%m-%d%H:%M:%S")
    df['Time'] = pd.to_datetime((df['Time'].astype(np.int64)).astype('datetime64[ns]'))
    df = df.set_index(["Time"])
    df_unique = df[~df.index.duplicated(keep='first')]

    # new range (once a second), resample and interpolate
    new_range = pd.date_range(df_unique.index[0], df.index.values[-1], freq=str(frequency)+'s')
    
    interpolated_df = df_unique.reindex(new_range).interpolate().reset_index()
    interpolated_df.rename(columns= {'index' : 'Datetime'}, inplace=True)
    interpolated_df['Longitude'] = interpolated_df['Longitude'].apply(lambda x: round(x, 7))
    interpolated_df['Latitude'] = interpolated_df['Latitude'].apply(lambda x: round(x, 7))

    print('Coords loaded successfully!')

    return interpolated_df


if __name__ == "__main__":
    path = "data/SailorKarlsoPositionsEchoview2020-2023.gps.csv"
    coords = load_coordinates(path)
    print("Done!")

    coords.to_csv("postprocessing/SailorBaltic2024/interpolated_coords20-23.csv", index=False)

