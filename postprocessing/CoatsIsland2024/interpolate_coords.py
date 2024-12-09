import pandas as pd
import numpy as np

def load_coordinates(path, frequency=1):
    """
    Function to load and interpolate coordinates
    Note that the output frequency after the interpolation will be 1S
    no matter of input frequencey, to change this pass frequency argument (int).
    """
    
    print('Loading and interpolating coordinates...')
    # Read tab separated file

    df = pd.read_csv(path, sep ='\t')
    #df.head()

    # convert the times column
    df['Time'] = pd.to_datetime(df['Time'])
    df['Time'] = pd.to_datetime((df['Time'].astype(np.int64)).astype('datetime64[ns]'))

    # new range (once a second), resample and interpolate
    new_range = pd.date_range(df.Time[0], df.Time.values[-1], freq=str(frequency)+'s')
    interpolated_df = df.set_index('Time').reindex(new_range).interpolate().reset_index()
    interpolated_df.rename(columns= {'index' : 'Datetime'}, inplace=True)
    interpolated_df['Long'] = interpolated_df['Long'].apply(lambda x: round(x, 7))
    interpolated_df['Lat'] = interpolated_df['Lat'].apply(lambda x: round(x, 7))

    print('Coords loaded successfully!')

    return interpolated_df


if __name__ == "__main__":
    path = "coords_data/SB2017A.txt"
    coords = load_coordinates(path)
    print("Done!")

    coords.to_csv("coords_data/interpolated_coords_coats24.csv", index=False)

