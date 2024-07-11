import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d
from geopy.distance import geodesic
from geopy import Point

# Functions

# For interpolations of data series in data frames
def approx1(x, n):
    nan_indices = np.isnan(x)
    idx = np.arange(len(x))

    # Kopiera x till en temporär array för att undvika ändringar i ursprungliga x
    x_interp = x.copy()

    # Iterera över index där x[i] är NaN
    for i in tqdm(idx[nan_indices]):
        left_bound = max(0, i - 1)
        right_bound = min(len(x), i + n)

        yt0 = x_interp[left_bound:right_bound]

        if len(yt0) < n + 1:
            yt0 = np.append(yt0, [np.nan] * (n + 1 - len(yt0)))

        # Skapa en interpolationsfunktion för det aktuella segmentet
        xt1 = np.arange(len(yt0))
        f = interp1d(xt1, yt0, kind='linear', bounds_error=False, fill_value='extrapolate')

        # Interpolera yt2 för hela segmentet
        xt2 = np.arange(len(yt0))
        yt2 = f(xt2)

        # Ersätt NaN-värden i det ursprungliga x-arrayet med de interpolerade värdena
        replace_length = right_bound - left_bound

        # Om yt2 är längre än replace_length, trimma det till rätt längd
        if len(yt2) > replace_length:
            yt2 = yt2[:replace_length]

        # Om yt2 är kortare än replace_length, lägg till NaN-värden i slutet
        elif len(yt2) < replace_length:
            yt2 = np.append(yt2, [np.nan] * (replace_length - len(yt2)))

        # Tilldela yt2 till x_interp inom det angivna intervallet
        x_interp[left_bound:right_bound] = yt2

    return x_interp

# Fill last number with NA
def na_lomf(x):
    def na_lomf_0(x):
        non_na_idx = np.where(~np.isnan(x))[0]
        if np.isnan(x[0]):
            non_na_idx = np.insert(non_na_idx, 0, 0)
        return np.repeat(x[non_na_idx], np.diff(np.append(non_na_idx, len(x))))
    
    if x.ndim == 1:
        return na_lomf_0(x)
    else:
        return np.apply_along_axis(na_lomf_0, x.ndim - 1, x)

# Read data
# Autopilot
data = pd.read_csv("coords_data/SB2017A-2.txt", delimiter="\t")
data['DateTime'] = pd.to_datetime(data['Time'])

# Data logger
datd = pd.read_csv("coords_data/SB2017D.txt", delimiter="\t")
datd['DateTime'] = pd.to_datetime(datd['Time'])

# Calculate data
# Define survey
data['Survey'] = "Coats-2024"
datd['Survey'] = "Coats-2024"

# For SBA
data.loc[data['DateTime'] > "2024-06-25 00:00", 'Survey'] = "Coats-2024"

# For SBD
datd.loc[datd['DateTime'] > "2024-06-25 00:00", 'Survey'] = "Coats-2024"

# Filter data
d1 = data[data['Lat'] > 60].dropna(subset=['Survey'])
d2 = datd[datd['Lat'] > 60].dropna(subset=['Survey'])

# Algorithm for making reasonable turns (for interpolation of GPS positions)
surveys = d1['Survey'].dropna().unique()
saildata = pd.DataFrame()

for survey in surveys:
    postest = d1[d1['Survey'] == survey].iloc[1:]  # skip the first row
    postest['tacknum'] = np.arange(1, len(postest) + 1)
    
    # Merge with positions from data logger
    postestx = pd.concat([postest, d2[d2['Survey'] == survey]], ignore_index=True)
    
    # Sort combined data frame
    postestx = postestx.sort_values(by='DateTime')
    
    # Remove duplicates in time (remove data logger data)
    postestx = postestx.drop_duplicates(subset='DateTime')
    
    # Remove first uneven number
    postestx = postestx.iloc[1:]
    
    # Linear interpolation of positions
    start = postestx.iloc[0]['DateTime']
    stop = postestx.iloc[-1]['DateTime']
    timeseq = pd.date_range(start, stop, freq='30s')
    dx = pd.DataFrame({'DateTime': timeseq})
    postest2 = pd.merge(postestx, dx, on='DateTime', how='right')
    
    postest2['LatLinear'] = approx1(postest2['Lat'].values, 200)
    postest2['LongLinear'] = approx1(postest2['Long'].values, 200)
    
    # Fill tack information, tack number etc.
    postest2['tacknum'] = na_lomf(postest2['tacknum'].values)
    
    # Pick out sailing data
    posail = postest2[['Lat', 'Long', 'LongLinear', 'LatLinear', 'Heading', 'WindDirection', 'CurrentTack', 'tacknum', 'Velocity', 'WBT_Status', 'DateTime']].copy()
    posail['DirComp'] = posail['Heading'] - posail['WindDirection']
    posail['DirComp'] = np.where(posail['DirComp'] < 0, posail['DirComp'] + 360, posail['DirComp'])
    posail[['LongTack', 'LatTack']] = np.nan
    posail = posail.dropna(subset=['tacknum'])
    
    # Creating a simulated sailing route
    tackspeed = 30
    max_i = int(posail['tacknum'].max()) - 1
    
    # Loop through all tacks
    for i in tqdm(range(2, max_i + 1)):
        print(f"Started processing tack {i} / {max_i} for survey {survey}")
        starti = i
        stopi = i + 1
        startdat = posail[posail['tacknum'] == starti].iloc[0]
        stopdat = posail[posail['tacknum'] == stopi].iloc[0]
        
        startrow = startdat.name
        stoprow = stopdat.name
        
        dt1 = posail.loc[startrow:stoprow].copy()
        
        dt1[['LongTack', 'LatTack']] = dt1[['Long', 'Lat']]
        
        if startdat['CurrentTack'] != stopdat['CurrentTack'] and abs(startdat['DirComp'] - stopdat['DirComp']) > tackspeed:
            dir = tackspeed if startdat['DirComp'] < stopdat['DirComp'] else -tackspeed
            tackseqComp = np.arange(startdat['DirComp'], stopdat['DirComp'], dir)
            
            if len(tackseqComp) > 1 and len(dt1) >= len(tackseqComp):
                tackseq = tackseqComp + startdat['WindDirection']
                tackseq = np.where(tackseq > 360, tackseq - 360, tackseq)
                dt1.loc[dt1.index[:len(tackseq)], 'Heading'] = tackseq
                
                for j in range(len(tackseq) - 1):
                    if np.isnan(dt1.iloc[j]['LatTack']) or np.isnan(dt1.iloc[j]['LongTack']):
                        # Skip calculations if NaN values are encountered
                        continue
                    else:
                        point = Point(dt1.iloc[j]['LatTack'], dt1.iloc[j]['LongTack'])
                        distance = dt1.iloc[j]['Velocity'] * tackspeed
                        new_point = geodesic(meters=distance).destination(point, dt1.iloc[j + 1]['Heading'])
                        
                        # Assign new values to LatTack and LongTack
                        dt1.at[dt1.index[j + 1], 'LatTack'] = new_point.latitude
                        dt1.at[dt1.index[j + 1], 'LongTack'] = new_point.longitude
        dt1[['LongTack', 'LatTack']] = dt1[['LongTack', 'LatTack']].apply(approx1, args=(200,), axis=0)
        
        out1 = pd.DataFrame({
            'GPS_date': dt1['DateTime'].dt.date,
            'GPS_time': dt1['DateTime'].dt.time,
            'Survey': survey,
            'Longitude': dt1['LongTack'],
            'Latitude': dt1['LatTack']
        })
        
        saildata = pd.concat([saildata, out1], ignore_index=True)
        print(f"This survey finished processing in approx. {(max_i - i)} minutes")

# Export file for Echoview
saildata.to_csv("coords_data/coats2024-gps.csv", index=False)
