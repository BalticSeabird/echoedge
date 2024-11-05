import numpy as np
import echopype as ep
import pandas as pd
import glob
import os
import warnings
import tqdm
import sys
import datetime
from dateutil import tz


from functions import process_data, find_bottom, data_to_images, find_waves, find_fish_median2, medianfun2, find_layer, remove_vertical_lines, clean_times
from functions import download_posi, get_interpolated_gps, haversine, cut_zone, inter_positions_new, resize_image



warnings.filterwarnings("ignore")

# Read aarguments
#savefold = sys.argv[1]
#datfold = sys.argv[2]
#remove = sys.argv[3]


# PATH ##########################################################################################################################################################


root_path = "../Run_info" # PATH TO GO ONT HE FOLDER WITH THE INPUT/OUTPUT

savefold = f"{root_path}/Output/Corrected_img_wave_bottom"
datfold = f"{root_path}/Input"
remove = f"{root_path}/Output/Corrected_img_wave_bottom"

#################################################################################################################################################################

#filepath = '../../sailor_data/SURVEY2023/Test'

# Create savefold if not already existing
checkdir = os.path.exists(f"{savefold}")
if checkdir == False: 
    os.mkdir(f"{savefold}")

# Remove files from previous runs (optional)
if remove == "remove":
    files = glob.glob(f'{savefold}/*')
    for f in files:
        os.remove(f)

# Start where last loop stopped 
previous = glob.glob(f'{savefold}/*.csv')
donefiles = []
for file in previous: 
    donefiles.append(file.replace('.csv', '.raw').split("/")[4])


# Set environmental and calibration parameters
env_params = {
    'temperature' :20.338, # temperature in celsius
    'salinity' :34.27, # salinity from PPS78 (R gsw_SP_from_C)
    'pressure' :10.1325 # pressure in dBar
}
cal_params = {
    'gain_correction':28.49, # center value of gain (here 19th frequency: 202657)
    'equivalent_beam_angle' : -21.0 # ctrl+F in the calibration file
} 

######################################################################
##########       The interpolation of the GPS position      ##########
path_position = "Data_gps/"

files = os.listdir(path_position)
gps_files = [file for file in files if file.endswith("gps.csv")]
frequency_value = 2

mean_velocity = 0.63   # to decide the velocity here by histogram output

for gps_file in gps_files:
    file_path = os.path.join(path_position,gps_file)
    new_file_name = file_path.replace('_aussi_time.gps.csv', '_interpolated.csv')
    if os.path.exists(new_file_name):
        print(f'The file {new_file_name} exists.')
    else:
        interpolated_df = get_interpolated_gps(file_path,frequency=frequency_value)
        interpolated_df.to_csv(new_file_name)
######################################################################

# Create the velocity array
Velocity_dis = []

# Extract date from rawfiles
rawfiles = os.listdir(datfold)
for file in tqdm.tqdm(rawfiles): 
    # [a:n] to run on specific files in path
    if file not in donefiles:   
        try:   # Only process new files
            if '.raw' or '.RAW' in file: 
                new_file_name = file.replace(file[-4:], '')
                print(f"Working with {new_file_name}")
                        
                # Load and process data
                echodata, ping_times = process_data(f'{datfold}/{file}', env_params, cal_params)
                ping_times_series = pd.to_datetime(ping_times)
                to_zone =  tz.gettz('Australia/Melbourne')
                ping_times_melbourne = ping_times_series.tz_localize('UTC').tz_convert(to_zone)
                ping_times_melbourne_array = np.array(ping_times_melbourne) # in local time
                activedata = echodata.Sv.to_numpy()[0]
                echodata, nan_indicies = remove_vertical_lines(activedata)
                

                print("Data loaded successfully!")    

                echodata_swap = np.swapaxes(echodata, 0, 1) # turn the iage horizontally (to the left)

                data_to_images(echodata_swap, f'Output/Raw_img/{new_file_name}') # save img without ground

                # Detect bottom algorithms
                depth, hardness, depth_roughness, new_echodata = find_bottom(echodata_swap)
                #hardness = hardness[4:-4]

                # New find waves function
                new_echodatax = new_echodata.copy()
                layer = find_layer(new_echodatax)
                if layer:
                    new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodata, 68)
                else:
                    new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodata, 77)
                    #wave_dif = wave_avg / wave_smoothness

                    if wave_avg > 70 or (wave_avg > 34 and wave_smoothness < 8): 
                        new_echodata, wave_line, wave_avg, wave_smoothness = find_waves(new_echodatax, 65)

                #echodata_unswap = np.rot90(new_echodata, k=1, axes=(0,1))# back vertical (turn to the rigth)
                data_to_images(new_echodata, f'{savefold}/Img/{new_file_name}complete') # save img without ground

                # Find fish cumsum, median depth and inds
                depthx = [int(d) for d in depth]

                # wave_line = wave_line[4:-4]
                nasc = find_fish_median2(echodata, wave_line, depthx) 
                nasc0, fish_depth0 = medianfun2(nasc, 0, 100)
                nasc1, fish_depth1 = medianfun2(nasc, 0, 35)
                nasc2, fish_depth2 = medianfun2(nasc, 35, 80)
                nasc3, fish_depth3 = medianfun2(nasc, 80, 100)

                depth = [d*0.1 for d in depth] # convert in m
                wave_line = [w*0.1 for w in wave_line]# convert in m


                if nan_indicies.size != 0:
                    ping_times_melbourne_array = clean_times(ping_times_melbourne_array, nan_indicies)

                #ping_times_melbourne_array = ping_times_melbourne_array[4:-4]

                # Get the same time format as in positon file
                time = np.vectorize(cut_zone)(ping_times_melbourne_array)
                
                # Create the variable 'Longitude' and 'Latitude'
                Longitude = np.array([np.nan] *len(ping_times_melbourne_array))
                Latitude = np.array([np.nan] *len(ping_times_melbourne_array))
                Velocity = np.array([np.nan] *len(ping_times_melbourne_array))
                df_posi = {
                    
                    'Longitude' : Longitude,
                    'Latitude' : Latitude,
                    'time' : time,
                    'Velocity' : Velocity
                }
                df_posi = pd.DataFrame(df_posi)
            
                # Update by files of positon
                files_path = "Data_gps"
                files = os.listdir(files_path)
                interpolated_files = [fil for fil in files if fil.endswith("interpolated.csv")]
                for fil in interpolated_files:
                    fil = os.path.join(files_path,fil)
                    file_posi = pd.read_csv(fil)
                    download_posi(df_posi,file_posi)

                Longitude = df_posi['Longitude']
                Latitude = df_posi['Latitude']
                Velocity = df_posi['Velocity']

                # Calculate the average velocity
                Velocity_m = np.round(df_posi['Velocity'].mean(),3)
                #Velocity_dis.append(Velocity)

                # Calculate the distance between two points on the Earth's surface using the Haversine formula
                Longitude_0 = [Longitude[0]] + [Longitude[i-1] for i in range(1,len(Longitude))]
                Latitude_0 = [Latitude[0]] + [Latitude[i-1] for i in range(1,len(Latitude))]


                # Save results to df and export to csv
                data_dict = {
                    'time': ping_times_melbourne_array,
                    'Longitude' : Longitude,
                    'Latitude' : Latitude,
                    'Velocity' : Velocity,
                    'bottom_depth': depth,
                    'bottom_hardness': hardness,
                    'bottom_roughness': depth_roughness,
                    'wave_depth': wave_line,
                    'nasc0': nasc0,
                    'fish_depth0': fish_depth0, 
                    'nasc1': nasc1,
                    'fish_depth1': fish_depth1, 
                    'nasc2': nasc2,
                    'fish_depth2': fish_depth2, 
                    'nasc3': nasc3,
                    'fish_depth3': fish_depth3, 
                }

                df = pd.DataFrame(data_dict)

                cwd = os.getcwd()

                df.to_csv(f'{savefold}/Csv/{new_file_name}.csv')
                #os.remove(os.path.join(cwd, f'../../runs/{savefold}/{new_file_name}complete_greyscale.png'))
                #os.remove(os.path.join(cwd, f'../../runs/{savefold}/{new_file_name}_greyscale.png'))
        except:
            print(f'Problems with {new_file_name}...')

        #Velocity_dis = np.array(Velocity_dis).flatten()

        #np.savetxt(f'../../runs/{savefold}/{save_velocity}.csv', Velocity_dis, delimiter=',')    
