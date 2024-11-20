from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import sys
import pickle


from Double_thresh_function import resize_matrix_31052024


#################GET THE MEAN VELOCITY######################### 
###############################################################
# remove the empty speed file and the 0 speed files
###############################################################

# PATH ##############################################################

csv_path = "D:/SURVEY2023/PREPROCESS_DATA/Csv"
npi_path = "D:/SURVEY2023/PREPROCESS_DATA/npy"
dest_path = "D:/SURVEY2023/PREPROCESS_DATA/Resize_Img"

###############################################################
# Check if the directory exists
if not os.path.exists(dest_path):
    # Create the directory if it doesn't exist
    os.makedirs(dest_path)
    
files = os.listdir(csv_path)
csv_files = [file for file in files if file.endswith(".csv")]
Velocity_all = pd.Series()

total_files = len(csv_files)
progress_bar = tqdm(total=total_files, desc="Processing Files", unit="file")
count = 0 
for i, csv_file in enumerate(csv_files, start=1):
    csv_file_path = os.path.join(csv_path, csv_file)
    csv_file_info = pd.read_csv(csv_file_path)
    Velocity_info = pd.Series(csv_file_info['Velocity'])
    
    if np.all(np.logical_and(Velocity_info != 0, ~np.isnan(Velocity_info))):
        Velocity_all = pd.concat([Velocity_all, Velocity_info])
        count = count + 1

print("the size of velocity is:", len(Velocity_all))
print("mean velocity is :", Velocity_all.mean())
print('total files :',count)


# Create a histogram
print("mean velocity is :", np.round(Velocity_all.mean(),3))
plt.hist(Velocity_all, bins=60, density=True, alpha=0.7, color='blue', edgecolor='black')

# Add labels and title
mean_velocity = Velocity_all.mean()
plt.axvline(mean_velocity, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_velocity:.2f} m/s')
plt.xlabel('Velocity (m/s)')
plt.ylabel('Probability Density')
plt.title('Histogram of All Velocities')
# Show the plot
plt.show()


############################################################################
######### not conclude the file with missing values and speed == 0 #########
############################################################################
import warnings
warnings.filterwarnings("ignore")

mean_speed = np.round(Velocity_all.mean(),3)

#mean_speed = 0.648 in 2023 breeding season
print("mean speed is :",mean_speed)

# create a mapping table to save the mapping info
mapping_info = {}
for i, csv_file in enumerate(tqdm(csv_files, desc="Processing matrix"), start=1):
    
    # test one file
    # if "D20230927-T000243-0" in csv_file: 
    mapping = []
    csv_file_path = os.path.join(csv_path, csv_file)
    csv_file_info = pd.read_csv(csv_file_path)
    
    Velocity_info = np.array(csv_file_info['Velocity'])
    # depth = csv_file_info['bottom_depth']
    
    matrix_file = csv_file.replace('.csv','.npy')
    matrix_file_path = os.path.join(npi_path,matrix_file)
    
    if np.all(np.logical_and(Velocity_info != 0, ~np.isnan(Velocity_info))):
        img = np.load(matrix_file_path)
        img_resized, mapping = resize_matrix_31052024(img, velocity = Velocity_info, meanvelocity = mean_speed)
        resize_file = matrix_file.replace('.npy','_new.npy')
        
        # if need to save resized image, can use next line
        np.save(f'{dest_path}/{resize_file}',img_resized)   
        
        mapping_info[csv_file] = mapping

mapping_info_path = os.path.join(dest_path, 'mapping_info.pkl')
with open(mapping_info_path, 'wb') as f:
    pickle.dump(mapping_info, f) 