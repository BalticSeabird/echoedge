import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,reconstruction,closing, square,binary_erosion,rectangle
import numpy.ma as ma
import skimage
from skimage.measure import label, regionprops
from skimage.color import label2rgb
import math
import pandas as pd
import shutil
from tqdm import tqdm
import sys
import os
import pickle


from created_function import parameters_correction,find_original_position,find_center_square,find_edges, npy_correction_v3


#######################################################################################################################################################
################### STEP 1 : Define a function to obtain schools description ##########################################################################
#######################################################################################################################################################

##############################################################################################
##############                       version time  20240806                     ##############
##############################################################################################
def sam_info(file,sam_result,npy_path,csv_path,dest_path):    # version time : 20240806
    # Step 1 : change the file name to get the corresponding name in sam result
    file_npy =file[:-4]+'.npy'

    # Step 2 : load the image from original directionary + mask
    img = np.load(os.path.join(npy_path,file_npy))
    csv_name = file.replace('complete_new.png','.csv')
    csv_table = pd.read_csv(os.path.join(csv_path,csv_name))
    bottom = int(csv_table['bottom_depth'].median()*10)
   
    length_limit = bottom *0.62
    size_limit = bottom*img.shape[1]*0.5            #  2 : bigger than before = 0.42
    nomal_limit = bottom*img.shape[1]*0.1

    # Step 3 : find the corrections parameters
    shape_top,shape_bottom,first_lines,shape_top_desc,shape_bottom_desc = parameters_correction(img,bottom,threshold = -30)
   
    # Step 4 : get the mask 

    im = npy_correction_v3(img,first_lines,shape_top,shape_bottom)           # 1 : new mask
    mask = im.mask
    
    dis_limit = 10
    dis_check = 50 # width limit
    width_image = img.shape[1]
    height_image = img.shape[0]

    # Step 5 : Initialising the output variables
    label_file = []     # class 1
    bbox_file = []       
    width_length_file = []  
    axis_ellipse_file = []  #  major , minor
    perimeter_file = []
    size_file = []      
    is_very_wide_s = []
    is_very_tall_s = [] 
    dis_level_s = []  
    dis_to_surface_s = []  
    dis_to_bottom_s = []   

    intensity_school = []    # class 2 
    intensity_img = [np.mean(img[img!=0]),np.min(img),np.max(img[img!=0])]
    intensity_c_img_mask = np.mean(im[~mask])                                       # 2: use corrected img intensity

    dif_intensity_s_i = []
    center_square_intensity_file = []    
    edges_school_intensity_file = [] 
    dif_intensity_center_edges_file  = []
    std_intensity_school_file  = []
    gradient_school_file = []    
    gradient_school_center_file = []
    gradient_school_edges_file = []
    dif_gradient_center_edges_file = []
    
    width_length_ratio_file = []    # class 3   elongation 
    axis_ellipse_ratio_file = []  #  short / long   eccentricity 
    solidity_file = []
    compactness_file = []
    inertia_tensor_eigvals_ratio_file = []    
    perimeter_area_ratio_file = []

    centroid_file = []    # file class 4
    depth_file = []     
    coords_file = []  

    edges_school_file = []  # file class 5
    center_square_file = []

    # Step 6 : Read the info from the sam pkl file
    seg_new = sam_result[file]
    segmentations = seg_new['segmentation']
    bboxs = seg_new['bbox']
    areas = seg_new['area']
    if len(segmentations)>0 :
        binary_matrice = np.zeros((np.array(segmentations[0])).shape)

        # Step 5 : Get the union of segmentations of fish school
        for j in range(len(segmentations)):
            seg = segmentations[j]
            intensity_c_seg = np.mean(im[seg])                                      # 2: use corrected seg intensity

            bbox = bboxs[j]
            # Check if there are any common positions
            common_positions = np.logical_and(mask, seg)
            intersections_mask = np.any(common_positions)

            # Criterion 1
            if (not intersections_mask) and (intensity_c_seg>intensity_c_img_mask) and (areas[j]<size_limit) and (bbox[3]<length_limit):   # 2 : big or small both use the same intensity limit
                    binary_matrice[seg] = 1
                                  
        label_img = label(binary_matrice)
        regions = regionprops(label_img)
        
        # Step 6 : Get the informations from the fish school detected
        if len(regions) > 0 : 
            plt.imsave(f'{dest_path}\{file[:-4]}.png', binary_matrice, cmap='gray')
            for region in  regions:
                if region.area > 1 :
                    coords = region.coords
                    y_coords = coords[:, 0]
                    x_coords = coords[:, 1]
                    bbox = region.bbox
                    dis_to_bottom = bottom - bbox[1]-bbox[3]/2
                    dis_to_surface = bbox[1]
                    is_very_wide = 1 if (bbox[3]-bbox[1])>width_image/2 else 0
                    is_very_tall = 1 if (bbox[2]-bbox[0])>height_image/2 else 0
                    
                    if dis_to_surface > dis_check and dis_to_bottom > dis_check:
                        dis_level = 2
                    elif dis_to_bottom < dis_limit or dis_to_surface <dis_limit:
                        dis_level = 0
                    else:
                        dis_level = 1
                    # Calculate the average coordinates
                    avg_0 = sum(coord[0] for coord in coords) / len(coords)
                    avg_1 = sum(coord[1] for coord in coords) / len(coords)

                    intensities = [img[coord[0], coord[1]] for coord in coords]
                    max_index = np.argmax(intensities)
                    max_intensity_coords = coords[max_index]
                
                    edges_school = find_edges(binary_matrice, y_coords, x_coords)
                    center_square = find_center_square(max_intensity_coords, region.coords)  # use max_intensity_coords not center
                    center_square_intensity =  np.mean([img[coord[0], coord[1]] for coord in center_square])
                    edges_school_intensity =  np.mean([img[coord[0], coord[1]] for coord in edges_school])
                    dif_intensity_center_edges = center_square_intensity - edges_school_intensity
                    std_intensity_school = np.std([img[coord[0], coord[1]] for coord in coords])
                
                    gradient_y, gradient_x = np.gradient(coords)
                    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                    gradient_school = np.mean(gradient_magnitude)
                    gradient_y_center, gradient_x_center = np.gradient(center_square)
                    gradient_magnitude_center = np.sqrt(gradient_x_center**2 + gradient_y_center**2)
                    gradient_school_center = np.mean(gradient_magnitude_center)
                    gradient_y_edges, gradient_x_edges = np.gradient(edges_school)
                    gradient_magnitude_edges = np.sqrt(gradient_x_edges**2 + gradient_y_edges**2)
                    gradient_school_edges = np.mean(gradient_magnitude_edges)
                    dif_gradient_center_edges = abs(gradient_school_center - gradient_school_edges)
        
                    centroid_file.append(region.centroid)
                    label_file.append(region.label)
                    bbox_file.append(region.bbox)
                    width_length_file.append([region.bbox[3]-region.bbox[1],region.bbox[2]-region.bbox[0]])
                    axis_ellipse_file.append([region.axis_major_length,region.axis_minor_length])
                    perimeter_file.append(region.perimeter)
                    size_file.append(region.area)
                    is_very_wide_s.append(is_very_wide)
                    is_very_tall_s.append(is_very_tall)
                    dis_level_s.append(dis_level)
                    dis_to_surface_s.append(dis_to_surface)
                    dis_to_bottom_s.append(dis_to_bottom)

                    intensity_school.append([np.mean(img[y_coords, x_coords]),np.min(img[y_coords, x_coords]),np.max(img[y_coords, x_coords])])
                    dif_intensity_s_i.append(np.mean(intensities) - intensity_img[0])
                    
                    center_square_intensity_file.append(center_square_intensity)
                    edges_school_intensity_file.append(edges_school_intensity)
                    dif_intensity_center_edges_file.append(dif_intensity_center_edges)
                    std_intensity_school_file.append(std_intensity_school)
                    gradient_school_file.append(gradient_school)    
                    gradient_school_center_file.append(gradient_school_center)
                    gradient_school_edges_file.append(gradient_school_edges)
                    dif_gradient_center_edges_file.append(dif_gradient_center_edges)

                    width_length_ratio_file.append((region.bbox[3]-region.bbox[1])/(region.bbox[2]-region.bbox[0]))
                    axis_ellipse_ratio_file.append(region.axis_minor_length/region.axis_major_length)
                    solidity_file.append(region.solidity)
                    compactness_file.append(4*math.pi*(region.area)/(region.perimeter**2))
                    inertia_tensor_eigvals_ratio_file.append(region.inertia_tensor_eigvals[1]/region.inertia_tensor_eigvals[0])
                    perimeter_area_ratio_file.append(region.perimeter/region.area)
            
                    coords_file.append(region.coords)
                    depth_file.append(avg_0/10)
                    edges_school_file.append(edges_school)
                    center_square_file.append(center_square)

            y_all = np.dot(size_file, [coord[0] for coord in centroid_file]) / np.sum(size_file)
            x_all = np.dot(size_file, [coord[1] for coord in centroid_file]) / np.sum(size_file)
            intensity_school_array = np.vstack(intensity_school)
            mean_intensity_school = np.dot(size_file, intensity_school_array[:,0])/np.sum(size_file)
            
            file_info = {"label":label_file,
                        "size":size_file,
                        "bbox":bbox_file,
                        "depth":depth_file,
                        "center":centroid_file,
                        "width_length":width_length_file,
                        "is_very_wide":is_very_wide_s,
                        "is_very_tall":is_very_tall_s,
                        'dis_to_surface':dis_to_surface_s,
                        'dis_to_bottom':dis_to_bottom_s,
                        'dis_level':dis_level_s,
                        "axis_ellipse": axis_ellipse_file,
                        "perimeter_school": perimeter_file,
                        "intensity_school":intensity_school,
                        "dif_intensity_school_image": dif_intensity_s_i,
                        "intensity_img":intensity_img,
                        "center_square_intensity":center_square_intensity_file, 
                        "edges_school_intensity":edges_school_intensity_file,
                        "dif_intensity_center_edges":dif_intensity_center_edges_file,
                        "std_intensity_school":std_intensity_school_file,
                        "gradient_school":gradient_school_file,
                        "gradient_school_center":gradient_school_center_file,
                        "gradient_school_edges":gradient_school_edges_file,
                        "dif_gradient_center_edges":dif_gradient_center_edges_file,
                        "width_length_ratio":width_length_ratio_file,
                        "axis_ellipse_ratio":axis_ellipse_ratio_file,
                        "solidity":solidity_file,
                        "compactness":compactness_file,
                        "inertia_tensor_eigvals_ratio":inertia_tensor_eigvals_ratio_file,
                        "perimeter_area_ratio":perimeter_area_ratio_file,
                        "coords":coords_file,
                        "nbr_school":len(label_file),
                        "total_area":sum(size_file),
                        "mean_intensity_school":mean_intensity_school,
                        "center_all":[y_all,x_all],     #[row,column]
                        "mean_depth":y_all/10}
            new_row_table = {'file':file, 
                        'nbr_school':len(label_file), 
                        'all_size':sum(size_file),
                        'mean_intensity_school':mean_intensity_school,
                        'mean_depth':y_all/10,
                        'mean_intensity_imgwithout0':np.mean(img[img!=0])} 

            return file_info ,new_row_table,centroid_file
        else:
            return None, None, None
    else:
        return None, None, None

#######################################################################################################################################################
####################################### STEP 2 : Generate images information ##########################################################################
#######################################################################################################################################################

##############################################################################################
##############                       version time  20240806                     ##############
##############################################################################################

# PATH ################################################################################################################################################

root_path = "../Run_info" 

dest_path = f"{root_path}/Output/SAM/Segment_anything"    # path to save the choosed informations from the sam segmentations
npy_path = f"{root_path}/Output/Resize_img/08_08_2024"            #  Path where the resized .npy files are stored
csv_path = f"{root_path}/Output/Corrected_img_wave_bottom/Csv/09_08_2024"                #  Path where the original .csv files are stored
img_path = f"{root_path}/Output/Mask_img/SAM"       #  Path where the resized and corrected images are stored
mapping_path = os.path.join(npy_path,"mapping_info.pkl")      #  the mapping_info.pkl file is stored in the same path as the resized npy files
sam_result_path = f"{root_path}/Output/SAM/Segment_anything/03_09_2024_5607/sam_results.pkl"    #  Path where the segmentations by sam is stored

#######################################################################################################################################################

if not os.path.exists(dest_path):
    os.makedirs(dest_path)

with open(mapping_path, 'rb') as f:
    loaded_mapping_info = pickle.load(f)

# Load the results from the SAM
with open(sam_result_path, 'rb') as f:
    loaded_segment_sam_results = pickle.load(f)

img_all = os.listdir(img_path)   
img_all_png = [image for image in img_all if image.endswith('.png')]
# Step 2:  Initialize the variables
segment_sam_info = {}
segment_sam_table = pd.DataFrame(columns=['file', 'nbr_school','all_size','mean_intensity_school','mean_depth','mean_intensity_imgwithout0'])

# Step 3 : segment_sam_infoSAM methode
for file in tqdm(img_all_png, desc="Processing matrix"):
    file_info,new_row,center_file = sam_info(file = file ,sam_result = loaded_segment_sam_results,npy_path=npy_path,csv_path=csv_path,dest_path = dest_path)

    if file_info is not None and new_row is not None and center_file is not None:
        # Step 4 : From the pixels in center_file, to get the original corresponding pixel position for school center ( npy resized ---> npy )
        pixel_original = []
        depth_sea = []
        gps_lon_lat = []
        time = []
        csv_name = file.replace('complete_new.png','.csv')
        csv_table = pd.read_csv(os.path.join(csv_path,csv_name))
        # Step 4-1 : From center_file to get the school center in resized file
        cut_intervals = loaded_mapping_info[csv_name] 
        pixel_positions = [row[1] for row in center_file]
        
        # Step 4-2 : From mapping table to get the school position in original file
        for pixel_position in pixel_positions:
            original_position = find_original_position(pixel_position, cut_intervals)
            pixel_original.append(original_position)
            depth_sea.append(csv_table.iloc[original_position-1]['bottom_depth'])
            gps_lon_lat.append([csv_table.iloc[original_position-1]['Longitude'],csv_table.iloc[original_position-1]['Latitude']])
            time.append(csv_table.iloc[original_position-1]['time'])

        # Save all the output by double threshold 
        file_info['sea_depth'] = depth_sea
        file_info['school_seabed_distance'] = [a - b for a, b in zip(file_info['sea_depth'], file_info['depth'])]
        file_info['gps_lon_lat'] = gps_lon_lat
        file_info['time'] = time
        file_info['depth_ratio'] = [d / s for d, s in zip(file_info['depth'], file_info['school_seabed_distance'])]
        segment_sam_info[file] = file_info
        segment_sam_table = pd.concat([segment_sam_table, pd.DataFrame([new_row])], ignore_index=True)

# Step 5 : Save all output
segment_sam_table.to_csv(f'{dest_path}/sam_image_masktb200_afterv0.csv', index=False)   
# Save the results to the specified file using pickle
save_path = os.path.join(dest_path, 'sam_info.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(segment_sam_info, f)  





# read the output file by GPU
dis_limit = 10
dis_check = 50

with open(os.path.join(dest_path,'sam_info.pkl'), 'rb') as file:
    segment_sam_info = pickle.load(file)
##########################################################################################
# Combine the school informations from pkl file by sam  
##########################################################################################
file_list = list(segment_sam_info.keys())
seg_new = {}
for file in file_list:
    seg_new[file] = segment_sam_info[file]

# label all the school    
j = 1
# sam_school_table = pd.DataFrame(columns=['number', 'label','file_name','size','depth','intensity_school','center','sea_depth','school_seabed_distance','gps_lon_lat','time'])
sam_school_table = pd.DataFrame(columns=['number', 'label','file_name','size','bbox','center','depth','sea_depth',\
                                     'school_seabed_distance','gps_lon_lat','time',
                                     "width_bbox","length_box","is_very_wide","is_very_tall","dis_to_surface",\
                                     'dis_level',"axis_major_length","axis_minor_length","perimeter_school",\
                                    'intensity_school',"intensity_img","dif_intensity_school_image","center_square_intensity","edges_school_intensity",\
                                    "dif_intensity_center_edges","std_intensity_school","gradient_school",\
                                    "gradient_school_center","gradient_school_edges","dif_gradient_center_edges",\
                                    "width_length_ratio","axis_ellipse_ratio","solidity","compactness",\
                                    "inertia_tensor_eigvals_ratio","perimeter_area_ratio"])

for file in file_list:
    nbr_school = seg_new[file]['nbr_school']
    index_nbr = 1
    for i in range(len(seg_new[file]['label'])):
    # for label in seg_new[file]['label']:
        new_row = {'number':j, 
                   'label':seg_new[file]['label'][i],
                    'file_name':file,
                    'size':seg_new[file]['size'][i],
                    'bbox':seg_new[file]['bbox'][i],
                    'depth':seg_new[file]['depth'][i],
                    'center':seg_new[file]['center'][i],
                    'sea_depth':seg_new[file]['sea_depth'][i],
                    'school_seabed_distance':seg_new[file]['school_seabed_distance'][i],
                    'gps_lon_lat':seg_new[file]['gps_lon_lat'][i],
                    'time':seg_new[file]['time'][i],
                    "width_bbox":seg_new[file]['width_length'][i][0],
                    "length_box":seg_new[file]['width_length'][i][1],
                    "is_very_wide":seg_new[file]['is_very_wide'][i],
                    "is_very_tall":seg_new[file]['is_very_tall'][i],
                    'dis_to_surface':seg_new[file]['dis_to_surface'][i],

                    "axis_major_length":seg_new[file]['axis_ellipse'][i][0],
                    "axis_minor_length":seg_new[file]['axis_ellipse'][i][1],
                    "perimeter_school":seg_new[file]['perimeter_school'][i],
                    'intensity_school':seg_new[file]['intensity_school'][i][0],
                    "intensity_img":seg_new[file]['intensity_img'][0],
                    "center_square_intensity":seg_new[file]['center_square_intensity'][i],
                    "edges_school_intensity":seg_new[file]['edges_school_intensity'][i],

                    "dif_intensity_center_edges":seg_new[file]['dif_intensity_center_edges'][i],
                    "std_intensity_school":seg_new[file]['std_intensity_school'][i],
                    "gradient_school":seg_new[file]['gradient_school'][i],
                    "gradient_school_center":seg_new[file]['gradient_school_center'][i],
                    "gradient_school_edges":seg_new[file]['gradient_school_edges'][i],
                    "dif_gradient_center_edges":seg_new[file]['dif_gradient_center_edges'][i],\
                    "width_length_ratio":seg_new[file]['width_length_ratio'][i],
                    "axis_ellipse_ratio":seg_new[file]['axis_ellipse_ratio'][i],
                    "solidity":seg_new[file]['solidity'][i],
                    "compactness":seg_new[file]['compactness'][i],
                    "inertia_tensor_eigvals_ratio":seg_new[file]['inertia_tensor_eigvals_ratio'][i],
                    "perimeter_area_ratio":seg_new[file]['perimeter_area_ratio'][i]}
        new_row['dis_to_bottom'] = new_row['sea_depth']*10 - new_row['length_box']/2 - new_row['depth']*10
        if new_row['dis_to_surface'] > dis_check and new_row['dis_to_bottom'] > dis_check:
            dis_level = 2
        elif new_row['dis_to_surface'] < dis_limit or new_row['dis_to_bottom'] < dis_limit:
            dis_level = 0
        else:
            dis_level = 1
        new_row['dif_intensity_school_image'] = new_row['intensity_school'] - new_row['intensity_img']
        new_row['dis_level'] = dis_level
        index_nbr = index_nbr + 1
        sam_school_table = pd.concat([sam_school_table, pd.DataFrame([new_row])], ignore_index=True)

        j = j + 1


sam_school_table.to_csv(f'{dest_path}/sam_school_table.csv', index=False)   