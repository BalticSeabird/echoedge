import numpy as np 
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation,reconstruction,rectangle
import numpy.ma as ma
from skimage.transform import rescale
import skimage
import os
from skimage.measure import label, regionprops
import pandas as pd
from dateutil import parser
from astral import LocationInfo
from astral.sun import sun
import datetime

########################################   list of functions  ########################################
#  line : 32        apply_day_or_night(row)               (not used in the key code part)
#  line : 36        convergence_test_new20240730(arr)      (to test the convergence of the images)
#  line : 84        day_or_night(coords,timezone,time)      (not used in the key code part)
#  line : 122       find_center_square(centroid, coords)      (get the square 3*3 centered by a choosen pixel)  
#  line : 139       find_edges(image, row, col)
#  line : 178       find_first_incolumns(matrix,element)       (not used in the key code part)
#  line : 195       find_minimum_dis(matrix2,matrix1)        
#  line : 214       find_original_position(resized_pixel, intervals)
#  line : 227       inter_positions_new(arr)
#  line : 244       npy_correction_v3(img,total_rows,shape_top,shape_bottom)
#  line : 280       parameters_correction(img,median_sea_depth,threshold = -30)          
#  line : 335       resize_matrix_31052024(matrix,velocity,meanvelocity)
########################################   end of list        ########################################


# Define a helper function to apply the function day_or_night(coords,timezone,time) to a dataframe
def apply_day_or_night(row):
    return day_or_night(row['gps_lon_lat'], 'Australia/Melbourne', row['time'])


def convergence_test_new20240730(arr):
    """
    Determine convergence type and intervals in a sequence.

    Args:
        arr (list): Input sequence.

    Returns:
        tuple: A tuple containing:
            - max_start_index (int): Start index of the longest convergence interval.
            - max_end_index (int): End index of the longest convergence interval.
            - convergence (int): Type of convergence (0, 1, or 2). 
    """

    # Initialize list to store segment indices
    segments = []
    # Find segment indices where the value changes
    for i in range(1,len(arr)):
        if arr[i] != arr[i-1]:
            segments.append(i)
    segments.insert(0, 0)

    temp = segments.copy()
    temp.append(len(arr))
    temp = temp[1:]
    # Compute the length of each segment
    seq_length = [j - i for i, j in zip(segments, temp)] 
    # Find the maximum segment length and its index
    max_length = np.max(seq_length)
    max_p = len(seq_length) - 1 - seq_length[::-1].index(max_length)
    # max_p = seq_length.index(max_length)

    max_start_index = segments[max_p]
    max_end_index = temp[max_p]-1

    if arr[max_start_index] == 0:
        convergence = 0
    else:
        if max_length >= 4:
            convergence = 2
        else:
            convergence = 1
       
    

    # Convert max_start_index and max_end_index to integers
    max_start_index = int(max_start_index)
    max_end_index = int(max_end_index)
    return max_start_index, max_end_index, arr[max_start_index],convergence


def day_or_night(coords,timezone,time):
    # Remove the square brackets
    coordinates_str = coords.strip("[]")

    # Split the string by comma to get a list of strings
    coordinates_list_str = coordinates_str.split(",")

    # Convert the list of strings to a list of floats
    coordinates = [float(coord) for coord in coordinates_list_str]

    # Access the first and second numbers
    lon = coordinates[0]
    lat = coordinates[1]

    loc = LocationInfo(name='Melbourne', region='Australia', timezone=timezone,
                   latitude=lat, longitude=lon)

    # Convert the string to a datetime object
    date_datetime = parser.isoparse(time)
    year = date_datetime.year
    month = date_datetime.month
    day = date_datetime.day
  
    s = sun(loc.observer, date=datetime.date(year,month,day), tzinfo=loc.timezone)
    sunrise_time = s['sunrise']
    sunset_time = s['sunset']

    # Format the sunrise and sunset times
    sunrise_str = sunrise_time.strftime('%Y-%m-%d %H:%M:%S %Z')
    sunset_str = sunset_time.strftime('%Y-%m-%d %H:%M:%S %Z')

    if sunrise_time < date_datetime < sunset_time :
        return 1,sunrise_str,sunset_str
    else:
        return 0,sunrise_str,sunset_str



def find_center_square(centroid, coords):
    """
    Get the square of the center  in the image.
    
    """
    int_centroid = np.round(centroid).astype(int)
    square_matrice = [[int_centroid[0]-1,int_centroid[1]-1],[int_centroid[0]-1,int_centroid[1]],[int_centroid[0]-1,int_centroid[1]+1],\
                               [int_centroid[0],int_centroid[1]-1],[int_centroid[0],int_centroid[1]],[int_centroid[0],int_centroid[1]+1],\
                                [int_centroid[0]+1,int_centroid[1]-1],[int_centroid[0]+1,int_centroid[1]],[int_centroid[0]+1,int_centroid[1]+1]]
    square_matrice_set = set(map(tuple, square_matrice))
    coords_set = set(map(tuple, coords))
    common_coords = square_matrice_set.intersection(coords_set)
    
    # Convert the set back to a list of lists
    return list(map(list, common_coords))


def find_edges(image, row, col):
    """
    Get the four connected neighbors of a given position in the image.

    Parameters:
    - image: numpy array representing the image or matrix
    - row, col: coordinates of the position in the image

    Returns:
    - List of tuples containing coordinates of the four connected neighbors
    """
    edges_object = []
    neighbors_pixel_sum_c = []
    for i, j in zip(row,col):
        neighbors = []
        # Check north neighbor (above)
        if i > 0:
            neighbors.append((i - 1, j))

        # Check south neighbor (below)
        if i < image.shape[0] - 1:
            neighbors.append((i + 1, j))

        # Check west neighbor (left)
        if j > 0:
            neighbors.append((i, j - 1))

        # Check east neighbor (right)
        if j < image.shape[1] - 1:
            neighbors.append((i, j + 1))

        sum_neighbors_pixels = np.sum([image[n[0], n[1]] for n in neighbors])
        
        if sum_neighbors_pixels<4:
            edges_object.append([i,j])
    return edges_object



def find_first_incolumns(matrix,element): 
    # find the element position in each columne
    coords = np.argwhere(matrix == element)
    coords_r = coords[:, 0]
    coords_c = coords[:, 1]
    column_count = []
    first_postion = []
    last_postion = []
    for i in np.unique(coords_c):
        coords_f = [coord[0] for coord in coords if coord[1] == i]
        first_postion.append(np.min(coords_f))
        last_postion.append(np.max(coords_f))
        column_count.append(i)
    return column_count,first_postion,last_postion



def find_minimum_dis(matrix2,matrix1):
    # matrix2 to matrix1 
    column_count1,first_postion1,last_postion1 = find_first_incolumns(matrix1,False)
    column_count2,first_postion2,last_postion2 = find_first_incolumns(matrix2,True)
    
    distance_top = []
    distance_bottom = []
    for i in range(0,len(column_count2)):
        index1 = np.where(np.array(column_count1) == column_count2[i])[0]
        index1 = index1[0]
        # print('i,index1',i,index1,'top_f,m_top_t',first_postion1[index1],first_postion2[i],first_postion2[i]-first_postion1[index1])
        # print('i,index1',i,index1,'bot_f,m_bot_t',last_postion1[index1],last_postion2[i],last_postion1[index1]-last_postion2[i])
        distance_top.append(first_postion2[i]-first_postion1[index1])
        distance_bottom.append(last_postion1[index1]-last_postion2[i])
    # print(np.min(distance_bottom),np.min(distance_top))
    return np.min(distance_top),np.min(distance_bottom)



def find_original_position(resized_pixel, intervals):
    for (orig_start, orig_end, resized_start, resized_end) in intervals:
        if resized_start <= resized_pixel < resized_end:
            # Calculate the scaling factor for this segment
            scaling_factor = (orig_end - orig_start) / (resized_end - resized_start)
            # Calculate the original pixel position
            orig_pixel_float = orig_start + (resized_pixel - resized_start) * scaling_factor
            orig_pixel = int(max(orig_start, min(orig_pixel_float, orig_end - 1)))  # Ensure orig_pixel is within the interval
            return orig_pixel # Return original pixel and interval index
    return None  # Pixel not found in any segment



def inter_positions_new(arr):
    segments = []
    arr_unique = [arr[0]]
    for i in range(1,len(arr)):
        if arr[i] != arr[i-1]:
            segments.append(i)
            arr_unique.append(arr[i])

    segments.insert(0, 0)
    result_dict = {
        "segments": segments,
        "unique_values": arr_unique
    }
    return result_dict



def npy_correction_v3(img,total_rows,shape_top,shape_bottom):
    """
    Image corrections.
    
    Parameters:
    img: np array loaded from npy file.

    Returns:
    np.ma.core.MaskedArray: The corrected and masked image array.
    """
    dilation_top = 85
    # Step 1 : mask the image 
    mask = np.zeros(img.shape)
    mask[img >-30] = 1 
    top_mask = mask[:total_rows, :]
    bottom_mask = mask[total_rows:, :]
    dilated_bottom_mask = binary_dilation(bottom_mask, footprint = shape_bottom)
    dilated_top_mask = binary_dilation(top_mask, footprint = shape_top)
    mask[:total_rows, :] = dilated_top_mask
    mask[total_rows:, :] = dilated_bottom_mask
    masked_img = ma.array(img, mask = mask)

    # Step 2 : Calculate the median at the same depth level
    fond = np.ma.median(masked_img, axis = 1).reshape((masked_img.shape[0], 1))
    im_fond = np.repeat(fond, masked_img.shape[1], axis=1)

    # Step 3 : Get the corrected image
    corrected_img = 1 - masked_img/im_fond
    
    # Step 4 : Gaussian filtering
    tmp = skimage.filters.gaussian(corrected_img, sigma=0.3) 
    im = ma.array(tmp, mask = corrected_img.mask)
    return(im)



def parameters_correction(img,median_sea_depth,threshold = -30):
    sharp_wave = 160
    sharp_bottom = 20

    rows_matrix, columns_matrix = img.shape
    max_wave_height_top = []
    max_wave_height_bot = []
    for i in range(columns_matrix):
        arr = (img[:,i]>threshold).astype(int)
        # for the top
        min_value = np.min(arr)
        first_min = np.where(arr == min_value)[0][0]
        max_wave_height_top.append(first_min)

        # for the bottom
        reversed_index = np.where(arr[::-1] == min_value)[0][0]
        last_min = len(arr)- reversed_index
        max_wave_height_bot.append(last_min)
    
    if median_sea_depth < 25:
        total_lines = int(median_sea_depth*10-50)
    elif median_sea_depth < 35:
        total_lines = 200
    elif median_sea_depth < 40:
        if np.max(max_wave_height_top)<50:
            total_lines = 200
        else:
            total_lines = 300
    else:
        if np.max(max_wave_height_top)<50:
            total_lines = 200
        elif np.max(max_wave_height_top)<100:
            total_lines = 300
        else:
            total_lines = 350

    if np.max(max_wave_height_top) > sharp_wave:
        shape_top = rectangle(130,20)
        shape_top_desc = "rectangle(130, 20)"
    else:
        shape_top = rectangle(80,20)
        shape_top_desc = "rectangle(80,20)"

        
    if np.var(max_wave_height_bot) > sharp_bottom:
        shape_bottom = rectangle(20,10)
        shape_bottom_desc = "rectangle(20,10)"
    else:
        shape_bottom = rectangle(10,20)
        shape_bottom_desc = "rectangle(10,20)"

    return shape_top,shape_bottom,total_lines,shape_top_desc,shape_bottom_desc



def resize_matrix_31052024(matrix,velocity,meanvelocity):
    """
    matrice : matrix for a data.npy file
    velocity : the number of lines = the pixel count vertically in the matrice
    """

    # get the division positions and factors
    div = inter_positions_new(velocity)
    div_pix = div['segments']
    div_fac = div['unique_values']

    # Initialize an empty image with the same dimensions as the input image
    matrix_groupe = matrix[:, :0].copy()
    # matrix_groupe = np.zeros((matrix.shape[0], 0), dtype=matrix.dtype)
    mapping = []
    for i in range(len(div_pix)):
        # get the division position of pixels
        p_start = div_pix[i]
        p_end = div_pix[i + 1] if i + 1 < len(div_pix) else matrix.shape[1]
        
        # get the zoom factor fx, (fy=1)
        f_x = (div_fac[i] / meanvelocity)

        # chop the image divided by pixels
        left_matrix = matrix[:,p_start:p_end]
        if f_x > 1:
            matrix_chop =  rescale(left_matrix, (1,f_x), anti_aliasing=True, anti_aliasing_sigma=1)
        else:
            matrix_chop =  rescale(left_matrix, (1,f_x), anti_aliasing=True, anti_aliasing_sigma=1)
      
        mapping.append((p_start, p_end, matrix_groupe.shape[1],matrix_groupe.shape[1]+matrix_chop.shape[1]))
        # Group the image with the images gotten before
        matrix_groupe = np.concatenate((matrix_groupe, matrix_chop), axis=1)
    return matrix_groupe, mapping