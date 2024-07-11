import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Plot and Visualize data
def data_to_images(data, filepath='', normalization=False, upper=False, lower=False):
    """
    Function to create images (greyscale & viridis) from np_array with high resolution.
    """

    np_data = np.nan_to_num(data, copy=True)

    if normalization == 'low-low':

        np_data[np_data < lower] = lower
        np_data[np_data > upper] = lower
    
    else: 
        
        np_data[np_data < lower] = lower
        np_data[np_data > upper] = upper

    np_data = (np_data - lower)/(upper - lower)
    np_data = np_data*256
    
    # flip_np_data = cv2.flip(np_data, 1) # flip the image to display it correctly

    cv2.imwrite(f'{filepath}_greyscale.png', np_data) # greyscale image

    image = cv2.imread(f'{filepath}_greyscale.png', 0)
    colormap = plt.get_cmap('viridis') # 
    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'{filepath}.png', heatmap)
    os.remove(f'{filepath}_greyscale.png')