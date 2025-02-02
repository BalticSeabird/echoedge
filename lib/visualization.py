import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Plot and Visualize data
def data_to_images(data, img_path='', npy_path = '', normalization=False, upper=False, lower=False):
    """
    Function to create images (greyscale & viridis) from np_array with high resolution and exportmatrix of the image as .npy
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
    
    if img_path.endswith('complete') and not os.path.exists(f'{npy_path}.npy'):
        np.save(f'{npy_path}', data)

    # flip_np_data = cv2.flip(np_data, 1) # flip the image to display it correctly

    #cv2.imwrite(f'{filepath}_greyscale.png', np_data) # greyscale image
    cv2.imwrite(f'{img_path}_greyscale.png', np_data) # greyscale image

    image = cv2.imread(f'{img_path}_greyscale.png', 0)
    colormap = plt.get_cmap('viridis') # 
    heatmap = (colormap(image) * 2**16).astype(np.uint16)[:,:,:3]
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    cv2.imwrite(f'{img_path}.png', heatmap)
    #os.remove(f'{filepath}_greyscale.png')