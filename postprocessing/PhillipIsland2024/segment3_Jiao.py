from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import os
import sys
from tqdm import tqdm
import pickle
#sys.path.append("..")

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



device = "cuda"
sam = sam_model_registry["vit_h"](checkpoint="C:/Users/jliu/Documents/sam_vit_h_4b8939.pth")
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# PATH ###############################################################################
# path : The directory from which to read the files.
# dest_path : The output directory where the output images will be saved.
path = "Output/Mask_img/Test"
dest_path = "Output/SAM/Segment_anything/Test"

######################################################################################


if not os.path.exists(dest_path):
    os.makedirs(dest_path)

files = os.listdir(path)
files_png = [file for file in files if file.endswith('png')]

sam_mask = {}


for file in  tqdm(files_png):    
    file_path = os.path.join(path,file)
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = mask_generator.generate(image)
    # save all the elements from masks
    seg_file = []
    area_file = []
    bbox_file = []
    predicted_iou_file = []
    point_coords_file = []
    stability_score_file = []
    crop_box_file = []

    for i, mask in enumerate(masks):
        seg = mask['segmentation']  # The binary mask of the segment
        area = mask['area']  # The area of the segment
        bbox = mask['bbox']  # The bounding box of the segment (x, y, width, height)
        predicted_iou = mask['predicted_iou']
        point_coords= mask['point_coords']
        stability_score= mask['stability_score']
        crop_box = mask['crop_box']

        seg_file.append(seg)
        area_file.append(area)
        bbox_file.append(bbox)
        predicted_iou_file.append(predicted_iou)
        point_coords_file.append(point_coords)
        stability_score_file.append(stability_score)
        crop_box_file.append(crop_box)

    
    # save the nbr of school and pixel
    file_info = {"segmentation":seg_file,
                 "area":area_file,
                 "bbox":bbox_file,
                 "predicted_iou":predicted_iou_file,
                 "point_coords":point_coords_file,
                 "stability_score":stability_score_file,
                 "crop_box":crop_box_file}
    
    sam_mask[file] = file_info

    # Clear the plot
    plt.clf()

    plt.figure(figsize=(5,5))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(f'{dest_path}/{file[:-4]}_seg.png', dpi=300)
    # Close any existing plots to free memory
    plt.close('all')

# Save the results to the specified file using pickle
with open(os.path.join(dest_path, 'sam_results.pkl'), 'wb') as f:
    pickle.dump(sam_mask, f)
