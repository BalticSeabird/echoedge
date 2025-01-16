#Load libraries
from pathlib import Path
import moviepy.editor as mpy
import glob
import imagesize
import pandas as pd
import cv2
import datetime
import os

# First remove all files in dump foldet
files = glob.glob("dump/*")

for file in files:
    os.remove(file)


#Create a sorted list with all the png files in our directory
file_list = sorted(glob.glob("../../../../../mnt/BSP_NAS2/Sailor/Echopype_results/Baltic2024/run10okt24/img/*0_complete.png")) 


# Check image size of all images 
ws = []
hs = []
for file in file_list: 
    width, height = imagesize.get(file)
    ws.append(width)
    hs.append(height) 

# Subset file list based on image size
include = []
final_list = []
for i in range(0, len(ws)): 
    if (ws[i] > 149 and hs[i] == 1000):
        include.append(1)
        final_list.append(file_list[i])
    else:
        include.append(0) 

# Date for each file 
date = []

for file in final_list:
    t = Path(file)
    date.append(t.name.split("-")[2][1:]) 
    
dates = []
for x in date:
    if x not in dates:
        dates.append(x)


# Stack images horizontally 

# horizontally concatenates images 
# of same height  

tempfiles = []
counter = 0
imcounter = 0 
for file in final_list:
    if counter < 20:
        tempfiles.append(file)
        counter += 1
    else:
        images = [cv2.imread(file) for file in tempfiles]
        im_h = cv2.hconcat(images)
        d1, t1 = Path(tempfiles[0]).name.split("-")[2:4]
        dx1 = datetime.datetime.strptime(d1+t1, "D%Y%m%dT%H%M%S") 
        str1 = dx1.strftime("%d %B, %H:%M")
        d2, t2 = Path(tempfiles[-1]).name.split("-")[2:4]
        dx2 = datetime.datetime.strptime(d2+t2, "D%Y%m%dT%H%M%S") 
        str2 = dx2.strftime(" - %H:%M")
        str3 = str1+str2
        text = str3
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (1900, 950)
        fontScale = 2
        color = (0, 0, 255)
        thickness = 2
        im_h = cv2.putText(im_h, text, org, font, fontScale, 
                        color, thickness, cv2.LINE_AA, False)
        imcounter_fill = str(imcounter).zfill(4)    
        cv2.imwrite(f'dump/added_{imcounter_fill}.png', im_h)
        tempfiles = []
        counter = 0
        imcounter += 1


# Check resulting size of images and cut horizontally to get equal size
file_list = sorted(glob.glob("dump/added*")) 

# Check image size of all combined images 
ws = []
hs = []
for file in file_list: 
    width, height = imagesize.get(file)
    ws.append(width)
    hs.append(height) 
min(ws)

# Cut to match minimum size

counter = 0
for file in file_list: 
    im = cv2.imread(file)
    im_crop = im[:,0:2999]
    filenum = str(counter).zfill(4)
    cv2.imwrite(f'dump/added_cropped_{filenum}.png', im_crop)
    counter += 1


# Just take 20 images at the time... 
combined = sorted(glob.glob("dump/added_cropped*"))


#frames per second
fps = 2

#Create a clip instance using ImageSequenceClip included in moviepy
clip = mpy.ImageSequenceClip(combined, fps=fps)

#name of the animation
gif_name = 'echogram_baltic2024_masked'

#No we can write the animation as a a gif
clip.write_gif("dump/"+gif_name+'.gif')

#No we can write the animation as a a mp4
clip.write_videofile("dump/"+gif_name+'.mp4')