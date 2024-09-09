#Load libraries
from pathlib import Path
import moviepy.editor as mpy
import glob
import imagesize
import pandas as pd
import cv2
import datetime

#Create a sorted list with all the png files in our directory
file_list = sorted(glob.glob("../../../research/Acoustics/AISailor/VOTO_Hudson/Analysis/HudsonBay2024/Run2/img/*0_complete.png")) 


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
    if (ws[i] == 160 and hs[i] == 1500):
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
        org = (1900, 1450)
        fontScale = 2
        color = (255, 255, 255)
        thickness = 2
        im_h = cv2.putText(im_h, text, org, font, fontScale, 
                        color, thickness, cv2.LINE_AA, False)    
        cv2.imwrite(f'dump/added_{imcounter}.png', im_h)
        tempfiles = []
        counter = 0
        imcounter += 1

# text

# Just take 20 images at the time... 
combined = sorted(glob.glob("dump/added*")) 

#frames per second
fps = 2

#Create a clip instance using ImageSequenceClip included in moviepy
clip = mpy.ImageSequenceClip(combined, fps=fps)
#clip = clip.resize((200, 200))

#name of the animation
gif_name = 'echograms_complete_added'

#No we can write the animation as a a gif
clip.write_gif("dump/"+gif_name+'.gif')

#No we can write the animation as a a mp4
clip.write_videofile("dump/"+gif_name+'.mp4')