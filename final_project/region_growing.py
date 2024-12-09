"""
Image Analysis Final Project - Region Growing
Bret A. Brouse Jr. - 12.10.2024
"""

# Preamble
######################################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import segmentation
from collections import deque
from skimage import data
# from skimage import segmentation, io, color, img_as_float




# Functions
######################################################################

""" region growing functiont hat takes in an RGB image, a seed point to 
    base the regions on, and a threshold for similarity and returns a 
    binary mask of the segmented region
"""
def region_growing(img, seed, threshold):
    
    # img dimensions
    height, width = img.shape[0], img.shape[1]
    
    # initialize a segmented region mask
    segged = np.zeros((height, width), dtype=np.uint8)
    
    # convert img to float
    img = img.astype(np.float32)
    
    # get the seed values
    seed_pt = img[seed[1], seed[0]]
    
    # make a queue for region growing
    queue = deque([seed])
    segged[seed[1], seed[0]] = 1    # seed is part of this region
    
    # define 8-connectivity offsets
    bros = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
            (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in bros:
            nx, ny = x + dx, y + dy
            
            # check if neighbor is within bounds
            if 0 <= nx < width and 0 <= ny < height and not segged[ny, nx]:
                
                # calculate the pixel similarity
                r = np.linalg.norm(img[ny, nx] - seed_pt)
                
                if r < threshold:
                    segged[ny, nx] = 1
                    queue.append((nx, ny))
                    
    return segged

# read in images
def img_read(path, filename):
    img = cv2.imread(path + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img
    
    
# Variables
######################################################################
# paths
imgs_folder = 'images/'
save_path = 'images/pres_images/'

# images
laney_jpg = 'laney.jpg'
bobby_jpg = 'pissed_bobby.jpg'
neuro_tif = 'bk_ch00.tif'
monarch_png = 'monarch.png'

# pick a seed point and set the threshold
# seed_pt = (200, 300) # Laney Blind 1
# seed_pt = (300, 300) # Laney Blind 2
# seed_pt = (630, 410) # Laney Heart 1
# seed_pt = (700, 450) # Laney Heart 2
seed_pt = (375, 1250) # Neuron
# seed_pt = (200, 160)    # Astronaut
thresh = 30.0


# Execution
######################################################################
# read in img
# img = img_read(imgs_folder, laney_jpg)    # laney imp
# img = img_read(imgs_folder, bobby_jpg)    # bobby imp
img = img_read(imgs_folder, neuro_tif)    # neuron imp
# img = img_read(imgs_folder, monarch_png)  # monarch imp (testing)
# img = data.astronaut()                    # astronaut

# plt.imshow(img)

# apply the region growing algorithm
seg_mask = region_growing(img, seed_pt, thresh)

# use mask to show boundaries
bnds =  segmentation.mark_boundaries(img, seg_mask, color=(0, 1, 1))


plt.figure(dpi=300)
plt.title(f'Seed value = {img[seed_pt[1], seed_pt[0]]}')
plt.axis('off')

plt.plot(seed_pt[0], seed_pt[1], 'o', color='red', markersize=3.7)
plt.imshow(bnds)

plt.savefig(save_path + 'neuron_30.png')

# print(img[seed_pt[1], seed_pt[0]])

# print(img[160, 200])






























    
