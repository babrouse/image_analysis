"""
Image Analysis Final Project - k-means
Bret A. Brouse Jr. - 12.10.2024
"""

# Preamble
######################################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import segmentation
from skimage.data import astronaut





# Functions
######################################################################

# read in images
def img_read(path, filename):
    img = cv2.imread(path + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

# def k_means(img):
#     segged_img = img
    
#     pixel_vals = np.float32(img.reshape((-1, 3))
    
#     return segged_img


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

# k-means variables
iterations = 100   # number of iterations to end at
epsilon = 0.85     # 'distance' threshold
k = 3              # number of clusters

# Execution
######################################################################
# read in img
# img = img_read(imgs_folder, laney_jpg)    # laney imp
# img = img_read(imgs_folder, bobby_jpg)    # bobby imp
img = img_read(imgs_folder, neuro_tif)    # neuron imp
# img = img_read(imgs_folder, monarch_png)  # monarch imp (testing)
# img = astronaut()                         # astronaut

plt.figure(dpi=300)
plt.title('original')
plt.imshow(img)
plt.axis('off')
# plt.savefig(save_path + 'neuron.png')

# plt.imshow(img)

# reshape into a 2D array
pixel_vals = img.reshape((-1, 3))
pixel_vals = np.float32(pixel_vals)
# pixel_vals = pixel_vals.ravel()

crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iterations, epsilon)

# Run k-means algorithm
retval, labels, centers = cv2.kmeans(pixel_vals, k, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
""" Words on line 73
    Assignments:
        retval: sum of squared distances from each point of data to centroid
        labels: labels that indidcate which cluster a pixel belongs to
        centers: RGB color value of a clusters centroid
    Arguments:
        pixel_vals: input img
        k: number of clusters we want
        None: random labels to start, can give a set of labels
        crit: criteria above
        10: attempts; number of times algorithm is run with random points
        RANDOM_CENTERS: start with random points on the image, can also use
            cv2.KMEANS_PP_CENTERS
"""

# convert into 8-bit values
centers = np.uint8(centers)
seg_data = centers[labels.flatten()]

# Reshape to match original image
seg_img = seg_data.reshape((img.shape))


labels_reshaped = labels.reshape((img.shape[0], img.shape[1]))
bnd_img = segmentation.mark_boundaries(seg_img, labels_reshaped, color=(0, 1, 1))

plt.figure(dpi=300)
plt.title('k = 20 clusters')
plt.axis('off')
plt.imshow(seg_img)

# plt.savefig(save_path + 'astro_k20_nobnds.png')

# plt.hist(pixel_vals[pixel_vals > 5], bins=256, linewidth=0.0, edgecolor='black', color='purple')









