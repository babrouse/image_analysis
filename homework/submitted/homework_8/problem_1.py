import matplotlib.pyplot as plt
import numpy as np

from skimage.morphology import (erosion, dilation, disk, closing, opening)
from skimage import io, filters, morphology, segmentation, measure
from skimage.feature import peak_local_max
from skimage import measure as skmeasure
from scipy import ndimage

def renormalize_img(img):
    img_max = np.max(img)
    img_min = np.min(img)
    
    A = img - img_min
    B = img_max - img_min
    C = 255 * A/B
    
    img = C.astype(np.uint8)
    
    return img


# # was running into dpi issues only on this pc ugh (work pc)
plt.figure(dpi=300)


img_gray = io.imread('images/Lichtenstein_imageDuplicator_1963_gray.png')
img_color = io.imread('images/Lichtenstein_imageDuplicator_1963.png')

# plt.imshow(img_gray, cmap='gray')
# plt.imshow(img_color)

selem = disk(3)

# img_closed = closing(img_gray, selem)
# img_dilated = dilation(img_gray, selem)
# img_eroded = erosion(img_gray, selem)
# img_opened = opening(img_gray, selem)

# plt.imshow(img_opened, cmap='gray')

# closing and dilated are similar while erosion and opening are similar
# all of them remove dots, the first two mess with letters and the second two
# darken the face quite a bit (washing it all out)

red_channel = img_color[:, :, 0]
green_channel = img_color[:, :, 1]
blue_channel = img_color[:, :, 2]

red_mask = (red_channel > 150) & (green_channel < 100) & (blue_channel < 100)
large_mask = dilation(red_mask, disk(1))

# plt.imshow(red_mask)

img_removed = img_color.copy()
img_removed[large_mask] = [0, 0, 0]

plt.imshow(img_removed)











# ################################################
# # Prob 2a

# petri_img = io.imread("images/dsc_0357_gray.png")

# # apply a gaussian
# gauss_img = filters.gaussian(petri_img, sigma=55)*255

# # apply a highpass filter
# hp_img = petri_img - gauss_img

# # normalize
# hp_img = renormalize_img(hp_img)


# # plt.hist(hp_img.ravel(), bins=256, edgecolor='black', linewidth=0.5)
# # plt.imshow(hp_img, cmap='gray')

# # set some threshold boundary to get rid of the lesser intensity things
# thresh = 150

# thresh_img = hp_img.copy()
# thresh_img[thresh_img < thresh] = 0

# # make black and white
# bw_img = np.where(thresh_img > 0, 1, 0).astype(np.uint8)

# # plt.imshow(bw_img, cmap='gray')





# ################################################
# # Prob 2b

# label_img = skmeasure.label(bw_img) # label connected pixels
# stats = skmeasure.regionprops(label_img) # stats on the object

# # Region properties
# label_img = skmeasure.label(bw_img)
# stats = skmeasure.regionprops(label_img)
# # Plot properties of threshold-segmented regions
# areas = [stats[j]['Area'] for j in range(len(stats))]
# eccentricities = [stats[j]['Eccentricity'] for j in range(len(stats))]

# # plotting
# plt.figure(figsize=(8, 6))
# plt.scatter(eccentricities, np.log10(areas), color='purple')
# plt.xscale('log')

# plt.xlabel('Areas (log)')
# plt.ylabel('Eccentricities')
# plt.title('Eccentricites vs log(Areas)')
# plt.grid(True, which='both', linestyle='--', linewidth=0.3)
# plt.show()





# ################################################
# Prob 3

# sobel_x = np.array([[2, 1, 0, -1, -2],
#                     [3, 2, 0, -2, -3],
#                     [4, 3, 0, -3, -4], 
#                     [3, 2, 0, -2, -3], 
#                     [2, 1, 0, -1, -2]])

# sobel_y = np.array([[2, 3, 4, 3, 2], 
#                     [1, 2, 3, 2, 1], 
#                     [0, 0, 0, 0, 0], 
#                     [-1, -2, -3, -2, -1], 
#                     [-2, -3, -4, -3, -2]])

# petri_img = io.imread("images/dsc_0357_gray.png")

# # apply a gaussian
# gauss_img = filters.gaussian(petri_img, sigma=55)*255

# # apply a highpass filter
# hp_img = petri_img - gauss_img

# # normalize
# hp_img = renormalize_img(hp_img)
# hp_img = hp_img / 255

# img_sobel_x = ndimage.convolve(hp_img, sobel_x)
# img_sobel_y = ndimage.convolve(hp_img, sobel_y)

# gradient_img = np.sqrt(img_sobel_x**2 + img_sobel_y**2)

# plt.imshow(gradient_img)





# # ###############################################
# # Prob 4

# sobel_x = np.array([[2, 1, 0, -1, -2],
#                     [3, 2, 0, -2, -3],
#                     [4, 3, 0, -3, -4], 
#                     [3, 2, 0, -2, -3], 
#                     [2, 1, 0, -1, -2]])

# sobel_y = np.array([[2, 3, 4, 3, 2], 
#                     [1, 2, 3, 2, 1], 
#                     [0, 0, 0, 0, 0], 
#                     [-1, -2, -3, -2, -1], 
#                     [-2, -3, -4, -3, -2]])

# petri_img = io.imread("images/dsc_0357_gray.png")

# # apply a gaussian
# gauss_img = filters.gaussian(petri_img, sigma=55)*255

# # apply a highpass filter
# hp_img = petri_img - gauss_img

# # normalize
# hp_img = renormalize_img(hp_img)
# hp_img = hp_img / 255

# img_sobel_x = ndimage.convolve(hp_img, sobel_x)
# img_sobel_y = ndimage.convolve(hp_img, sobel_y)

# gradient_img = np.sqrt(img_sobel_x**2 + img_sobel_y**2)

# # watershed_img = segmentation.watershed(gradient_img)
# # plt.figure()
# # plt.imshow(watershed_img, cmap='prism')







# ###############################################
# # Prob 5a

# # First copy pasting stuff from before and using the threshed version to find peaks
# petri_img = io.imread("images/dsc_0357_gray.png")

# # apply a gaussian
# gauss_img = filters.gaussian(petri_img, sigma=55)*255

# # apply a highpass filter
# hp_img = petri_img - gauss_img

# # normalize
# hp_img = renormalize_img(hp_img)

# # plt.imshow(hp_img, cmap='gray')

# # plt.hist(hp_img.ravel(), bins=256, edgecolor='black', linewidth=0.5)
# # plt.imshow(hp_img, cmap='gray')

# # set some threshold boundary to get rid of the lesser intensity things
# thresh = 150

# thresh_img = hp_img.copy()
# thresh_img[thresh_img < thresh] = 0

# # make black and white
# bw_img = np.where(thresh_img > 0, 1, 0).astype(np.uint8)

# # plt.imshow(bw_img, cmap='gray')


# # New stuff
# bw_mask = bw_img.copy()
# bw_mask = erosion(bw_mask, disk(4))

# local_max = peak_local_max(hp_img, min_distance=20)
# # print(local_max)

# max_mask = np.zeros_like(hp_img, dtype=bool)
# max_mask[local_max[:, 0], local_max[:, 1]] = True

# both_max = max_mask & (bw_mask == 1)
# y, x = np.where(both_max)

# # plt.imshow(hp_img, cmap='gray')
# # plt.scatter(x, y, color='purple', marker='x', s=2)





# ###############################################
# # Prob 5b

# labels = segmentation.watershed(gradient_img, markers=measure.label(both_max))
# # plt.imshow(labels, cmap='prism')
# # plt.title("")

# # looks like it sometimes tells the difference between colonies that are touching
# # but not entirely





# ###############################################
# # Prob 5c

# # Region properties
# stats = skmeasure.regionprops(labels)
# # Plot properties of threshold-segmented regions
# areas = [stats[j]['Area'] for j in range(len(stats))]
# eccentricities = [stats[j]['Eccentricity'] for j in range(len(stats))]

# # plotting
# plt.figure(figsize=(8, 6))
# plt.scatter(eccentricities, np.log10(areas), color='purple')
# plt.xscale('log')

# plt.xlabel('Areas (log)')
# plt.ylabel('Eccentricities')
# plt.title('Eccentricites vs log(Areas)')
# plt.grid(True, which='both', linestyle='--', linewidth=0.3)
# plt.show()























