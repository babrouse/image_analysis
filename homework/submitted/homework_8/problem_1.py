from skimage.morphology import (erosion, dilation, disk, closing, opening)
from skimage import io, filters, morphology
import matplotlib.pyplot as plt
import numpy as np

def renormalize_img(img):
    img_max = np.max(img)
    img_min = np.min(img)
    
    A = img - img_min
    B = img_max - img_min
    C = 255 * A/B
    
    img = C.astype(np.uint8)
    
    return img


# img_gray = io.imread('images/Lichtenstein_imageDuplicator_1963_gray.png')
# img_color = io.imread('images/Lichtenstein_imageDuplicator_1963.png')

# # plt.imshow(img_gray, cmap='gray')
# # plt.imshow(img_color)

# selem = disk(3)

# img_closed = closing(img_gray, selem)
# img_dilated = dilation(img_gray, selem)
# img_eroded = erosion(img_gray, selem)
# img_opened = opening(img_gray, selem)

# plt.imshow(img_opened, cmap='gray')

# # closing and dilated are similar while erosion and opening are similar
# # all of them remove dots, the first two mess with letters and the second two
# # darken the face quite a bit (washing it all out)

# red_channel = img_color[:, :, 0]
# green_channel = img_color[:, :, 1]
# blue_channel = img_color[:, :, 2]

# # detect red dots: red intensity higher than others
# channel_thresh = 1.2
# red_mask = (red_channel > channel_thresh * green_channel) & (red_channel > channel_thresh * blue_channel)

# red_mask = opening(red_mask, selem)

# img_red_scrubbed = img_color.copy()
# img_red_scrubbed[:, :, 0][red_mask] = (green_channel[red_mask] + blue_channel[red_mask]) // 2
# red_dots = img_red_scrubbed - img_color

# plt.imshow(red_dots)

# red_dot_mask = (red_channel > green_channel + 50) & (red_channel > blue_channel + 50)
# filtered_img = img_color.copy()
# filtered_img[red_dot_mask] = np.mean(img_color[~red_dot_mask], axis=0).astype(np.uint8)

# filtered_img_plus_red = filtered_img + img_red_scrubbed

# plt.imshow(filtered_img_plus_red)






################################################
# Prob 2a

# was running into dpi issues only on this pc ugh
plt.figure(dpi=300)

petri_img = io.imread("images/dsc_0357_gray.png")

# apply a gaussian
gauss_img = filters.gaussian(petri_img, sigma=55)*255

# apply a highpass filter
hp_img = petri_img - gauss_img

# normalize
hp_img = renormalize_img(hp_img)


# plt.hist(hp_img.ravel(), bins=256, edgecolor='black', linewidth=0.5)
# plt.imshow(hp_img, cmap='gray')

# set some threshold boundary to get rid of the lesser intensity things
thresh = 150

thresh_img = hp_img.copy()
thresh_img[thresh_img < thresh] = 0

# make black and white
bw_img = np.where(thresh_img > 0, 1, 0).astype(np.uint8)

# plt.imshow(bw_img, cmap='gray')





################################################
# Prob 2b

from skimage import measure as skmeasure
label_img = skmeasure.label(bw_img) # label connected pixels
stats = skmeasure.regionprops(label_img) # stats on the object





























