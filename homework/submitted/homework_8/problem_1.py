from skimage.morphology import (erosion, dilation, disk, closing)
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

img_gray = plt.imread('images/Lichtenstein_imageDuplicator_1963_gray.png')
img_color = plt.imread('images/Lichtenstein_imageDuplicator_1963.png')

# plt.imshow(img_gray, cmap='gray')
# plt.imshow(img_color)

ste = disk(5)
print(ste)