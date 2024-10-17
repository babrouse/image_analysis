# -*- coding: utf-8 -*-
# fourier_masking_HWproblem.py
"""
Author:   Raghuveer Parthasarathy
Created on Tue Oct  8 07:25:53 2024
Last modified on Oct. 12, 2024

Description
-----------

For a homework problem on masking Fourier Transforms

"""

import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io # input output sub-package 


#%% Load the image

parentDir = r'C:\Users\Raghu\Documents\Teaching\Image Analysis Course\Images for Class'
fileName = r'Lincoln_Coleman_40-copyright-havecamerawilltravel-com_crop512_gray.png'

im = io.imread(os.path.join(parentDir, fileName))

if im.ndim > 2:
    # A bit silly, since I know this is 2D
    im = np.mean(im, axis=2, dtype=im.dtype)

print('Image shape: ', im.shape)
# Image size; I'm not checking if it's square!
# The Lincoln Memorial image is 512x512
N = im.shape[0]

plt.figure()
plt.imshow(im, 'gray')
plt.title('Original Image')

#%% Fourier Transform

# Perform 2D Fourier transform
F = np.fft.fft2(im)  # Fast Fourier Transform
F_shifted = np.fft.fftshift(F)  # Shift so zero frequency is in the center

# Calculate the amplitude and phase
amplitude = np.abs(F_shifted)
phase = np.angle(F_shifted)

# Display amplitude as an image
plt.figure()
plt.title("Fourier Transform Amplitude (log scale)")
# Should maybe add an offset to avoid -Inf,
# but I've tested and there are no zeros.
plt.imshow(np.log(amplitude), cmap='gray') 
plt.colorbar()
plt.show()

# Display phase as an image
plt.figure()
plt.title("Fourier Transform Phase (radians)")
plt.imshow(phase)
plt.colorbar()
plt.show()

#%% Masking

# "Fundamental frequency" for the mask
f0 = 15 # I determined this "by hand"
# Full width of the mask -- should be an even number
df = 4

# Create a mask array
mask = np.ones((N, N))
for k in range(1, N//(2*f0)):
    center_f = N/2 + k*f0
    mask[:, int(center_f - df/2):int(center_f + df/2)] = 0
    center_f = N/2 - k*f0
    mask[:, int(center_f - df/2):int(center_f + df/2)] = 0

# Create a new amplitude array that is the original multiplied by this mask
new_amplitude = amplitude * mask

# Display the new amplitude as an image
plt.figure()
plt.title("Amplitude * Mask")
plt.imshow(np.log(new_amplitude + 0.1), cmap='gray') # + 0.1 because of zeros.
plt.colorbar()
plt.show()


# Combine new amplitude with original phase
new_F_shifted = new_amplitude * np.exp(1j * phase)

# Perform the inverse Fourier transform
new_F = np.fft.ifftshift(new_F_shifted)
new_im = np.fft.ifft2(new_F)
new_im = np.abs(new_im)

# Display the resulting image
plt.figure()
plt.title("Image based on Inverse FT")
plt.imshow(new_im, cmap='gray')
plt.colorbar()
plt.show()

    