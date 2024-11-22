# -*- coding: utf-8 -*-
# simPointSource.py

"""
Author:   Raghuveer Parthasarathy
Last modified on Nov. 21, 2024

Description
-----------

Simulated image function, extracted from particle_localization_functions.py

   simPointSource()
   calc_RMSE()

"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as scipy_special  # scipy is too big to import all of it.



#%% Simulated image function

def simPointSource(N, scale=0.1, fineScale = 0.01, NA = 0.9, 
                   lam = 0.53, Nphoton = 1000, xc = 0, yc = 0, 
                   bkgPoissMean=0, makeFigures=False):
    """
    Function to create a simulated point source image: PSF, pixelation, noise
    Returns a square 2D array, NxN, normalized to sum==Nphoton.
    Inputs:
       N : final camera image array size (N x N pixels)
       scale : final camera scale, um/px
       fineScale: microns / pixel for high-resolution PSF
       NA : numerical aperture
       lam : free-space wavelength of light, *microns*
       Nphoton : number of photons, total; rescale output to sum to this
       xc, yc : point source center location, *microns*
       bkgPoissMean : mean per-pixel background intensity
       makeFigures : if true, show fine PSF
     Output: 
       simImage : NxN integer array, simulated image, photons at each px.
   """
    scaleMultiplier = scale / fineScale
    # Check that it's an integer multiple
    if np.abs(scaleMultiplier % 1) > 0.01:
        print('Warning! camera scale is not an integer multiple of the fine scale')
        print(f'Original scaleMultiplier: {scaleMultiplier:.3e}')
        fineScale = scale / np.round(scaleMultiplier)
        print(f'Changing fineScale to be {fineScale:.3e}')
    else:
        scaleMultiplier = np.round(scaleMultiplier)
    
    # High-resolution PSF
    # Could call my PSF function, but I'll make this self-contained.
    N_fine = int(N*scaleMultiplier) # pixels for the fine array
    # High-resolution PSF
    x1D = fineScale*np.arange(-(N_fine-1)/2,(N_fine+1)/2) - xc
    y1D = fineScale*np.arange(-(N_fine-1)/2,(N_fine+1)/2) - yc
    x, y = np.meshgrid(x1D, y1D)  # grid positions, microns
    rpositions = np.sqrt(x*x + y*y);  # matrix of distances from the center, in microns
    v = (2*np.pi/lam)*NA*rpositions;  # array of 'v' values    
    psf = 4*(scipy_special.j1(v)/v)**2
    psf[int((N_fine-1)/2), int((N_fine-1)/2)]=1 # Correct the undefined central value
    if makeFigures:
        plt.figure()
        plt.imshow(psf, cmap='gray')
        plt.title('Fine resolution')
    
    # Pixelate -- will loop, rather than doing something elegant; N isn't large, anyway.
    simImage = np.zeros((N, N))
    for j in range(N):
        for k in range(N):
            j_range = np.arange(j*scaleMultiplier, (j+1)*scaleMultiplier).astype(int)
            k_range = np.arange(k*scaleMultiplier, (k+1)*scaleMultiplier).astype(int)
            simImage[j,k] = np.sum(psf[j_range, k_range])    
    
    # Rescale so that the total intensity is Nphoton
    simImage = Nphoton*simImage / np.sum(simImage)
    
    # Photon noise
    # Draw the intensity values from a Poisson distribution for each pixel
    # Note: integer values
    simImage = np.random.poisson(simImage, simImage.shape); 
    
    # Add Poisson background
    if bkgPoissMean > 0:
        im_bkg = np.random.poisson(bkgPoissMean, simImage.shape) # Background
        simImage += im_bkg

    return simImage
