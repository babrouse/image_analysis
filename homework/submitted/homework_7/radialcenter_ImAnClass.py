# -*- coding: utf-8 -*-
# radialcenter_ImAnClass.py
"""
Author:   Raghuveer Parthasarathy
Created on Mon Oct 31 13:33:05 2022
Last modified on Mon Oct 31 13:33:05 2022

Description
-----------

Particle localization by radial symmetry
Python translation of MATLAB radialcenter.m

** Version for Image Analysis Class**
Same as radialcenter.py , but with sigma and meand2 outputs removed, 
   and output positions returned relative to image center.

Uses 0 indexing of positions (unlike MATLAB)
NOTE: *Does not* optimize for image stacks (like radialcenter_stk.m);
      just single image

Copyright 2011-2022, Raghuveer Parthasarathy, The University of Oregon

Calculates the center of a 2D intensity distribution.
Method: Considers lines passing through each half-pixel point with slope
parallel to the gradient of the intensity at that point.  Considers the
distance of closest approach between these lines and the coordinate
origin, and determines (analytically) the origin that minimizes the
weighted sum of these distances-squared.
Applies simple smoothing if size > 3x3

Inputs
  I  : 2D intensity distribution (i.e. a grayscale image)
       Size need not be an odd number of pixels along each dimension

Outputs
  xc, yc : the center of radial symmetry, px, relative to image center
           Note that y increases with increasing row number (i.e. "downward")

To do:
    - Test more (like MATLAB version)
    - Faster grid creation than meshgrid? (like in MATLAB code)
    
see notes August 19-25, Sept. 9, Sept. 19-20 2011
Raghuveer Parthasarathy
The University of Oregon
August 21, 2011 (begun)

"""

import numpy as np   # Will assume numpy is already imported as np !


def radialcenter(I):
    # The main function -- see header comments for details
    
    (Ny, Nx) = I.shape
    # grid coordinates are -n:n, where Nx (or Ny) = 2*n+1
    # grid midpoint coordinates are -n+0.5:n-0.5
    xm, ym = np.meshgrid(np.arange(-(Nx-1)/2.0 + 0.5, (Nx-1)/2.0+0.5), 
                         np.arange(-(Ny-1)/2.0 + 0.5, (Ny-1)/2.0+0.5))
    # Calculate derivatives along 45-degree shifted coordinates (u and v)
    # Note that y increases "downward" (increasing row number) -- we'll deal
    # with this when calculating "m" below.
    dIdu = I[0:Ny-1,1:]-I[1:,0:Nx-1]
    dIdv = I[0:Ny-1,0:Nx-1]-I[1:,1:]
    # Smoothing -- perhaps should be optional
    fdu = dIdu # will overwrite if smoothing
    fdv = dIdv
    if np.min((Nx, Ny))>3:
        # Only smooth if image is >3px in the smallest dimension
        # Smooth by simple 3x3 boxcar, which I'll code directly rather than
        #    calling a convolution.
        # Zero-pad (expand by 1 on each side)
        dIdu_pad = np.zeros((Ny+1,Nx+1)) # dIdu array is size Ny-1, Nx-1
        dIdv_pad = np.zeros((Ny+1,Nx+1)) # dIdv array is size Ny-1, Nx-1
        dIdu_pad[1:Ny, 1:Nx] = dIdu
        dIdv_pad[1:Ny, 1:Nx] = dIdv
        fdu = np.zeros_like(dIdu)
        fdv = np.zeros_like(dIdv)
        for j in range(Ny-1):
            for k in range(Nx-1):
                fdu[j,k] = np.mean(dIdu_pad[j:j+3,k:k+3])
                fdv[j,k] = np.mean(dIdv_pad[j:j+3,k:k+3])
    dImag2 = fdu*fdu + fdv*fdv # gradient magnitude, squared
    
    # Slope of the gradient .  Note that we need a 45 degree rotation of 
    # the u,v components to express the slope in the x-y coordinate system.
    # The negative sign "flips" the array to account for y increasing
    # "downward"
    m = -(fdv + fdu) / (fdu-fdv)
    # Not smoothed version: m = -(dIdv + dIdu) ./ (dIdu-dIdv)
    
    infslope = 9e9 #replace infinite slope values with this extremely large number
    m[np.isinf(m)] = infslope
    
    # Shorthand "b", which also happens to be the
    # y intercept of the line of slope m that goes through each grid midpoint
    b = ym - m*xm
    
    # Weighting: weight by square of gradient magnitude and inverse 
    # distance to gradient intensity centroid.
    sdI2 = np.sum(dImag2)
    xcentroid = np.sum(dImag2*xm)/sdI2
    ycentroid = np.sum(dImag2*ym)/sdI2
    w  = dImag2/np.sqrt((xm-xcentroid)*(xm-xcentroid) + 
                        (ym-ycentroid)*(ym-ycentroid))
    
    # if the intensity is completely flat, m will be NaN (0/0)
    # give these points zero weight (and set m, b = 0 to avoid 0*NaN=NaN)
    w[np.isnan(m)]=0
    b[np.isnan(m)]=0
    m[np.isnan(m)]=0
    
    # least-squares minimization to determine the translated coordinate
    # system origin (xc, yc) such that lines y = mx+b have
    # the minimal total distance^2 to the origin:
    # Unilke the MATLAB version, where I have a separate function
    #   for this (lsradialcenterfit), I'll just write the calculation here:
    # Note m, b, w are defined on a grid;  w are the weights for each point
    wm2p1 = w/(m*m+1)
    sw  = np.sum(wm2p1)
    smmw = np.sum(m*m*wm2p1)
    smw  = np.sum(m*wm2p1)
    smbw = np.sum(m*b*wm2p1)
    sbw  = np.sum(b*wm2p1)
    det = smw*smw - smmw*sw
    xc = (smbw*sw - smw*sbw)/det # relative to image center
    yc = (smbw*smw - smmw*sbw)/det # relative to image center
    
    
    return xc, yc

