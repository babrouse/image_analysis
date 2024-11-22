# -*- coding: utf-8 -*-
# SVMexercise.py

"""
Author:   Raghuveer Parthasarathy
Created on Nov. 21, 2024
Last modified on Nov. 21, 2024


Description
-----------
Scripts for MLP / Neural Network exercise, Image Analysis in-class activity

"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from simPointSource import simPointSource

def make_Nparticle_image(M = 100, Nparticles = 2, position_range_um = 0.05,
                         N = 9 , scale = 0.1, fineScale = 0.01,
                         lam = 0.51, NA = 0.9, Nphoton = 1000 , bkgPoissMean = 10):
    """
    Simulate N particle image, with random positions in some range
    Sums single-particle images, each with Nphoton/Nparticles photons, and
        bkgPoissMean / Nparticles background
    Inputs:
        M = 100 # Number of images
        Nparticles = 2 # number of particles
        position_range_um = 0.05 : # range of true centers, microns
        Nphoton : number of photons, total
        bkgPoissMean = : mean background, total
        N = 9 # Size of the simulated image, NxN Pixels
        scale = 0.1 # camera scale, microns/px
        fineScale = scale / 20 # for high-res sim image
        lam = 0.51 # wavelength of light, microns
        NA = 0.9 # NA
    
    Returns
    -------
    im : NxNxM float array, simulated images

    """
    
    # True positions, random
    x0 = np.random.uniform(low=-1.0*position_range_um, high=position_range_um, 
                           size=(M,Nparticles)) # x-offset, microns
    y0 = np.random.uniform(low=-1.0*position_range_um, high=position_range_um, 
                           size=(M,Nparticles)) # y-offset, microns

    # Make images
    im =  np.zeros((N, N, M), dtype=float)
    for j in range(M):
        singleImage = np.zeros((N,N,Nparticles), dtype=float)
        for k in range(Nparticles):
            singleImage[:,:,k] = simPointSource(N, scale, fineScale = fineScale, 
                            NA=NA, lam=lam, Nphoton=Nphoton/Nparticles, 
                            xc=x0[j, k], yc=y0[j, k], 
                            bkgPoissMean=bkgPoissMean/Nparticles)
        im[:,:,j] = np.sum(singleImage, axis=2)
        
    return im
        
#%% Training data

# Simulated images
N = 9 # Size of the simulated image, NxN Pixels
scale = 0.1 # camera scale, microns/px
fineScale = scale / 20 # for high-res sim image
lam = 0.51 # wavelength of light, microns
NA = 0.9 # NA

Nphoton = 1000 # total number of photons
bkgPoissMean = 10

position_range_um = 0.2 # range of true centers, microns

M = 100 # number of training images
Nparticles = 1 # number of particles
im1 = make_Nparticle_image(M = M, Nparticles = Nparticles, 
                          position_range_um = position_range_um, N = N , 
                          scale = scale, fineScale = fineScale,
                          lam = lam, NA = NA, Nphoton = Nphoton, 
                          bkgPoissMean = bkgPoissMean)

Nparticles = 2 # number of particles
im2 = make_Nparticle_image(M = M, Nparticles = Nparticles, 
                          position_range_um = position_range_um, N = N , 
                          scale = scale, fineScale = fineScale,
                          lam = lam, NA = NA, Nphoton = Nphoton, 
                          bkgPoissMean = bkgPoissMean)

#%% Display some images

showImages = True
if showImages:
    plt.figure()
    plt.imshow(im1[:,:,0])
    plt.title('1 particle')
    
    plt.figure()
    plt.imshow(im2[:,:,0])
    plt.title('2 particles')


#%% Training the MLP model

im_train = np.concatenate((im1.transpose(2, 0, 1).reshape(M, -1),
                     im2.transpose(2, 0, 1).reshape(M, -1)), axis=0)

labels = np.concatenate((np.ones((M,)), 
                             2.0*np.ones((M,))), axis=0)

hidden_layer_sizes = (10) # number of neurons in each hidden layer
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, max_iter = 1000,
                    hidden_layer_sizes=hidden_layer_sizes)
mlp.fit(im_train, labels)



#%% Testing data

M_test = 100 # number of test images

Nparticles = 1 # number of particles
im1_test = make_Nparticle_image(M = M_test, Nparticles = Nparticles, 
                          position_range_um = position_range_um, N = N , 
                          scale = scale, fineScale = fineScale,
                          lam = lam, NA = NA, Nphoton = Nphoton, 
                          bkgPoissMean = bkgPoissMean)

Nparticles = 2 # number of particles
im2_test = make_Nparticle_image(M = M_test, Nparticles = Nparticles, 
                          position_range_um = position_range_um, N = N , 
                          scale = scale, fineScale = fineScale,
                          lam = lam, NA = NA, Nphoton = Nphoton, 
                          bkgPoissMean = bkgPoissMean)
im_test = np.concatenate((im1_test.transpose(2, 0, 1).reshape(M_test, -1),
                     im2_test.transpose(2, 0, 1).reshape(M_test, -1)), axis=0)

labels_test = np.concatenate((np.ones((M_test,)), 
                             2.0*np.ones((M_test,))), axis=0)

prediction = mlp.predict(im_test)

print(f'Accuracy: {100*np.sum(prediction==labels_test)/(2*M_test):.2f} %')