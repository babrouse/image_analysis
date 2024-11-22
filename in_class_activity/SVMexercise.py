# -*- coding: utf-8 -*-
# SVM_exercise.py
"""
Author:   Raghuveer Parthasarathy
Created on Sat Nov. 19, 2022
Last modified on Nov. 21, 2024


Description
-----------
Scripts for SVM exercise, Image Analysis in-class activity

"""

import numpy as np
import matplotlib.pyplot as plt
import os

from skimage import io # input output sub-package 
from skimage import filters as skfilters # filtering sub-package 
from skimage import measure as skmeasure
from skimage.morphology import disk, closing

from sklearn.svm import SVC # support vector machine classifier

from SVM_training_images_helper import (rescale_features,
                                        clickAndGetIndexes, plotObjectLabels)

#%% Load and process image

plt.figure(dpi=300)

def load_and_process_image(parentDir, fileName = r'dsc_0357.jpg'):
    # pasted from HW8 Segmentation by Thresholding, with slight modifications.
    # Loads image, bandpass filters, thresholds, and performs morphological closing
    # Also displays image
    # Returns:
    #   im_bw : thresholded (binary) image
    #   im : original image
    
    im = io.imread(os.path.join(parentDir, fileName))
    # plate_IMG_20211222.png (if using this, invert, im = 255-im  # Invert
    im = np.mean(im,2) # make grayscale
    
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title('Original image (gray)')
    
    # Parameters (filter sizes)
    HPfiltersize = 50; # for high pass filtering (removing "slow" intensity variation)
    LPfiltersize = 4;  # for low pass filtering (smoothing over noise)
    
    # High-pass filter 
    # Note that if im is float, the HP filter is *not* scaled to 0-1, so shouldn't
    #    rescale here. Or check to be sure
    im_Gaussfilt = skfilters.gaussian(im, HPfiltersize, mode = 'nearest')
    if np.max(im_Gaussfilt) <= 1.0:
        im_Gaussfilt *= 255.0
        
    im_hp = im - im_Gaussfilt;
    
    # Low-pass filter (smoothing)
    im_lp = skfilters.gaussian(im_hp, LPfiltersize, mode = 'nearest')

    # Threshold the filtered image
    # Otsu's method for determining the optimal level
    thresh_Otsu_Gauss = skfilters.threshold_otsu(im_lp)
    print(f'Threshold level: {thresh_Otsu_Gauss:.3f}')
    im_bw = im_lp > thresh_Otsu_Gauss
    
    # Morphological closing
    im_bw = closing(im_bw, disk(2))
    plt.figure()
    plt.imshow(im_bw, cmap='gray')
    plt.title('Thresholded and closed')
    
    return im_bw, im

def getFeatures(im_bw, stats, N_pts = 15, im = None, imTitle = 'Image'):
    # user-selection of test dataset 
    # redundant code with above -- should have made one function
    # Inputs: 
    #   im_bw : binary image
    #   stats : object stats
    #   N_pts = 15 # number of objects to select
    #   im : image to display, if not None
    #   imTitle : title for image
    # Outputs
    #   f_rescaled : features array, rescaled and normalized
    #   features : features array, not rescaled
    
    # Select testing points, colony and not-colony
    colony_pts, colony_idx = clickAndGetIndexes(im_bw, stats, N_pts, 
                                                titleStr='Colonies', 
                                                regionColor='g')
    
    not_colony_pts, not_colony_idx = clickAndGetIndexes(im_bw, stats, N_pts,
                                                        titleStr='Not Colonies', 
                                                        regionColor='r')
    
    # Put all the features into one array
    # rows = objects; columns = features (area, eccentricity)
    features_colonies = np.stack((np.array([stats[j]['Area'] for j in colony_idx]), 
                                  np.array([stats[j]['Eccentricity'] for j in colony_idx])),
                                 axis=1)
    features_not_colonies = np.stack((np.array([stats[j]['Area'] for j in not_colony_idx]), 
                                  np.array([stats[j]['Eccentricity'] for j in not_colony_idx])),
                                 axis=1)
    features = np.concatenate((features_colonies, features_not_colonies), axis=0)
    
    f_rescaled = rescale_features(features)

    if im is not None:
        # Display the selected regions on the original (inverted) image
        plt.figure()
        plt.imshow(im, cmap = 'gray')
        for j in range(N_training_pts):
            coords_array_colonies = np.array(stats[colony_idx[j]]['coords'])
            plt.scatter(coords_array_colonies[:,1], coords_array_colonies[:,0], 
                        s=2, c='g')
            coords_array_not_colonies = np.array(stats[not_colony_idx[j]]['coords'])
            plt.scatter(coords_array_not_colonies[:,1], coords_array_not_colonies[:,0], 
                        s=2, c='r')
        plt.title(imTitle)
    
    return f_rescaled, features
    
#%% -------------------------

if __name__ == '__main__':
      
        
    #%% Directory
    
    
    parentDir = r'C:\Users\bretb\Documents\school\image_analysis\in_class_activity'
    fileName = r'dsc_0357.jpg'
    im_bw, im = load_and_process_image(parentDir, fileName = r'dsc_0357.jpg')
    
    #%% Misc figure code, if necessary
    
    # For getting the figure clicking to work -- suggest to others if necessary
    # import matplotlib
    # matplotlib.use('TkAgg')
    
    # Default figure size, since reshaping can cause problems.
    # strange, makes it slow
    # width, height in inches, at 100 dpi (default)
    # plt.rcParams["figure.figsize"] = (8, 6)
    
    
    #%% Region properties
    label_img = skmeasure.label(im_bw)
    stats = skmeasure.regionprops(label_img)
    # e.g., stats[j]['Area'] is the area of object j
    
    #%% SVM! Training regions -- clicking
    
    # Select training points, colony and not-colony
    # Display the selected regions on the original (inverted) image
    # Create a feature array, and an array of labels
    

    N_training_pts = 8
    f_rescaled, features =  getFeatures(im_bw, stats, N_pts = N_training_pts, 
                                        im = None, 
                                        imTitle = 'Training dataset')
    
    #%% Labels: ones for the colonies, zeros for the not-colonies
    labels = np.concatenate((np.ones((N_training_pts,)), 
                             np.zeros((N_training_pts,))), axis=0)
    
    
    #%% SVM: learning (training)
    
    C = 1.0 # "regularization parameter" which = soft margin parameter (1.0 is default)
    model = SVC(kernel='poly', degree=3, C = C)  # degree irrelevant for rbf
    model.fit(f_rescaled, labels)
    
    predictions = model.predict(f_rescaled)
    print(f'\n\nTraining accuracy: {np.mean(predictions == labels)*100:.2f}%')

    #%% Plot training results
    plotObjectLabels(features, labels, predictions, 
                         colony_color = np.array([0.2, 0.7, 0.6]),
                         not_colony_color = np.array([0.8, 0.5, 0.2]),
                         titleStr = 'Training Data', xLabelStr  = 'Area', 
                         yLabelStr  = 'Eccentricity')


    #%% Testing! Pick a new set of points, and apply the trained model
    
    performTest = True
    if performTest:
        N_testing_pts = 15 # number of test objects to select
        test_f_rescaled, test_features = getFeatures(im_bw, stats, 
                                                     N_pts = N_testing_pts)
        
        # Testing accuracy, on the trained model
        test_predictions = model.predict(test_f_rescaled)
        # labels: ones for the colonies, zeros for the not-colonies
        test_labels = np.concatenate((np.ones((N_testing_pts,)), 
                                 np.zeros((N_testing_pts,))), axis=0)
        print(f'Testing accuracy (!): {np.mean(test_predictions == test_labels)*100:.2f}%')
    
        # Plot test results
        plotObjectLabels(test_features, test_labels, test_predictions, 
                             colony_color = np.array([0.2, 0.7, 0.6]),
                             not_colony_color = np.array([0.8, 0.5, 0.2]),
                             titleStr = 'Test Data', xLabelStr  = 'Area', 
                             yLabelStr  = 'Eccentricity')

