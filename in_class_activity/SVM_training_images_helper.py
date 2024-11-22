# -*- coding: utf-8 -*-
# SVM_training_images_helper.py
"""
Created on Sat Nov. 19, 2022
Last modified on Nov. 20, 2024

Description
-----------
Helper code for SVM exercise, Image Analysis Homework 9 or in-class activity

"""

import numpy as np
import matplotlib.pyplot as plt


#%% Rescaling features 
# for SVM to work well, we need all features to be centered around zero
#   and have similar variances; rescale accordingly
# There are built-in functions for this, but let's do it manually...
def rescale_features(features):
    # features = array of features; each column is a feature
    f_mean = np.mean(features, axis=0)
    f_std = np.std(features, axis=0)
    f_rescaled = (features-f_mean)/f_std
    return f_rescaled


#%% A function for clicking and identifying regions

def clickAndGetIndexes(im, stats, N_pts=10, titleStr='', regionColor='g'):
    # A function for clicking points on an image and assigning
    #   each clicked point to a region; colors selected region
    # Calls closestRegionIndex function
    # Inputs
    #   im : Image to display
    #   stats : list of region properties, previously calculated
    #   N_pts : number of points the user will click
    #   titleStr : for the plot title, e.g. 'Colonies'
    #   regionColor : color, for displaying the region
    # Outputs
    #   selected_pts : array of coordinate pairs, dimensions (N_pts, 2)
    #   selected_idx: index of regions (corresponding to stats) selected
    plt.figure()
    plt.imshow(im, 'gray')
    plt.title(f'After key press, will select {N_pts} regions: {titleStr}')
    print('Press a key (on the keyboard) after making the window larger; then see the title.')
    # Without this, resizing leads to false "clicks"
    zooming = False
    while not zooming:
        zooming = plt.waitforbuttonpress()
    print('\nReady to select objects.\n')
    plt.title('First point (discard)')
    this_point = plt.ginput(1)[0]  # a list, so keep only 1st element
    # Will select each point individually and display the region
    selected_pts = np.zeros((N_pts, 2))
    selected_idx = np.zeros((N_pts, )).astype('uint32')
    for j in range(N_pts):
        plt.title(f'Select {N_pts} regions: {titleStr}, point no. {j+1}')
        this_point = plt.ginput(1)[0]  # a list, so keep only 1st element
        selected_pts[j,:] = np.array(this_point)
        # Find the closest region to this point
        # This makes no sense, but having the output of closestRegionIndex
        #   go to a single-element variable rather than simply
        #   selected_idx[j] seems neccesary ; otherwise, it fails *some* of 
        #   the time ??!
        sI = closestRegionIndex(stats, this_point)
        selected_idx[j] = sI
        # display (an inelegant method of plotting all the points in the region!)
        # Oddly, this doesn't always update!
        coords_array = np.array(stats[selected_idx[j]]['coords'])
        plt.scatter(coords_array[:,1], coords_array[:,0], 
                    s=2, c=regionColor, marker='.')
        y, x = (stats[selected_idx[j]]['Centroid'])
        plt.scatter(x, y, s=10, c='y', marker='x')
        plt.show()
    return selected_pts, selected_idx


#%% A function for finding the region that corresponds to a point

def closestRegionIndex(stats, coord_pair):
    # A function that returns the index of the region containing a given
    # (x, y) tuple, or the closest region if there is no region containing (x, y)
    # Inupts: 
    #    stats is a list of regions from skmeasure.regionprops
    #    coord_pair is a 1,2 array of (x, y) coordinates, e.g. from ginput
    idx = np.NaN # index initialized to be not a number
    for j in range(len(stats)):
        # reverse tuple, to match x, y and row, column
        flip_coord_pair = (np.round(coord_pair[1]), np.round(coord_pair[0]))
        # Check if *both* coordinates are in the pixel list for this region
        # There's probably a nicer way to do this
        # The following doesn't work, as it's true if *either* element is present:
            # if flip_coord_pair in stats[j]['coords']:
        coord_array = np.array(stats[j]['coords'])
        # Note for below: [0] needed as output for np.where, to ignore the rest of the tuple
        foundIndex = np.where(np.logical_and(coord_array[:,0]==flip_coord_pair[0], 
                                             coord_array[:,1]==flip_coord_pair[1]))[0]
        if len(foundIndex)>0:
            idx = j
            print(f'Index {j}, coordinates {coord_pair[0]:.2f}, {coord_pair[1]:.2f}')
    if np.isnan(idx):
        print('No exact match; Finding closest region.')
        d_min = 9e99 # placeholder for the minimal distance found so far
        for j in range(len(stats)):
            # no region was found; look for closest region
            y, x = (stats[j]['Centroid'])  # note: row, column!
            d = np.sqrt((x - coord_pair[0])**2 + (y - coord_pair[1])**2)
            if d < d_min:
                d_min = d
                idx = j
        print(f'Closest Index: {idx}')
    return idx

def plotObjectLabels(features, labels, predictions, 
                     colony_color = np.array([0.2, 0.7, 0.6]),
                     not_colony_color = np.array([0.8, 0.5, 0.2]),
                     titleStr = 'Data', xLabelStr  = 'Area', 
                     yLabelStr  = 'Eccentricity'):
    # Plot objects in feature space, with different colors for the two labels; 
    # indicate correct and incorrect predictions
    # Inputs
    #    features: N x 2 array of features, rescaled
    #    labels: N x 1 array of true labels
    #    predictions: : N x 1 array of predictions
    #    colony_color : 1 x 3 array RGB color to use for colonies 
    #    not_colony_color : 1 x 3 array RGB color to use for not-colonies 
    #    titleStr : title string
    #    xLabelStr, yLabelStr : labels for x, y axes
    
    plt.figure()
    
    # Colonies, correctly labeled
    col_correct_ind = np.logical_and(labels.ravel()==1, predictions == labels.ravel())
    plt.semilogx(features[col_correct_ind, 0], features[col_correct_ind, 1], 
                 marker='o', linestyle = 'none', c=colony_color, 
                 label='Colonies, correct')
    # Colonies, incorrectly labeled
    col_incorrect_ind = np.logical_and(labels.ravel()==1, predictions != labels.ravel())
    plt.semilogx(features[col_incorrect_ind, 0], features[col_incorrect_ind, 1], 
                 marker='x', linestyle = 'none', c=colony_color, 
                 label='Colonies, incorrect')
    # Not-colonies, correctly labeled
    not_col_correct_ind = np.logical_and(labels.ravel()==0, predictions == labels.ravel())
    plt.semilogx(features[not_col_correct_ind, 0], features[not_col_correct_ind, 1], 
                 marker='o', linestyle = 'none', c=not_colony_color, 
                 label='Not Colonies, correct')
    # Not-colonies, incorrectly labeled
    not_col_correct_ind = np.logical_and(labels.ravel()==0, predictions != labels.ravel())
    plt.semilogx(features[not_col_correct_ind, 0], features[not_col_correct_ind, 1], 
                 marker='x', linestyle = 'none', c=not_colony_color, 
                 label='Not Colonies, incorrect')
    plt.xlabel(xLabelStr)
    plt.ylabel(yLabelStr)
    plt.title(titleStr)
    plt.legend()
