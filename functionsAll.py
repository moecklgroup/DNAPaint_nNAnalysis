# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:26:52 2024

@author: Admin
"""


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
import statistics as stat
import random
import math
from astropy.stats import RipleysKEstimator
import scipy.spatial as spat


import glob 
from pathlib import Path    




#%% clusteringDBSCAN


def clusteringDBSCAN(X, epsChoice, minSampleChoice):
    
    """
    Finds clusters in data from coordinates of points 
    
    Prints the number of identifies clusters and the number of outlier points
    
    Input: 
        
        X: Coordinates of points to cluster (2D array, float)
        
        epsChoice: Max distance to a point for the other to be considered its neighbor 
        ~ max radius of cluster (float)
        
        minSampleChoice: Min neighbors for a point to be a core point of cluster 
        ~ min number of points in a cluster (int)
    
    Output:
        
        labels: Labels of points (=cluster they belong to ; -1 means noise) (1D array, int)
    """
    
    db = DBSCAN(eps=epsChoice, min_samples=minSampleChoice).fit(X)

    labels = db.labels_ 
    # one number for each different cluster; -1 for noise

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) 
    #number of cluster without counting the noisy points (label = -1)
    n_noise_ = list(labels).count(-1) 
    #count number of noise points (labels = -1)


    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    
    return labels




def displayPointsSize(dictFunct, x, y):
    
    """
    Displays points as scatter from array of coordinates 

    Input:
        
        dictFunct: dictionary of arrays of points to display (dict of 2D array)
        
        x and y: dimensions of the figure (float)

    Output: 
        
        display
    """
    
    if len(dictFunct) == 1: #if only one figure to plot
        
        X = list(dictFunct.values())[0] # array in position 0 in the dictionary
        plt.figure(figsize=(x,y)) # size of the figure (user input in call of function)
        plt.plot(X[:, 0], X[:, 1], '.', markersize=1)
        plt.title(list(dictFunct.keys())[0], fontsize=x) #title of figure => font size propotional to fig size
        
        
    else: #if multiple figures to plot 
        
        for i in dictFunct.keys(): #i is the array for which we are tracing the histogramm

            plt.figure(figsize=(x,y)) # size of the figure (user input in call of function)
            
            X = dictFunct[i] #i-eme array in dictionary 
            plt.plot(X[:, 0], X[:, 1], '.', markersize=1)
            plt.title(i, fontsize=x) #title of figure => font size propotional to fig size
    
    plt.show()
    
    
    

def displayPointsCentroids(dictFunct, dictCentroids, x, y, title):
    
    """
    Display points and centroids of clusters from 2D arrays of coordinates
    
    Input:

        dictFunct: dictionary of arrays of points to display - raw data (dict of 2D array)
        
        dictCentroids: dictionary of arrays of points to display - centroids (dict of 2D array)

        x, y: dimensions of the figure (float)
        
        title: title of the figures 
        (1D array of size = len(dictionaries) or string if 1 element in dict)

    Output:

        display
    """
    
    if len(dictFunct) == 1:  #if only one figure to plot
        
        X = list(dictFunct.values())[0] #points to plot
        C = list(dictCentroids.values())[0] #centroids for the points to plot 
        plt.figure(figsize=(x,y))# array in position 0 in the dictionary
        plt.plot(X[:, 0], X[:, 1], '.', markersize=2, label='points ' + list(dictFunct.keys())[0]) #points 
        plt.plot(C[:, 0], C[:, 1], 'o', color='red', markersize=20, 
                 markerfacecolor="none", markeredgewidth=3, 
                 label='controids ' + list(dictCentroids.keys())[0]) #centroids
        
        plt.title(title, fontsize=x) 
        #title of figure => font size propotional to fig size
        plt.legend(fontsize='x-large', loc='upper right')

        
        
        
    else: #if multiple figures to plot 
        
        for i in range(0, len(dictFunct)): #i is the array for which we are tracing the histogramm
            X = dictFunct[list(dictFunct)[i]] #i-eme array in dictionary 
            C = dictCentroids[list(dictCentroids)[i]] #i-eme array in dictionary 
            
            plt.figure(figsize=(x,y))# array in position 0 in the dictionary

            plt.plot(X[:, 0], X[:, 1], '.', markersize=2, label='points ' + list(dictFunct.keys())[i]) #points
            plt.plot(C[:, 0], C[:, 1], 'o', color='red', markersize=20, 
                     markerfacecolor="none", markeredgewidth=3, 
                     label='controids ' + list(dictCentroids.keys())[i]) #centroids 
            
            plt.title(title[i], fontsize=x)
            #title of figure => font size propotional to fig size
            plt.legend(fontsize='x-large', loc='upper right')


    plt.show()
    
    
    
    
def displayHistFigure(dictFunct, rangeUp, binsize, path):
    
    
    """
    Displays histograms of distance to NN, all in the same figure

    Save histogram as png
    
    Input: 

        dictFunct: Dictionary of arrays of distance to NN (dict of 1D arrays)

        rangeUp: 0-rangeUp for histogram display (float)
        
        binsize: Size of histogram bins (float)
        
        path: Path to folder where the figures will be saved 

    Output:

        Display histogram ; png
    """
    
    plt.figure(figsize=(15,5)) # size of the figure 
    
    if len(dictFunct) == 1: #if one figure to plot 
        
        y, x, _ = plt.hist(list(dictFunct.values())[0], 
                           bins=math.ceil(math.ceil(max(list(dictFunct.values())[0]))/binsize), 
                           histtype='step', 
                           range = (0, math.ceil(max(list(dictFunct.values())[0]))),
                           label=list(dictFunct.keys())[0])
        #the number of bins is the max value divided by the size of the bins (rounded up)
        #the range insure that the bins have the same edges throughout the channels
        #   it goes from 0 to the right edge of the last bin
        #the label for each histogram is it's name in the dictionary
        
        xmax = x[np.argmax(y)] #x value for the highest bin (max occurence)
        ymax = y.max() #y (number of occurences value for the highest bin
        plt.annotate("x={:.3f}, y={:.3f}".format(xmax, ymax), xy=(xmax, ymax))        
        
    else: #if multiple figures to plot (in the same figure and same subplot)
    
        for i in dictFunct.keys(): #i is the array for which we are tracing the histogramm

            y, x, _  = plt.hist(dictFunct[i], 
                                bins=math.ceil(math.ceil(max(dictFunct[i]))/binsize), 
                                histtype='step', 
                                range = (0, math.ceil(max(dictFunct[i]))),
                                label=i) 
            #the number of bins is the max value divided by the size of the bins (rounded up)
            #the range insure that the bins have the same edges throughout the channels
            #   it goes from 0 to the right edge of the last bin
            #the label for each histogram is it's name in the dictionary
            
            #print max x and count on the histogram 
            xmax = x[np.argmax(y)] #x value for the highest bin (max occurence)
            ymax = y.max() #y (number of occurences value for the highest bin
            plt.annotate("x={:.3f}, y={:.3f}".format(xmax, ymax), xy=(xmax, ymax))

    plt.legend(fontsize='x-large', loc='upper right')
    plt.xlabel('Distance to nearest neighbor (nm)')
    plt.ylabel('Count')
    plt.xlim((0,rangeUp)) #x lim of the histogram display
    
    
    plt.savefig(path+'.png') #save figure as png in specified path

    


def calculateCentroids(labels, X):
    
    """
    Calculate centroids of identified clusters
    
    Input:

        labels: labels of identified clusters and all points  (array of int)

        X: coordinates of points (2D array of float)

    Output:

        centroids: containing centroids of clusters (2D array of float)
    """
    
    unique_labels = set(labels) #different clusters labels found by DBSCAN clustering
    
    centroids = np.ones((len(unique_labels)-(1 if (-1 in unique_labels) == True else 0) , 2)) #create array of coordinates (2 columns) and (nb of clusters) lines
    #if there are noise points, len = len-1 sice we do not count them as cluster
    #else, stays len = len

    for k in unique_labels: 
        if k != -1: # labels = -1 means the point doesn't belong to a cluster (outlier) => no need for centroid
            points_of_cluster = X[labels==k] # select point of one cluster
            centroids[k] = np.mean(points_of_cluster, axis=0) # centroid of the selected cluster 
    
    return centroids # return the coordinates of the centroids of the clusters as an array




def nearestNeighborsClean(X, Y): 
    
    """
    Search in X the NN for each point of Y 
    
    Input:

        X: coordinates of points pool of possible NN (2D array of float)

        Y: coordinates of points in search of NN (2D array of float)

    Output:

        distances: distance to NN for all points of Y in X (1D array of float)
    """
    
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X) #basic the array in which we search the NN
    #neighbors=2 bcs neighbors=1 is just the point
    distances, indices = nbrs.kneighbors(Y) #search the NN of the points in Y in X
    
    if np.array_equal(X,Y): #if the two arrays X and Y are the same
        return distances[:, 1] #return the second column of distances (true NN and not 0 for point itself)
    else: 
        return distances[:, 0] #return first column, no problem
    
    
    

def nearestNeighborsMult(X,Y, nbNeighbors): 
    
    """
    Search in X the [chosen number+1] NN for each point of Y 
    
    And gives back the distance to the chosen number of nearset neighbours
    
    Input:

        X: coordinates of points pool of possible NN (2D array of float)

        Y: coordinated of points in search of NN (2D array of float)

        nbNeighbors: number of neighbors to search for for each point of Y (int)

    Output:

        distances: distance to NN for all points of Y (in X) (nbNeighborsD array of float)
    """
    
    nbrs = NearestNeighbors(n_neighbors=nbNeighbors+1, algorithm='auto').fit(X) 
    #basic the array in which we search the NN
    #neighbors=2 bcs neighbors=1 is just the point
    distances, indices = nbrs.kneighbors(Y) #search the NN of the points in Y in X
    
    if np.array_equal(X,Y): #if the two arrays X and Y are the same
        return distances[:, 1:nbNeighbors+1] 
    #return the second column of distances (true NN and not 0 for point itself)
    else: 
        return distances[:, 0:nbNeighbors] #return first column, no problem        
    
    


def nearestNeighborsOneInAll(dictPoints,dictLocs): 
    
    """
    Ssearch in X the [chosen number+1] NN for each point of Y 
    
    Input:

        dictLocs: dictionary of arrays => coordinates of points pool of possible NN (2D array of float)

        dictPoints: dictionary of 1 array => coordinates of points in search of NN (2D array of float)

    Output:

        dictDist: distance to NN for all points of Y (in X) (nbNeighborsD array of float)
    """
    
    dictDist = {} #future dictionary of distances to NN for combinaisons of arrays 
    
    for i in dictLocs.keys():
        
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(dictLocs[i]) 
        #basic the array in which we search the NN
        #neighbors=2 bcs neighbors=1 is just the point if we search in the same array 
    
        distances, indices = nbrs.kneighbors(list(dictPoints.values())[0]) 
        #search the NN of the points in dictPoints inside ieme array of dictLocs
        
        # WARNING : THE KEYS HAVE TO BE THE SAME IF THE ARRAYS ARE THE SAME
        if list(dictPoints.keys())[0] == i: #if the two arrays X and Y are the same - by name
            dictDist['distNN_'+list(dictPoints.keys())[0]+'_in_'+i] = distances[:, 1] 
        #second column of distances (true NN and not 0 for point itself if same array)
        else: 
            dictDist['distNN_'+list(dictPoints.keys())[0]+'_in_'+i] = distances[:, 0] 
        #first column, no problem

    return dictDist



    
def dictionaryToCsv(dictFunct, nameCsv): 
    
    """
    Makes dateframe from arrays in dictionary and converts the dataframe into csv file

    Ajust dimentions of arrays if necessary
    
    Input:
 
        dictFunct: dictionary to save as csv (dict)
        
        nameCsv: name of the new csv file (string), including or not 
        the path to a specific folder in which it will be saved

    Output:

        csv file
    """
    
    maxLength = max(len(dictFunct[i]) for i in dictFunct.keys())
    # length of the biggest array in the dictionary 
    
    dfDistances = pd.DataFrame()
    
    for i in dictFunct.keys():
        #for each of the channels in the dictionary
        dictFunct[i] = np.append(dictFunct[i], np.repeat(-1, maxLength-len(dictFunct[i])))
        #add -1 distances to fill the empty spaces of the arrays => put everything to the same length
        dfDistances.insert(len(dfDistances.columns), i, dictFunct[i])
        #add to the dataframe

    dfDistances.to_csv(nameCsv) #save dataframe as csv
    
    


def alpha_shape(points, alpha, only_outer=True):
    """
    
    Compute the alpha shape (concave hull) of a set of points.
    
    :param points: np.array of shape (n,2) points.
    
    :param alpha: alpha value.
    
    :param only_outer: boolean value to specify if we keep only the outer border
    
    or also inner edges.
    
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    
    the indices in the points array.
    """
    
    assert points.shape[0] > 3, "Need at least four points"
    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))
    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges




def partOfYourWorld(x, y, indexEdges, points): 
    """
    Tells if point is inside or outside delimited space (cell)
    
    Input: 

        x: x coordinate of the point (float)

        y: y coordinate of the point (float)

        indexEdges: index of points forming part of the edge of cell (2D array)

        points: points of the cell (inside and outside) (2D array)

    Output: 

        TRUE (~inside) or FALSE (~outisde) (bool)
    """    

    xEdgesInf = np.empty((50,2), int) ; inf = 0 #edges below the point (xaxis)
    xEdgesSup = np.empty((50,2), int) ; sup = 0 #edges above the point (xaxis)
    
    for i, j in indexEdges: #for each segment composing the edge
        
        #x axis
        if min(points[i, 0], points[j, 0]) <= x < max(points[i, 0], points[j, 0]):
            #if the point is between the two ends of the segment
            if stat.mean([points[i, 1], points[j, 1]]) <= y:
                #if the point is below the segment (yaxis)
                xEdgesInf[inf] = [i, j]
                #add the segemnt to the array of segments below the point (xaxis)
                inf = inf+1
            else:
                xEdgesSup[inf] = [i, j]
                # = above = 
                sup = sup+1
  
    xEdgesInf = np.delete(xEdgesInf, np.s_[inf:len(xEdgesInf)], 0)
    xEdgesSup = np.delete(xEdgesSup, np.s_[inf:len(xEdgesSup)], 0)

    
    if ((len(xEdgesInf)%2 == 0) or (len(xEdgesSup)%2 == 0)):
        #if there is an even number of edges on any side of the point
        #then the point is outside the cell
        return False
    else:
        #if there is an odd number of edges on all sides of the point
        #then it is inside the cell 
        return True
        



def randomPartOfYourWorld(indexEdges, points):
    
    """
    Draws random point, check if it is inside the cell, and keep it if it is
    
    Repeat until there is the same number of random points and experiemental points inside the figure
    
    Input: 

        indexEdges: couples of indexes in 'points' that are part of the edge of cell

        points: experimental points inside the cell 

    Output:

        randomPoints: array of random points inside the cell (same number than exp points)
    """
    
    #parameters for random.randrange => ranges for coordinates of random point
    xmin = math.ceil(min(points[:, 0])) ; xmax = math.ceil(max(points[:, 0]))
    ymin = math.ceil(min(points[:, 1])) ; ymax = math.ceil(max(points[:, 1]))

    #empty array for the random points inside the cell 
    randomPoints = np.empty((len(points),2), float)
    
    #random draws outside the loop to reduce running time (a bit)
    hold = np.empty((len(points)*4, 2), float)
    for i in range(0, len(hold)):
        hold[i] = [random.randrange(xmin*100, xmax*100)/100, 
                random.randrange(ymin*100, ymax*100)/100]
        
    count = -1
    
    for i in tqdm(range(0, len(points))): 
    
        valBool = False
        #hold = np.empty((0,2), float)
    
        while valBool != True:
            
            count = count+1
            
            valBool = partOfYourWorld(hold[count,0], hold[count,1], indexEdges, points)
            #valBool = True if the point is inside the cell
        
        randomPoints[i] = [hold[count,0], hold[count,1]]
        #add the point to the array of random points inside the cell 
        
        if count == len(hold)-1: 
            print('not enough random points initially drawn')
            print('count of random points inside the shape : %d / %d' % (i, len(points)))
            break
        
        continue
            
    return randomPoints
    



def randomDistributionAll(points, alphaParameter):
    
    """
    Generate an array of random points inside a cell, the number of random and experimental points is equal
    
    Input:
 
        points: coordinates of experimental points in a cell 
        
        alphaParameter : parameter for edges of alpha shape (depends on the shape precision wanted)
    
    Output:
        
        randomPoints: array of random points inside a cell, the number of random and experimental points is equal
    """
    
    edges = alpha_shape(points, alpha=alphaParameter, only_outer=True) #alphaParameter = 700 is nice for now 
    indexEdges = np.array(list(edges)) #convertion of edges from list to np array
    
    randomPoints = randomPartOfYourWorld(indexEdges, points)
    #array of random points inside a cell, the number of random and experimental points is equal
    
    #display figure with : 
        #edges of experimetal cell for the alphaParameter
        #random points generated 
    plt.figure(figsize=(50,50))
    plt.plot(randomPoints[:, 0], randomPoints[:, 1], '.')
    for i, j in edges:
        plt.plot(points[[i, j], 0], points[[i, j], 1], color='black')
    plt.show()
    
    return randomPoints




def ripleyParametersForClustering(array, cuts):
    
    """
    Returns the mean of the following values for the areas defined by cuts: 
    
    Determinate the mean size of clusters with ripley's H function 
    (maximum is the radius of clusters for maximum clustering )
    
    Input:

        array: points to cluster (1 channel) (2D array)

        cuts: coordinates of start of area on which to apply ripley's function (2D array)

    Output:
    
        stat.mean(radius): mean size of clusters with ripley's H function 
    """
    
    radius = []
    
    #for each of the areas defined in cuts
    for i, j in tqdm(cuts):
   
        #points of the array in the area (1500x1500 nm)
        dataTest = array[(array[:,0] >= i) & (array[:,0] <= i+1500) & 
                         (array[:,1] >= j) & (array[:,1] <= j+1500)]
    
        #definition of the study area parameter for ripley's function
        x_minTest = min(dataTest[:,0]); x_maxTest = max(dataTest[:,0])
        y_minTest = min(dataTest[:,1]); y_maxTest = max(dataTest[:,1])
        areaTest = (x_maxTest - x_minTest)*(y_maxTest - y_minTest)
        
        #radii to test for ripley's function
        r = np.linspace(0, 80, 800)
        
        #implementing ripley's H function
        R = RipleysKEstimator(areaTest, x_maxTest, y_maxTest, x_minTest, y_minTest)
        ripleyHR1WGA = R.Hfunction(data = dataTest, radii = r, mode='none')
        
        #the max of ripley's H function is the cluster radius of maximum clustering
        # = radius of clusters for DBSCAN
        radius = np.append(radius, r[np.argmax(ripleyHR1WGA)])
        
        continue
        
    return stat.mean(radius)




def MultChannelsCallToDict(path, dictionaryNames):
    
    """
    Import each CSV file in the folder, make array of coordinates, make dictionary of all those arrays
    
    Input:

        path: path to folder containing the desired csv files (string)

        dictionaryNames: equivalences between names of files and futur names (dict of string)

    Output:

        dictionaryLocalizations: dictionary of 2 arrays - coordinates of points for each channel
    """
    
    # array of all csv files in this folder (path to the files)
    filePaths = np.array(glob.glob(path+'/*.csv'))

    #new empty dictionary of localizations 
    dictionaryLocalizations = {}

    for i in filePaths:
        #names of the files only (without extention or path)
        name = Path(i).stem
        
        #import locations (csv)
        df = pd.read_csv(i)
        #coordinates of points only => in a np array
        df = df[['x [nm]','y [nm]']].to_numpy()
        
        #search for recognisable part in name of the dict and makes new names accordingly
        #names of lectins and state of randomness
        for i in dictionaryNames.keys():
            if name.casefold().find(i) >= 0:
                if name.casefold().find('random') >= 0:
                    name = dictionaryNames[i]+'random' 
                else:
                    name = dictionaryNames[i]

        #add arry to dictionary of localizations 
        dictionaryLocalizations[name] = df
        
    return dictionaryLocalizations
        
        
        
      