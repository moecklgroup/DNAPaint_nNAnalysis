# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 08:49:10 2024

@author: Chloe Bielawski
"""


#%% IMPORTS


from pathlib import Path    
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat
import scipy.spatial as spat
import datetime


import functionsAll as funct



# from sklearn import metrics
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from math import *
# from astropy.stats import RipleysKEstimator
# import pandas as pd
import math
# import scipy.stats as sts
# import glob 



#%% PATHS

# =============================================================================
# paths to folder containing the csv files to analyse
# =============================================================================

# path to the csv files of the points to cluster
pathLocsPoints = r'5_colors_csv\2024-07-17_MCF10A_Lectin_DS019\well2\Cell1'



# =============================================================================
# dictionary of equivalence names of source files and futur names of new 
# created files and figures
# 
# to update when new lectins are used 
# =============================================================================

dictionaryNames = {'wga':'R1WGA',  
                   'sna':'R2SNA', 
                   'phal':'R3PHAL', 
                   'aal':'R4AAL', 
                   'psa':'R5PSA', 
                   'mannaz':'R6ManNAz'}









#%% IMPORT DATA CLUSTERING


# makes 1 dictionary of arrays for the localizations of the points of all the channels
dictionaryLocalizations = funct.MultChannelsCallToDict(pathLocsPoints, dictionaryNames)









#%% NEW FOLDER FOR DATA ANALYSIS 

# =============================================================================
# crates new folder for the analysis data named as current date if no analysis 
# before on the same day
# if the folder already exists - nothing is done
# =============================================================================


pathNewFolder = pathLocsPoints + '/' + str(datetime.date.today())
if not Path(pathNewFolder).exists():
    Path(pathNewFolder).mkdir()








#%% DEFINITION OF STUDY AREAS FOR RIPLEYS ANALYSIS

# =============================================================================
# a few areas are selected for the ripley analysis instead off all the point 
# because of size issues
# they are elected automatically but can also be selected manually with the 
# 'cuts' array bellow
# depending on the sahpe of the cell, a manual selsction of the areas might be 
# more apropriate 
# =============================================================================



X = dictionaryLocalizations['R1WGA']

# =============================================================================
# automatic selection : 
# (of the bottom left corners of the boxes)
# =============================================================================
cuts = [[stat.mean(X[:,0]), stat.mean(X[:,1])],
        [stat.mean(X[:,0])+(max(X[:,0])-stat.mean(X[:,0]))/2, stat.mean(X[:,1])], 
        [stat.mean(X[:,0])-(stat.mean(X[:,0])-min(X[:,0]))/2, stat.mean(X[:,1])], 
        [stat.mean(X[:,0]), stat.mean(X[:,1])+(max(X[:,1])-stat.mean(X[:,1]))/2], 
        [stat.mean(X[:,0]), stat.mean(X[:,1])-(stat.mean(X[:,1])-min(X[:,1]))/2]]



# =============================================================================
# manual selection : 
# (imput the coordinates of the bottom left corners of the box in the array)
# should be commented when not used
# =============================================================================
cuts = [[500, 30000],
        [5000, 45000], 
        [10000, 30000], 
        [15000, 30000], 
        [20000, 30000]]




# =============================================================================
# display of the study areas for ripley
# the user might want to consider manually selecting the areas if more than 
# one is outside the boundaries of the cell
# =============================================================================

plt.figure(figsize=(50,50)) # size of the figure 
plt.plot(X[:, 0], X[:, 1], '.', ms=1) #the points of the cell 
for i, j in cuts:
    plt.plot([i, i, i+1500, i+1500, i], [j, j+1500, j+1500, j, j], color='red', linewidth=3)
    #the study areas for ripley
plt.show()










#%% RIPLEY'S PARAMETERS DBSCAN 1

# =============================================================================
# calculatiosn of the radius of clusters and min number of points 
# for each channel in the dictionry of localisations :
#     - the radius is the radius for the the maximum of ripley H function 
#     (maximum clustering)
#     - the min number of points is the mean number of 
#     neighbors of a point, inside the radius
#     (it should be changed from the automatic value if it exceeds 20-30)
# =============================================================================


radius = np.empty(len(dictionaryLocalizations)) #radius of ripley's H max
nb = np.empty(len(dictionaryLocalizations), int) #raw nb of points


for i, j in zip(dictionaryLocalizations.values(), range(0, len(dictionaryLocalizations))):

    radius[j] = funct.ripleyParametersForClustering(i, cuts) 
    
    nb[j] = math.ceil(stat.median(spat.cKDTree(i).query_ball_point(i, r = radius[j], return_length=True)))
    # the min number of points is the median of the number of neighbors
    # calculated for each poit of the channel
    
print(radius, nb)









#%% RIPLEY'S PARAMETERS DBSCAN 2

# =============================================================================
# this bloc is used to change the value of min number of points if the value 
# automatically calculated is too hight
# =============================================================================

nbUsed = nb.copy()
radiusUsed = radius.copy()

maximumNbPoints = 20

# if nb is > to 20, that value will be used instead
for i in range(0, len(nbUsed)):
    if nbUsed[i]>maximumNbPoints:  nbUsed[i]= maximumNbPoints



print(radiusUsed, nbUsed)


#%% MANUAL CHOICE OF PARAMETERS

# =============================================================================
# the user can control of the number of points per cluster and radius 
# with the following arrays
# (comment the bloc when not used)
# =============================================================================

# nbUsed = [10, 10, 10, 10, 10]


# radiusUsed = [10, 10, 10, 10, 10]




#%% CHECK ACCURACY OF CLUSERING

# =============================================================================
# this part is not saved but is a way to check the parameters for the clustering 
# and the accuracy of the results
# for that, the clustering is done on a small section of the data for each channels
# and a representation of the clusters is diplayed. 
# =============================================================================

for i, j, k in zip(dictionaryLocalizations.values(), range(0, len(dictionaryLocalizations)), dictionaryLocalizations.keys()):
    # for each of the channels in the dictionary 
    
    dataTest = i[(i[:,0] >= stat.mean(i[:,0])) & 
                 (i[:,0] <= stat.mean(i[:,0])+2000) & 
                 (i[:,1] >= stat.mean(i[:,1])) & 
                 (i[:,1] <= stat.mean(i[:,1])+2000)]
    # sample of the data to analyse 

    # title of the figure including channel and clustering parameters
    title = ['Ripley Parameters ' + k + ' : esp = ' + "{:.3f}".format(radiusUsed[j]) + ', nbmin = ' + str(nbUsed[j])]
    
    # clustering with DBSCAN
    labelsTest = funct.clusteringDBSCAN(dataTest, radiusUsed[j], nbUsed[j]) # 5, 5
    print(len(dataTest))
    centroids = funct.calculateCentroids(labelsTest, dataTest) #works
    
    # display of the points and centers of clusters
    funct.displayPointsCentroids({k:dataTest}, {k:centroids}, 30, 30, title)











#%% CLUSTERING AND SAVING OF RESULTS



timenow = datetime.datetime.now().strftime("%H-%M-%S-%f_")
# timestamp in the beginning of all new data files for identification
# is the same for all files of the same analysis (run of code)





# =============================================================================
# clustering of the data and saving of the results as csv files
# same principle than precedent bloc, but with saving of the data and no display 
# =============================================================================

for i, j, k in zip(dictionaryLocalizations.values(), range(0, len(dictionaryLocalizations)), dictionaryLocalizations.keys()):
    # for each of the channels in the dictionary 

    # clustering with DBSCAN
    labelsTest = funct.clusteringDBSCAN(i, radiusUsed[j], nbUsed[j]) # 5, 5
    print(len(i))
    centroids = funct.calculateCentroids(labelsTest, i) #works
    
    # path and name of the new CSV file containing localizations of centroids of clusters
    fileName = timenow + k + '_centroids' # _r' + "{:.3f}".format(radiusUsed[j]) + '_nb' + str(nbUsed[j])
    
    # save CSV
    funct.dictionaryToCsv({'x [nm]':centroids[:,0], 'y [nm]':centroids[:,1]}, 
                          nameCsv = pathNewFolder + '/' + fileName + '.csv')









# save analysis parameters in txt file

# =============================================================================
# all the parameters used in the analysis are saved in a txt file
# the txt files name starts with the same timestamp than the new data files
# the txt file is saved with the new data files in the folder created for the day
# =============================================================================


parametersfilename = timenow+'Parameters.txt'

#w tells python we are opening the file to write into it
outfile = open(pathNewFolder + '/' + parametersfilename, 'w')
 
outfile.write('Clustering method : Ripleys H function \n\n')
outfile.write('Path to points to cluster : ' + pathLocsPoints + '\n\n')
outfile.write('Areas Ripley analysis : ' + str(cuts) + '\n\n')
outfile.write('Max radius clusters calculated : ' + str(radius) + '\n\n')
outfile.write('Max radius clusters used : ' + str(radiusUsed) + '\n\n')
outfile.write('Min points cluster calculated : ' + str(nb) + '\n\n')
outfile.write('Min points cluster used : ' + str(nbUsed) + '\n\n')

outfile.close() #Close the file when done







