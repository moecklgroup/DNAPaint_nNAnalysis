# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 09:42:58 2024

@author: Chloe Bielawksi
"""


#%% imports


from pathlib import Path
import numpy as np
import pandas as pd
import statistics as stat
import scipy.spatial as spat
import glob 
import datetime


import functionsAll as funct



# from sklearn import metrics
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
# import math




#%% PATH TO THE NECESSARY DATA

# =============================================================================
# paths to folder containing the csv files to analyse
# =============================================================================

# path to the csv files of the points to cluster
pathLocsPoints = '5_colors_csv/2024-07-17_MCF10A_Lectin_DS019/well2/Cell1'



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




# =============================================================================
# path to nearest neighbors data
# =============================================================================

pathDistNN = '5_colors_csv/2024-07-17_MCF10A_Lectin_DS019/well2/Cell1/analysis'






#%% IMPORT DATA CLUSTERING


# makes 1 dictionary of arrays for the localizations of the points of all the channels
dictionaryLocalizations = funct.MultChannelsCallToDict(pathLocsPoints, dictionaryNames)








#%% new folder for analysis data 

# =============================================================================
# crates new folder for the analysis data named as current date if no analysis 
# before on the same day
# if the folder already exists - nothing is done
# =============================================================================


pathNewFolder = pathLocsPoints + '/' + str(datetime.date.today())
if not Path(pathNewFolder).exists():
    Path(pathNewFolder).mkdir()







#%% IMPORTING NEARSET NEIGHBOR DATA FOR CLUSTERING




# =============================================================================
# import already calculated distance to NN 
# =============================================================================


dictionaryDistNN = {}

filePaths = np.array(glob.glob(pathDistNN+'/*.csv'))

for i in filePaths:
    #names of the files only (without extention or path)
    name = Path(i).stem
    
    #import locations (csv)
    df = pd.read_csv(i)
    
    #search for recognisable part in name of the dict and makes new names accordingly
    #names of lectins and state of randomness
    for i, j in zip(dictionaryNames.keys(), range(0, len(dictionaryNames.keys()))):
        if name.casefold().find(i) >= 0:
            # interseting columns are coordinates => to new array => in dictionary 
            df = df.iloc[:, j+1:j+2].to_numpy()
            name = dictionaryNames[i] + 'In' + dictionaryNames[i]

    #add arry to dictionary of localizations 
    dictionaryDistNN[name] = df








#%% PARAMETERS CLUSTERING DBSCAN 1

# =============================================================================
# calculatiosn of the radius of clusters and min number of points 
# for each channel in the dictionry of localisations :
#     - the radius is the radius 90% of nearest neighbors are inside of  
#     (maximum clustering)
#     - the min number of points is the mean number of 
#     neighbors of a point, inside the radius
#     (it should be changed from the automatic value if it exceeds 20-30)
# =============================================================================


radius = np.empty(len(dictionaryLocalizations)) #radius of clustering
nb = np.empty(len(dictionaryLocalizations)) #raw nb of points


for i, j, k in zip(dictionaryLocalizations.values(), range(0, len(dictionaryLocalizations)), dictionaryDistNN.values()):

    radius[j] = np.percentile(k, 90)  # return nth percentile
    
    nb[j] = stat.median(spat.cKDTree(i).query_ball_point(i, r = radius[j], return_length=True))
    # the min number of points is the median of the number of neighbors
    # calculated for each poit of the channel
    
print(radius, nb)



nbUsed = nb.copy()

radiusUsed = radius.copy()






#%% MANUAL CHOICE OF PARAMETERS

# =============================================================================
# the user can control of the number of points per cluster with the 
# following arrays
# (comment the bloc when not used)
# =============================================================================

# nbUsed = [10, 10, 10, 10, 10]

# radiusUsed = [10, 10, 10, 10, 10]

# print(radiusUsed, nbUsed)



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
    title = ['Precision Parameters ' + k + ' : esp = ' + "{:.3f}".format(radiusUsed[j]) + ', nbmin = ' + str(nbUsed[j])]
    
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

    # title of the figure including channel and clustering parameters
    title = 'Precision Parameters ' + k + ' : esp = ' + "{:.3f}".format(radiusUsed[j]) + ', nbmin = ' + str(nbUsed[j])
    
    # clustering with DBSCAN
    labels = funct.clusteringDBSCAN(i, radiusUsed[j], nbUsed[j]) # 5, 5
    print(len(i))
    centroids = funct.calculateCentroids(labels, i) #works
    
    # path and name of the new CSV file containing localizations of centroids of clusters
    fileName = timenow + k + '_centroids'
    
    # save CSV
    # funct.dictionaryToCsv({k + ' - centroids':centroids}, pathName + '.csv')
    
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
 
outfile.write('Clustering method : 90nth percentile \n\n')
outfile.write('Path to points for which to find neighbors : ' + pathLocsPoints + '\n\n')
outfile.write('Max radius clusters calculated : ' + str(radius) + '\n\n')
outfile.write('Max radius clusters used : ' + str(radiusUsed) + '\n\n')
outfile.write('Min points cluster calculated: ' + str(nb) + '\n\n')
outfile.write('Min points cluster used : ' + str(nbUsed) + '\n\n')

outfile.close() #Close the file when done





