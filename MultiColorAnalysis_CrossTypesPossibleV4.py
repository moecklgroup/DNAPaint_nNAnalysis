# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:40:03 2024

@author: Chloe Bielawski
"""

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from math import *
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stat
import math
import csv
import scipy.stats as sts

import functionsAll as funct

import datetime

import glob 
from pathlib import Path    


#%%



# =============================================================================
# paths to folder containing the csv files to analyse
# 
# if the csv files ares the same for the opint in search of a neighbor 
# and those for the pool of potential neighbors, 
# then pathLocsPoints = 'path/to/folder'
# and pathLocsNeighbors = pathLocsPoints
# =============================================================================

# path to the csv files of the points in search of neigbors
pathLocsPoints = r'5_colors_csv\2024-07-17_MCF10A_Lectin_DS019\well2\Cell1'
# path to the csv files of the points - pool of potential neighbors 
pathLocsNeighbors = pathLocsPoints




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
                   'manaz':'R6MaNAz'}

    






#%% import localizations for all six channels

# makes 1 dictionary of arrays for the localizations of the points of all the channels
dictionaryLocalizationsPoints = funct.MultChannelsCallToDict(pathLocsPoints, dictionaryNames)

# if the paths are the same, there is no need to import the files again 
if pathLocsNeighbors == pathLocsPoints:
    dictionaryLocalizationsNeighbors = dictionaryLocalizationsPoints

else:
    dictionaryLocalizationsNeighbors = funct.MultChannelsCallToDict(pathLocsNeighbors, dictionaryNames)








#%% new folder for analysis data

# =============================================================================
# crates new folder for the analysis data named as current date if no analysis 
# before on the same day
# if the folder already exists - nothing is done
# =============================================================================


pathNewFolder = pathLocsPoints + '/' + str(datetime.date.today())
if not Path(pathNewFolder).exists():
    Path(pathNewFolder).mkdir()








#%% distance to first nearest neighbor for one in all six channels


timenow = datetime.datetime.now().strftime("%H-%M-%S-%f_")
# timestamp in the beginning of all new data files for identification
# is the same for all files of the same analysis (run of code)



# =============================================================================
# parameters for the histogram that can be chaged : 
    
#   max x for display of the histogram 
#         - 'rangeUp' parameter of the 'displayHistFigure' function
#         => using the variables rangeUpSameChannel and rangeUpCrossChannel

#   size of the bins for the histogram 
#         - 'binsize' parameter of the 'displayHistFigure' function
#         => using the variables binsizeSameChannel and binsizeCrossChannel
# =============================================================================

for i in tqdm(dictionaryLocalizationsPoints.keys()):
    
    #calculate the nearest neighbor for every point in that channel in all the others including itself
    dictionaryDist = funct.nearestNeighborsOneInAll({i:dictionaryLocalizationsPoints[i]}, dictionaryLocalizationsNeighbors)
    
    nameCSV = timenow+'DistNN_'+i+'_In_All'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('random') >= 0:
        nameCSV = nameCSV+'random'
    
    #save to csv (in the new folder inside original folder of the raw data)
    funct.dictionaryToCsv(dictionaryDist, pathNewFolder+'/'+nameCSV+'.csv')
    
    
    
# =============================================================================
#     distance to NN of one channel in itself is displayed separatly 
#     than the crosschannel distances to NN 
# =============================================================================
    
    
    nameFIG = timenow+'HistDistNN_'+i+'_In_'+list(dictionaryLocalizationsNeighbors.keys())[list(dictionaryLocalizationsPoints.keys()).index(i)]
    
    rangeUpSameChannel = 100
    binsizeSameChannel = 0.1
    
    #display hist of distance to NN for channel i in channel i
    funct.displayHistFigure({k: v for k, v in dictionaryDist.items() if k == list(dictionaryDist.keys())[list(dictionaryLocalizationsPoints).index(i)]}, 
                            rangeUp=rangeUpSameChannel, #max x of histogram display
                            binsize=binsizeSameChannel, #histogram binsize
                            path=pathNewFolder+'/'+nameFIG) 
    
    
    nameFIG = timenow+'HistDistNN_'+i+'_In_All'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('random') >= 0: nameFIG = nameFIG+'random'
    
    rangeUpCrossChannel = 200
    binsizeCrossChannel = 0.2
    
    #display hist of distance to NN for channel i in all others 
    funct.displayHistFigure({k: v for k, v in dictionaryDist.items() if not k == list(dictionaryDist.keys())[list(dictionaryLocalizationsPoints).index(i)]}, 
                            rangeUp=rangeUpCrossChannel, #max x of histogram display
                            binsize=binsizeCrossChannel, #histogram binsize
                            path=pathNewFolder+'/'+nameFIG) 
    

    # =============================================================================
    #     the csv files of distance to NN and figures 
    #     are automatically saved in a new folder (1/per day) inside the original 
    #     folder of the point in search of a neighbor (pathLocsPoints)
    # =============================================================================
    
    continue
    # (for tqdm)








# save analysis parameters in txt file

# =============================================================================
# all the parameters used in the analysis are saved in a txt file
# the txt files name starts with the same timestamp than the new data files
# the txt file is saved with the new data files in the folder created for the day
# =============================================================================


parametersfilename = timenow+'Parameters.txt'

#w tells python we are opening the file to write into it
outfile = open(pathNewFolder + '/' + parametersfilename, 'w')
 
outfile.write('Path to points for which to find neighbors : ' + pathLocsPoints + '\n\n')
outfile.write('Path to pools of potential neighbors : ' + pathLocsNeighbors + '\n\n')
outfile.write('Range histogram same channel : 0-' + str(rangeUpSameChannel) + '\n\n')
outfile.write('bin size histogram same channel : ' + str(binsizeSameChannel) + '\n\n')
outfile.write('Range histogram same channel : 0-' + str(rangeUpCrossChannel) + '\n\n')
outfile.write('bin size histogram same channel : ' + str(binsizeCrossChannel) + '\n\n')

outfile.close() #Close the file when done






