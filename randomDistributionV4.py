# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:28:23 2024

@author: Admin
"""


#%% imports


import functionsAll as funct
from pathlib import Path




# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# from tqdm import tqdm
# from scipy.spatial import Delaunay
# from pathlib import Path
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import statistics as stat
# import random
# import math
# import glob 





#%%


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
                   'psa':'R5PSA'}





#%% import localizations for all six channels


# makes 1 dictionary of arrays for the localizations of the points of all the channels
dictionaryLocalizations = funct.MultChannelsCallToDict(pathLocsPoints, dictionaryNames)




#%% new folder for analysis data 

# =============================================================================
# crates new folder for the random points
# if the folder already exists - nothing is done
# =============================================================================


pathNewFolder = pathLocsPoints + '/' + 'Random_Points'
if not Path(pathNewFolder).exists():
    Path(pathNewFolder).mkdir()





#%% distance to first nearest neighbor for one in all six channels



# dictionary of random localisation for each of the channels imported 
randomLocs = {}

#for eah channel imported 
for i in dictionaryLocalizations.keys():
    
    fileName = i + '_random'
    
    #calculate the nearest neighbor for every point in that channel in all the others including itself
    randomLocs[i+'random'] = funct.randomDistributionAll(dictionaryLocalizations[i], 700) #700
    
    #save to csv 
    funct.dictionaryToCsv({'x [nm]':(randomLocs[i+'random'])[:,0], 
                           'y [nm]':(randomLocs[i+'random'])[:,1]}, 
                          pathNewFolder + '/' + fileName + '.csv')   






