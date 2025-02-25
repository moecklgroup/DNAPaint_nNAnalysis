# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:28:23 2024

@author: Admin
"""


#%% imports


from pathlib import Path


import functionsAll as funct





#%%


# =============================================================================
# paths to folder containing the csv files to analyse
# =============================================================================

# path to the csv files of the points to cluster
pathLocsPoints = r"C:\Users\sfritsc\Desktop\Custom Centers_MCF10AT\Random_Pointshdf5"



# =============================================================================
# dictionary of equivalence names of source files and futur names of new 
# created files and figures
# 
# to update when new lectins are used 
# =============================================================================

dictionaryNames = {'wga':'WGA',  
                   'sna':'SNA', 
                   'phal':'PHAL', 
                   'aal':'AAL', 
                   'psa':'PSA'}





#%% import localizations for all six channels


# makes 1 dictionary of arrays for the localizations of the points of all the channels
dictionaryLocalizations = funct.MultChannelsCallToDict_hdf5(pathLocsPoints, dictionaryNames, "locs")




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
    randomLocs[i] = funct.randomDistributionAll(dictionaryLocalizations[i], 1000) #700
    
    #save to csv 
    #funct.dictionaryToHdf5({'x [nm]':(randomLocs[i])[:,0], 
    #                       'y [nm]':(randomLocs[i])[:,1]}, 
    #                       pathNewFolder + '/' + fileName + '.csv')   

    
    #save to hdf5
    funct.dictionaryToHdf5({'locs': randomLocs[i]}, pathNewFolder + '/' + fileName + '.hdf5', dictionaryNames)   

