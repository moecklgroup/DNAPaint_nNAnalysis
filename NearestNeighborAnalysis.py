# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:40:03 2024

@author: Chloe Bielawski
"""

# %% imports


from tqdm import tqdm
from pathlib import Path
import datetime
import numpy as np
import pandas as pd

import functionsAll as funct

# %% paths and name dictionary


# =============================================================================
# paths to the csv files to analyse
# 
# if the csv files ares the same for the points in search of a neighbor 
# and those that make up the pool of potential neighbors, 
# then pathLocsPoints = 'path/to/folder'
# and pathLocsNeighbors = pathLocsPoints
# =============================================================================

# path to the csv files of the points in search of neigbors
pathLocsPoints = r"C:\Users\dmoonnu\Desktop\PCA\MCF10AT+TGFb\Cell5"
# path to the csv files of the points - pool of potential neighbors 
pathLocsNeighbors = pathLocsPoints

# =============================================================================
# dictionary of equivalence: names of source files and futur names of new 
# created files and figures
# 
# to update when new lectins are used 
# =============================================================================

dictionaryNames = {'wga': 'R1WGA',
                   'sna': 'R2SNA',
                   'phal': 'R3PHAL',
                   'aal': 'R4AAL',
                   'psa': 'R5PSA'}

orderedNames = list(dictionaryNames.values())

# =============================================================================
# histogram parameters
# =============================================================================

# histogram distances points to nearest neighbors in same channel
rangeUpSameChannel = 400  # maximum display x axis
binsizeSameChannel = 2  # bin size

# histogram distances points to nearest neighbors in different channel
rangeUpCrossChannel = 300  # maximum display x axis
binsizeCrossChannel = 2  # binsize


    
#%% import from hdf5

# makes 1 dictionary of arrays for the localizations of the points of all the channels
dictionaryLocalizationsPoints = funct.MultChannelsCallToDict_hdf5(pathLocsPoints, dictionaryNames, "locs")

# if the paths are the same, there is no need to import the files again 
if pathLocsNeighbors == pathLocsPoints:
    dictionaryLocalizationsNeighbors = dict(dictionaryLocalizationsPoints)

else:
    dictionaryLocalizationsNeighbors = funct.MultChannelsCallToDict_hdf5(pathLocsNeighbors, dictionaryNames, "locs")


# %% new folder for analysis data

# =============================================================================
# crates new folder for the analysis data named as current date if no analysis 
# before on the same day
# if the folder already exists - nothing is done
# =============================================================================


pathNewFolder = pathLocsPoints + '/' + str(datetime.date.today())
if not Path(pathNewFolder).exists():
    Path(pathNewFolder).mkdir()

# %% distance to first nearest neighbor for one in all six channels


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

maxima_x = pd.DataFrame(index=dictionaryLocalizationsPoints.keys(), columns=dictionaryLocalizationsPoints.keys())

for i in tqdm(dictionaryLocalizationsPoints.keys()):

    # calculate the nearest neighbor for every point in that channel in all the others including itself
    dictionaryDist = funct.nearestNeighborsOneInAll({i: dictionaryLocalizationsPoints[i]},
                                                    dictionaryLocalizationsNeighbors)

    nameCSV = timenow + 'DistNN_' + i + '_In_All'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('random') >= 0:
        nameCSV = nameCSV + 'random'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('centroid') >= 0:
        nameCSV = nameCSV + '_centroids'

    # save to csv (in the new folder inside original folder of the raw data)
    funct.dictionaryToCsv(dictionaryDist, pathNewFolder + '/' + nameCSV + '.csv')

    # =============================================================================
    #     distance to NN of one channel in itself is displayed separatly
    #     than the crosschannel distances to NN
    # =============================================================================

    nameFIG = timenow + 'HistDistNN_' + i + '_In_' + list(dictionaryLocalizationsNeighbors.keys())[
        list(dictionaryLocalizationsPoints.keys()).index(i)]

    # display hist of distance to NN for channel i in channel i
    maxima_x = funct.displayHistFigure({k: v for k, v in dictionaryDist.items() if
                                        k == list(dictionaryDist.keys())[list(dictionaryLocalizationsPoints).index(i)]},
                                       rangeUp=rangeUpSameChannel,  # max x of histogram display
                                       binsize=binsizeSameChannel,  # histogram binsize
                                       path=pathNewFolder + '/' + nameFIG,
                                       maxima_matrix_x=maxima_x)

    nameFIG = timenow + 'HistDistNN_' + i + '_In_All'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('random') >= 0:
        nameFIG = nameFIG + 'random'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('centroid') >= 0:
        nameFIG = nameFIG + '_centroids'
        
    # reorder the matrix
    maxima_x = maxima_x.loc[orderedNames, orderedNames]

    # display hist of distance to NN for channel i in all others
    maxima_x = funct.displayHistFigure({k: v for k, v in dictionaryDist.items() if
                                        not k == list(dictionaryDist.keys())[
                                            list(dictionaryLocalizationsPoints).index(i)]},
                                       rangeUp=rangeUpCrossChannel,  # max x of histogram display
                                       binsize=binsizeCrossChannel,  # histogram binsize
                                       path=pathNewFolder + '/' + nameFIG,
                                       maxima_matrix_x=maxima_x)

    # =============================================================================
    #     the csv files of distance to NN and figures 
    #     are automatically saved in a new folder (1/per day) inside the original 
    #     folder of the point in search of a neighbor (pathLocsPoints)
    # =============================================================================

    continue
    # (for tqdm)

print(maxima_x)

nameFigMatrix = timenow + 'nN_matrix'
funct.plot_matrix_histogram(maxima_x, path=pathNewFolder + '/' + nameFigMatrix)

# save analysis parameters in txt file

# =============================================================================
# all the parameters used in the analysis are saved in a txt file
# the txt files name starts with the same timestamp than the new data files
# the txt file is saved with the new data files in the folder created for the day
# =============================================================================


parametersfilename = timenow + 'Parameters.txt'

# w tells python we are opening the file to write into it
outfile = open(pathNewFolder + '/' + parametersfilename, 'w')

outfile.write('Path to points for which to find neighbors : ' + pathLocsPoints + '\n\n')
outfile.write('Path to pools of potential neighbors : ' + pathLocsNeighbors + '\n\n')
outfile.write('Range histogram same channel : 0-' + str(rangeUpSameChannel) + ' (nm) \n\n')
outfile.write('bin size histogram same channel : ' + str(binsizeSameChannel) + ' (nm) \n\n')
outfile.write('Range histogram cross channel : 0-' + str(rangeUpCrossChannel) + ' (nm) \n\n')
outfile.write('bin size histogram cross channel : ' + str(binsizeCrossChannel) + ' (nm) \n\n')

outfile.close()  # Close the file when done
