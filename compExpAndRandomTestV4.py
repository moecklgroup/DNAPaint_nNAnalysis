# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:23:10 2024

@author: Admin
"""

#%% IMPORTS




import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import datetime
from pathlib import Path    





#%% PATHS

# =============================================================================
# paths to the files containing the distances to NN to compare 
# (the order of the path doesn't matter)
# =============================================================================

# the new data is saved in the folder of the first path
p = '5_colors_csv/2024-07-17_MCF10A_Lectin_DS019/well2/Cell1/analysis/DistNN_R1WGA_inAll.csv'
q = '5_colors_csv/2024-07-17_MCF10A_Lectin_DS019_randomDistrib/well2/Cell1/analysisRandInRand/DistNN_R1WGArandom_inAllRandom.csv'


# =============================================================================
# histograms parameters
# =============================================================================

rangeUp = 100 #maximum display x axis
binsize = 1 #bin size





#%% new folder for analysis data

# =============================================================================
# crates new folder for the analysis data named as current date if no analysis 
# before on the same day
# if the folder already exists - nothing is done
# =============================================================================


pathNewFolder = str(Path(p).parent) + '/' + str(datetime.date.today())
if not Path(pathNewFolder).exists():
    Path(pathNewFolder).mkdir()


#  the folder is created in the parent folder of the first path's file




#%% display histograms



timenow = datetime.datetime.now().strftime("%H-%M-%S-%f_")
# timestamp in the beginning of all new data files for identification
# is the same for all files of the same analysis (run of code)




# =============================================================================
# dictionries of distances to NN for the paths above
# =============================================================================

locsp = pd.read_csv(p)
locsp = locsp.iloc[:, 1:6].to_dict('list')

locsq = pd.read_csv(q)
locsq = locsq.iloc[:, 1:6].to_dict('list')

# =============================================================================
# histograms for each of the combinations in the distances to NN in files
# =============================================================================

for i, j in zip(locsp.keys(), locsq.keys()):
    # i, j, k in the different types of distance data
    
    plt.figure(figsize=(15,5))
    
    for key, crossType in zip([i, j], [locsp, locsq]):
        
        #display the histogram for each type of distance data
    
        y, x, _ = plt.hist(crossType[key],
                           bins=math.ceil(math.ceil(max(crossType[key]))/binsize), 
                           histtype='step', 
                           range = (0, math.ceil(max(crossType[key]))),
                           label=key)
                           
                           # bins=math.ceil(max(crossType[key])-min(crossType[key]))*5, 
                           # histtype='step', 
                           # label=key)
        xmax = x[np.argmax(y)]
        ymax = y.max()
        plt.annotate("x={:.3f}, y={:.3f}".format(xmax, ymax), xy=(xmax, ymax))        
        
    name = timenow + 'comparison_' + key
        
    plt.legend(fontsize='x-large')
    plt.xlabel('Distance to nearest neighbor (nm)')
    plt.ylabel('Count')
    plt.xlim((0,rangeUp))
    plt.show()
    
    #the figures ares saved in base folder
    plt.savefig(pathNewFolder + '/' + name + '.png')






# save analysis parameters in txt file

# =============================================================================
# all the parameters used in the analysis are saved in a txt file
# the txt files name starts with the same timestamp than the new data files
# the txt file is saved with the new data files in the folder created for the day
# =============================================================================


parametersfilename = timenow+'Parameters.txt'

#w tells python we are opening the file to write into it
outfile = open(pathNewFolder + '/' + parametersfilename, 'w')
 
outfile.write('Path to analysis 1 : ' + p + '\n\n')
outfile.write('Path to analysis 2 : ' + q + '\n\n')
outfile.write('Range histograms : 0-' + str(rangeUp) + ' (nm) \n\n')
outfile.write('bin size histograms : ' + str(binsize) + ' (nm) \n\n')

outfile.close() #Close the file when done







