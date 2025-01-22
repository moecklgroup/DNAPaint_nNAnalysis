# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:14:14 2025

@author: dmoonnu
"""
from custom_picasso import postprocess, io
from custom_picasso import clusterer
import numpy as np
import os 
from pathlib import Path
from tqdm import tqdm

#User Inputs
pixelsize =130
min_locs =1
frame_analysis = True
#Nena Factor for clustering
xnena = 2
localization_folder = Path(r"C:\Users\dmoonnu\Desktop\Test data\RAW DATA for custom")

locs_file = []
for hdf5 in localization_folder.glob('*.hdf5'):   
    locs_file.append(str(hdf5))
    
    
#%%
#Adjust the path to the localization hdf5 file
cluster_data_location= localization_folder/"Custom SMLM Clustered"
cluster_data_location.mkdir(exist_ok=True)
centers_location =localization_folder/"Custom Centers"
centers_location.mkdir(exist_ok=True)
for file in locs_file:
    #locs_path = file
    #fetch data fromthe file
    locs, info = io.load_locs(file)
    #Calculate NeNA Precision
    nena= postprocess.nena(locs, info)
    
    #Assign radius
    radius_xy = xnena*nena[1]
    print(f"Clustering radius = {radius_xy*pixelsize}nm")
    
    #Strings for creating new filename
    directory, filename = os.path.split(file)
    name, extension = os.path.splitext(filename)
    suffix= f"min_loc-{min_locs}_{xnena}XNeNA"
    
    output_name = f"{name}_clustered_{suffix}{extension}"
    output_name_centers = f"{name}_centers_{suffix}{extension}"
    
    
    #clustering
    clustered_locs=clusterer.cluster(locs, radius_xy, min_locs, frame_analysis)
    new_info = {"Number of Clusters": len(np.unique(clustered_locs.group)),
                "Min. cluster size": min_locs,
                "Performed basic frame analysis": frame_analysis,
                "Clustering radius xy (nm)": float(radius_xy * pixelsize),
                "NeNA precision(nm)": float(nena[1]*130),
                "Basic Frame analysis status": "First and last inclusive, 100bins used",
                "Percentage of localization clustered": (len(clustered_locs)/len(locs))*100
            }
    info = info + [new_info]
    outputpath_locs = os.path.join(str(cluster_data_location), output_name)
    io.save_locs(outputpath_locs, clustered_locs, info)
    print("Calculating Centers...")
    cluster_centers = clusterer.find_cluster_centers(clustered_locs,130)
    outputpath_centers = os.path.join(str(centers_location), output_name_centers)
    io.save_locs(outputpath_centers, cluster_centers, info)