# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:26:13 2024
This code is intended to cluster the multidimensional super resolution data within a given radius. Localization data(cluster centers) should be saved
in a folder with the filenames starting with the corresponding lectin followed by underscores. (eg: "AAL_cluster_centers").
This folder should be given as the input in the variable "localization folder". Also set the radius. The outputs are multidimensional 
cluster information as a json file, bar chart of the top classes, and the scatter plot of top x class location.  
@author: dmoonnu
"""

import numpy as np
from pathlib import Path
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import json
from datetime import datetime
import matplotlib
import yaml

# Set the graphics backend to Qt
matplotlib.use('Qt5Agg')
#kEY TO LOOK IN THE YAML FILE
key_for_area = "Total Picked Area (um^2)"
# Radius for neighborhood in nanometers ( Biologically relevant distance to find the neighbouring glycan)
radius = 5
number_to_plot =5 #tp x to plot
pathLocsPoints = r"C:\Users\dmoonnu\Desktop\PCA\MCF10A\Cell7"
localization_folder = Path(pathLocsPoints)

yaml_file = (list(localization_folder.glob("*.yaml")))[0]
with open(yaml_file,'r') as file:
    documents=yaml.safe_load_all(file)
    for info in documents:
        if isinstance(info, dict) and key_for_area in info:
            area_of_cell = np.float32(info[key_for_area])


#%%HDF5 handling



#iterating to look for hdf5 files in the folder
for file in localization_folder.glob('**/*.hdf5'):   
    print(file)

print("Files loaded successfully")
#create a dictionary of the dataset
data_dict={}
#Looping to create a dictionary with the dataset.
for file in localization_folder.glob("**/*.hdf5"):
    #Get the file stem for naming the keys
    file_name = file.stem.split("_")[0]
    #read the data 
    data = pd.read_hdf(file, key ='locs')
    #append the dict 
    data_dict[file_name] = data

#%%HDF5
#Dictionary for storing the neighbors with core point as the key
neighbor_master={}
#Dictionary to store the neighbors with core point as the key and each value as the list of tuple pair with the distance to to the considered core
distance_indexed_neighbor={}
#iterate thru each dataset
for df_key in data_dict:

    
    df_of_interest_key = df_key  

    com_name = f"neighbors_of_{df_key}" #center of mass name to be used as  the key in the dictionaries
    neighbor_master[com_name] = {}
    distance_indexed_neighbor[com_name]={}
    # Create a tree Excluding the key of interest
    # Convert each DataFrame's points to KDTree format, except the DataFrame of interest
    #trees = {key: KDTree((df[['x', 'y']]*130).values) for key, df in data_dict.items() if key != df_of_interest_key}
    trees = {key: KDTree((df[['x', 'y']]*130).values) for key, df in data_dict.items()}
    # Keep track of assigned points in each dataset
    assigned_points = {key: {} for key in trees}  # Store closest distance and index associated
    # Retrieve the DataFrame of interest
    df_of_interest = data_dict[df_of_interest_key]
    # Iterate through points in the DataFrame of interest and find neighbors in other DataFrames
    #print(f"Current core is {df_key}\n\n")
    for row_index_of_com, column in tqdm(df_of_interest.iterrows(), desc=f"Searching for neighbors of {df_of_interest_key}"):
        x1, y1 = column['x']*130, column['y']*130
           #now look for neighbors standing at this point
        # Check for neighbors in all other DataFrames using their KDTree
        for current_family, current_family_members in trees.items():
            #Generates a list of indices coresponding to the dataframe 
            indices = current_family_members.query_ball_point([x1, y1], r=radius)
            filtered_indices = [num for num in indices if df_key != current_family and num != row_index_of_com]
            #indices = current_family_members.query_ball_point([x1, y1], r=radius) if df_key==current_family and 
            # if len(indices)>2: 
            #     print(f"{df_key}_{row_index_of_com}--{current_family}-{indices}")
            #If indices exist ie. if neighbors exist for the current point from the main loop, these neighbors are stored to a dictionary with the 'id' of the point from main
            #loop as the key and list of row number as the value. Only if neighbours present then the if condition is satisfied and things are stored to the neighbor dictionary.
            if  filtered_indices:
                
                # Calculate scalar distances for each index # indices has the row number of the current family members
                dist_index_pairs = [(np.linalg.norm(current_family_members.data[idx] - [x1, y1]), idx) for idx in  filtered_indices]# creates a list of tuples with(distance, indices)
               
                # Sort by distance
                dist_index_pairs.sort(key=lambda x: x[0])
                
                # Find the closest point not already assigned or update if closer
                for distance, idx in dist_index_pairs: #idx is the index of current family member
                    # Only update if idx is unassigned or found closer
                    if idx not in assigned_points[current_family] or distance < assigned_points[current_family][idx][0]:#idx is the index of current family member
                        assigned_points[current_family][idx] = (distance, row_index_of_com)
                        break  # Only take the closest unassigned neighbor
            
    # After collecting the closest neighbors, populate neighbor_master with final assignments
    for neighbor_family, family_member in assigned_points.items():
        for idx, (distance, df_of_interest_idx) in family_member.items():
            neighbor_master[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append(f'{neighbor_family}_{idx}')    
            distance_indexed_neighbor[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append((f'{neighbor_family}_{idx}', distance))    
            
#%%convert sub dictionary in a nested dictionary to a list of tuples
def nested_dict_to_tuple_list(dict_to_convert_to_tuple):
    converted_dict = { key: [(k, *v) for k, v in sub_dict.items()] for key, sub_dict in dict_to_convert_to_tuple.items()}
    return converted_dict

converted_dict_indexed = nested_dict_to_tuple_list(neighbor_master)
# Check for more closeness on this converted_dict_indexed
#%% Make into a single list of tuples without duplicates ie. (PSA_123, WGA_345) = (WGA_345, PSA_123). No more same combination after this point.


def duplicates_removed_tuple_list(input_dict):
    tuple_list=list({tuple(sorted(t)) for value in input_dict.values() for t in value})
    tuple_list[:] = [tuple(sorted(tup)) for tup in tuple_list]

    return tuple_list

list_of_tuples_without_duplicates=duplicates_removed_tuple_list(converted_dict_indexed)
    
#%%Check for duplicate single elements. Because a single element should not appear more than once in the entire cell. ie. one element
#should be part of only one "glycan structure". It shouldn't be linked to two different neighbor groups

#flat_list = [element for tup in list_of_tuples_without_duplicates for element in tup]

def flatten_tuple_list(input_dict):
    flat_list= [element for tup in input_dict for element in tup]
    
    return flat_list

list_of_all = flatten_tuple_list(list_of_tuples_without_duplicates)
#duplicates = set([x for x in flat_list if flat_list.count(x)>1])
#%%Counter
element_counts = Counter(list_of_all) #counts how many times an element appears in the list.

frequency_counts = Counter(element_counts.values()) #Counts how many times 1,2,3,4,5,6 which are the values in the element count dictionary appears
    
# Define categories: 1, 2, 3, and "more than 3"
categories = ["1", "2", "3", ">3"]
counts = [
    frequency_counts[1],  # Count of elements appearing once
    frequency_counts[2],  # Count of elements appearing twice
    frequency_counts[3],  # Count of elements appearing thric
    
    sum(count for freq, count in frequency_counts.items() if freq > 3)  # Count of elements appearing more than thrice
]

# Plot the histogram
plt.figure(figsize=(8, 5))
plt.bar(categories, counts, color="skyblue")
plt.xlabel("Occurrence Frequency")
plt.ylabel("Number of Elements")
plt.title("Histogram of Element Frequencies")
plt.show()

#%%
def find_duplicates_with_counts(input_list):
    element_counts = Counter(input_list)
    duplicates = {element: count for element, count in element_counts.items() if count > 1}
    
    return duplicates
duplicates = find_duplicates_with_counts(list_of_all)
print(f"{len(duplicates)} duplicates found")

#For each of the duplicates found, check its positions in the unique combination form cell list
#%%
def eliminate_duplicates(neighbor_dictionary, duplicate_list):
    for duplicate_item in tqdm(duplicate_list, desc="Removing duplicates"): #iterate through each duplicated item
        smallest_value = float('inf') #reset the smallest value for each item
        #search for the smallest value in the nested dictionary
        for core_point, neighbors_sub_dict in neighbor_dictionary.items(): #diving inside the dictionary looking for the current duplicate element
            for key, index_distance_tuple in neighbors_sub_dict.items(): #diving inside the subdictionary looking for the current duplicate element
                for idx, (item,value) in enumerate(index_distance_tuple): #fetch items from the list of tuples by indexing them.
                    if item == duplicate_item and value<smallest_value:
                        smallest_value = value
                        smallest_location = (core_point, key, idx)
        
        for core_point, neighbors_sub_dict in neighbor_dictionary.items():
            for key, index_distance_tuple in neighbors_sub_dict.items():
                if (core_point, key) == smallest_location[:2]: #for the smallest location
                    neighbor_dictionary[core_point][key] = [(item,value) if (idx == smallest_location[2]) else None
                                                                  for idx, (item,value) in enumerate (index_distance_tuple)] #check the index in the list
                    
                    neighbor_dictionary[core_point][key] = [ tup for tup in neighbor_dictionary[core_point][key] if tup is not None] #Eliminating None valued keys
                else: #until the smallest location combiation of core and key is found this part ofthe loop is executed. Here the dictionary is updated if the item in the tuple pair is not the duplicate item
                    neighbor_dictionary[core_point][key] = [(item,value) for item, value in index_distance_tuple if item != duplicate_item]
    return neighbor_dictionary

dictionary_without_duplicates = eliminate_duplicates(distance_indexed_neighbor, duplicates)
        
#%%Remove distance from the updated dictionary to perform the initial filtering again                   
def remove_distance(data):
    # Iterate through each key-value pair in the outer dictionary
    for main_key, sub_dict in data.items():
        
        # Iterate through each key-value pair in the sub-dictionary
        for sub_key, tuple_list in sub_dict.items():
            
            # # Replace the integer in each tuple with the sub_key
            # updated_tuples = [(item, sub_key) for item, _ in tuple_list]
            
            # # Update the list in place
            # data[main_key][sub_key] = updated_tuples
            sub_dict[sub_key] = [elem[0] for elem in tuple_list]
    
    return data
distance_removed_unduplicated_dictionary = remove_distance(dictionary_without_duplicates)                        
#%%perform the initial filtering agin to remove pairwise duplicates

final_list_of_tuples = nested_dict_to_tuple_list(distance_removed_unduplicated_dictionary) #key value pairs to a list tuple
#%%
final_pair_wise_duplicates_removed = duplicates_removed_tuple_list(final_list_of_tuples)
#%%
#result_list = [tuple(element.split("_")[0] for element in tup) for tup in final_pair_wise_duplicates_removed]
result_list = [tuple(sorted(element.split("_")[0] for element in tup)) for tup in final_pair_wise_duplicates_removed]
result_list[:] = [tuple(sorted(tup)) for tup in result_list]
#%% Define possible combinations
from itertools import combinations_with_replacement

lectins = ['WGA','SNA','PHAL','AAL','PSA','DBCO']
possible_combinations= []
for i in range(2, 7):
    # Generate combinations with replacement for the current size i
    combs = list(combinations_with_replacement(lectins, i))
    possible_combinations.extend(combs)  # Store the combinations in the list

#hERE NO DUPLICATION SINCE THE TUPLES ARE CEREATED WITHOUT DUPLICATES. BUT FOR COMBINATIONS FROM THE CELLS DUPLICATES OCCUR.
#%%Counting classes
class_counter = {}

# Convert the standard list to sets for easier comparison
#standard_set = [set(tup) for tup in possible_combinations]

# Iterate through the standard list
for standard_tup in tqdm(possible_combinations, desc= "Counting Classes"):
    # Convert the current tuple to a set
    standard_tup_set = sorted(standard_tup)
    # Count occurrences in the tuple_list
    count = sum(1 for tup in result_list if sorted(tup) == standard_tup_set)
    # Update the result dictionary
    class_counter[standard_tup] = count

#%%Plotting class distribution

# timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# #sorting in the reverse order and picking th etop 10 class
# sorted_data = sorted(class_counter.items(), key=lambda clas: clas[1], reverse=True)[:10] 
# sorted_data = [(tuple(sorted(tup[0])), tup[1]) for tup in sorted_data]
# categories_ = [str(item[0]) for item in sorted_data]
# values_ = [item[1]/area_of_cell for item in sorted_data]
# #categories_ = [str(key) for key in class_counter.keys()]
# #values_ = list(class_counter.values())
# plt.figure(figsize=(8, 10))
# # Plot the histogram
# plt.bar(categories_, values_, color='red', edgecolor='black')

# # Add titles and labels
# plt.title('Lectin Classes', fontsize=13)
# plt.xlabel('Categories', fontsize=10)
# plt.ylabel('Count per μm\u00b2', fontsize=10)
# plt.xticks(rotation=45, ha='right')

# # Show the plot
# plt.show()   
# plt.savefig(localization_folder/f"{timestamp}_Class_Chart_{radius}nm",bbox_inches='tight')        


#%%Plotting class distribution  with normalization

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#sorting in the reverse order and picking th etop 10 class
sorted_data = sorted(class_counter.items(), key=lambda clas: clas[1], reverse=True)[:10] 
#Sorting the lectin classes
sorted_data = [(tuple(sorted(tup[0])), tup[1]) for tup in sorted_data]
#Fetching the list of categories
categories_ = [str(item[0]) for item in sorted_data]
#Scaling with
area_normalized_values = [item[1] / area_of_cell for item in sorted_data]

# Apply min-max normalization
# min_value = min(area_normalized_values)
# max_value = max(area_normalized_values)
# final_normalized_values = [(value - min_value) / (max_value - min_value) for value in area_normalized_values]

categories_ = [str(item[0]) for item in sorted_data]
plt.figure(figsize=(8, 10))
plt.bar(categories_, area_normalized_values, color='blue', edgecolor='black')
plt.xticks(rotation=45, ha='right')
plt.xlabel('Categories', fontsize=10)
plt.ylabel('Count per μm\u00b2', fontsize=10)
plt.title('Lectin Classes distribution', fontsize=13)
plt.show()

plt.savefig(localization_folder/f"{timestamp}_Lectin_Classes_ per_sq-microns_{radius}nm",bbox_inches='tight')   
     
#%%Save classes to json file

# Save class counts
output_file_path = localization_folder / f"{timestamp}_Number_of_Lectin_Classes_{radius}nm.json"

sorted_counter = dict(sorted(class_counter.items(), key=lambda item: item[1], reverse=True))
num_classes = {str(key): value for key, value in sorted_counter.items()}

# Save to a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(num_classes, json_file, indent=4)
#Save Class count per unit area   
PCA_output_file_path = localization_folder / f"{timestamp}_Lectin_Classes_ per_sq-microns_for_PCA_{radius}nm.json"

sorted_counter = dict(sorted(class_counter.items(), key=lambda item: item[1], reverse=True))
class_per_area = {str(key): value/area_of_cell for key, value in sorted_counter.items()}

# Save to a JSON file
with open(PCA_output_file_path, 'w') as json_file:
    json.dump(class_per_area, json_file, indent=4)
    
    
#%%
location_dictionary = {}

# Iterate through sorted data
for classes in tqdm(sorted_data, desc="Finding location of the classes "):

    #Extract only the tuple with Lectins
    class_looking_for_location = classes[0]
    location_dictionary[class_looking_for_location] = []
    # Iterate through the indexed tuples
    for indexed_tuple in final_pair_wise_duplicates_removed:
        # Convert each element in indexed_tuple to a tuple
        processed_tuple = tuple(q.split("_")[0] for q in sorted(indexed_tuple))
  
        # Check if it matches the class we are looking for
        if processed_tuple == class_looking_for_location:
            element, index = indexed_tuple[0].split("_")
            dataframe = data_dict[element]
            index = int(index)  # Ensure index is integer if needed
            x_val = dataframe.at[index, "x"]
            y_val = dataframe.at[index, "y"]
         
            location_dictionary[class_looking_for_location].append((x_val, y_val))
            
#%%Plot the class locations

# Define pixel size in nanometers
pixel_size_nm = 130  # 130 nanometers per pixel
scale_factor = pixel_size_nm / 1000  # Convert to micrometers (µm)

# Calculate full field of view in micrometers
field_of_view = 576 * scale_factor  # Field of view in µm (74.88 µm)

# Define plot dimensions in inches for a 576x576 pixel field
dpi = 100
fig_width = 576 / dpi
fig_height = 576 / dpi

# Sort the dictionary by the length of the coordinate list (in descending order)
location_dictionary_sorted = sorted(location_dictionary.items(), key=lambda item: len(item[1]), reverse=True)


# Select the top 5 entries based on the number of elements in the list
data_to_plot = location_dictionary_sorted[:number_to_plot]

# Plotting
plt.figure(figsize=(fig_width*2, fig_height*2), dpi=dpi)

# Assign unique colors for each key
colors = plt.cm.tab10(range(len(data_to_plot)))  # Generate distinct colors for the top 5
for (key, coords), color in zip(data_to_plot, colors):
    # Scale coordinates to micrometers
    scaled_coords = [(x * scale_factor, y * scale_factor) for x, y in coords]
    x_vals, y_vals = zip(*scaled_coords)  # Unpack x and y coordinates
    
    # Set a fixed spot size for all points (e.g., 50 points²)
    plt.scatter(x_vals, y_vals, label=str(key), color=color, s=4)  # Scatter plot

# Set axis limits to match the full field of view
plt.xlim(0, field_of_view)  # 0 to 74.88 µm
plt.ylim(0, field_of_view)  # 0 to 74.88 µm

# Invert the y-axis to set the origin at the top-left
plt.gca().invert_yaxis()

# Add legend and labels
plt.legend(title="Classes")
plt.xlabel("X Coordinate (µm)")
plt.ylabel("Y Coordinate (µm)")
plt.title(f"Top {number_to_plot} classes")
plt.grid(True)

# Show the plot
plt.show()
plt.savefig(localization_folder/f"{timestamp}_Class_location_{radius}nm",bbox_inches='tight') 

      
            
      
    



plt.close("all")




