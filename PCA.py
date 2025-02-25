# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:45:45 2025
This is a script to perform pricipal component analysis on a set of data. Any data input  for example, nn peaks distances or glyco numbers can
be used. The code normalizes the data using min max normaliztion. Shifting mean of the data set to origin is performed in the sklearn package.
This is supposed to run on a whole data set. That means either on a pannel dataset or a disease condotion dataset. It extracts the full dataset
@author: dmoonnu
"""
from sklearn.decomposition import PCA
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')

def find_json_files(base_dirs, keyword):
    """
    looks for json file in subfolders using a keyword  in the filename.

    Parameters
    ----------
    base_dirs : str
         
    keyword : str
        The default is "Peaks_Combined".

    Returns
    -------
    json_files : list of Path object


    """
    json_files = []
    for base_dir in base_dirs:
        for path in Path(base_dir).rglob(f"*{keyword}*.json"):
            json_files.append(path)
    return json_files

def normalize_dataframe(df):
    "performs Min Max Normalization of the entire dataframe"
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # Calculate global min and max from the entire DataFrame (only numerical columns)
    global_min = df[numerical_cols].min().min()
    global_max = df[numerical_cols].max().max()

    # Avoid division by zero if max == min
    if global_max != global_min:
        # Normalize only the numerical columns
        for col in numerical_cols:
            df[col] = (df[col] - global_min) / (global_max - global_min)
    else:
        # If all values are the same, set them to 0 (or any other strategy)
        df[numerical_cols] = 0
    
    return df

def perform_pca(df, n_components=3):
    numerical_cols = df.select_dtypes(include=['number']).columns
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(df[numerical_cols])

    for i in range(n_components):
        df[f'PC{i+1}'] = pca_components[:, i]

    return df, pca

def extract_key_value_pairs(json_files, folder_names,pca_axes):
    """
    Extract data from the list of of json files and append them to a dataframe with keys as the column name. Each file correspond to a cell.
    ie. Each row correspond to a cell.
    """
    rows = []
    
    for file in json_files:#iterat throgh each file
        try:
            with file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict): #Checks if the loaded data from json file is a dictionary
                    filtered_data = {k: v for k, v in data.items() if v}  # filtering Exclude zero-valued keys
                    
                    # Find the matching folder name
                    parent_folders = [p.name for p in file.parents] #check the file name for parent folder
                    #Fetching name of the subfolder
                    matched_name = next((name for name in folder_names if name in parent_folders), "Unknown") #Tries to match the fetched name with the given name of folders
                    
                    filtered_data["Cell Type"] = matched_name
                    rows.append(filtered_data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {file}: {e}")
    
    df = pd.DataFrame(rows)
    
    # Move "Cell Type" to the last column
    if "Cell Type" in df.columns:
        cols = [col for col in df.columns if col != "Cell Type"]
        cols.append("Cell Type")
        df = df[cols]
    
    # Normalize numerical columns using global min and max
    df = normalize_dataframe(df)
    df = df.fillna(0)
    # Apply PCA and reduce dimensionality to 2 components (or change as needed)
    df, pca = perform_pca(df, n_components=pca_axes)
    
    if pca_axes==2:
        plot_pca(df)
    elif pca_axes==3:
        plot_pca_3d(df)
    plot_scree(pca)

   
    
    return df, pca


def plot_pca_3d(df):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    unique_sources = df["Cell Type"].unique()
    for source in unique_sources:
        source_data = df[df["Cell Type"] == source]
        ax.scatter(source_data["PC1"], source_data["PC2"], source_data["PC3"], label=source, s=100, alpha=0.7)

    ax.set_title('3D PCA Plot')
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel('PC 3')
    ax.legend(title="Cell Type")
    
    plt.show()



def plot_pca(df):
    # Plotting the PCA results (PC1 vs PC2)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cell Type", palette="Set1", s=100, alpha=0.7)


    # Add labels and title
    plt.title('PCA: PC1 vs PC2')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')

    # Show the plot
    plt.legend(title="Cell Type")
    plt.show()
    
def plot_scree(pca_model):
    # Plotting the Scree Plot
    plt.figure(figsize=(8, 6))
    
    # Get the explained variance for each principal component
    explained_variance = pca_model.explained_variance_ratio_
    
    # Plot the explained variance as a bar chart
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='b')
    
    # Add labels and title
    plt.title('Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    
    # Show the plot
    plt.show()

def main():
    # List of folders to search in (update with actual folder paths)
    folders = [r"C:\Users\dmoonnu\Desktop\PCA"]
    folder_names = ["MCF10A", "MCF10AT","MCF10A+TGFb","MCF10AT+TGFb"]  # List of known folder names
    
   
    json_files = find_json_files(folders, "PCA")
    
    # Extract and process the data
    result_df, pca_model = extract_key_value_pairs(json_files, folder_names,3)
    
    
    return result_df 

# Call main function
result_df = main()
