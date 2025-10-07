# Glycan-Atlassing
**Summary** 
This code is intended to cluster the multidimensional super resolution data within a given radius, perform nearest neighbor analysis and GlyCo. Localization data from each cell should be saved in a folder with the filenames starting with the corresponding lectin followed by underscores. (eg: "AAL_and whatever necessary information"). This folder path is to be given as the input in the variable "localization folder" in the parameter_file.json.
The outputs are multidimensional cluster information as a json file, bar chart of the top classes, and the scatter plot of top x class location.  Codes are supposed to be run at the level of a single cell. i.e. Folder containing localization data of individual cells are required. 
The script uses Picasso software package from Jungmann lab (https://github.com/jungmannlab/picasso?tab=readme-ov-file) ver-0.7.4 for clustering with tailored modifications.


For reconstruction, drift correction, alignment and segmentation Picasso software version 0.7.4 is used. After segmentation the localizations from each cell are stored in separate folders. 

All the dependencies (listed in picassoenv.yml) are installed while installing the environment – explained in the further steps.

**How to proceed?**
	Setting up the system
* Clone or download the repository to your system
* Install anaconda navigator (Download Anaconda Distribution | Anaconda)
* Install the picasso environment using the file picassoenv.yml in Custom-Picasso folder. To install, open anaconda prompt on your pc and type in >>conda env create -f “path to the yml file” << and hit enter.
* This yml file holds all the required dependencies which will be seamlessly installed with the installation of the environment using the file. 
* After the installation continue with the anaconda prompt, type >>conda activate picassoenv<<
* After the environment is activated type >>spyder<<. A new window for spyder will open.
* Set the matplotlib plotting backend to Qt in spyder. Tools>Preferences>IPython console>Graphics> Select backend as Qt
* Edit python path to add the "Custom-Picasso" folder from the repository. Go to Tools>PYTHON PATHMANAGER> add the "Cutom-Picasso" folder.

**Runing the scripts**
* Copy the path to folder containing localizations from a single cell. Paste this path to the value localization_folder in parameter_file.json in the repository. Use “\\” instead of “\” in the path.
*  Open the GlyCo_hdf5+NNA.py file. In the user input cell, set the desired values and hit run. By default, the settings are:

num_channels = 5
pixelsize =130
min_locs =1
frame_analysis = True
xnena = 2
FIGFORMAT = '.pdf'


* The script will perform the clustering and calculate the cluster centers and stores them. 
* The cluster centers are used further in nearest neighbor analysis (NNA) and GlyCo. 
* The results from GlyCo (Location of detected classes, distribution of classes, input data for PCA) will be saved within the folder with cluster centers.
* The results from NNA (NN histograms and matrix) will be stored in a folder with timestamp within the centers folder.

In a normal desktop PC (2.90 GHz processor, 16GB RAM) the script takes about 20 minutes to run per cell. 

**PCA**

For PCA, we need a new file structure. To make it easy, copy the contents inside the localization centers folder (90_Custom Centers) from each cell and create a file structure as follows:
Parent_folder-
	Condition1-
		-Cell1
		-Cell2
		-Cell3
	Condition2-
		-Cell1
		-Cell2
		-Cell3
Copy the parent folder path to variable “folder” in the PCA.py script. Adjust the “folder_names” variables according to the folder names of conditions. Do the PCA separately on NN distances and glyco by adjusting the variable “data_to_plot”, hit run.




script documentation and use protocols on the group wiki: 
https://beta.slimwiki.com/moeckl-group/all-users-efxf7z4el/nearest-neighbor-and-cluster-analysis-by-chlo%C3%A9-ejerwhrdj-ps7a75axpifg
Guidelines on using codes from github: https://slimwiki.com/moeckl-group/all-users-efxf7z4el/guidelines-to-use-github-2vylx13ipa

