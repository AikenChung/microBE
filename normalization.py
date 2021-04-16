# -*- coding: utf-8 -*-
"""Normalization.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10wmI5MkICDH-5dsmnzaG-6OdumauMQ4I

The following code normalizes data three different ways:


*   Standardization
*   Min-max
*   MAD (Median Absolute Deviation)
"""

import os
import numpy as np
import pandas as pd
from statistics import mean, median
from sklearn.preprocessing import StandardScaler
from scipy.stats import median_absolute_deviation
from google.colab import drive
from glob import glob

drive.mount('/content/drive')

metadata = ['col_site', 'diagnosis', 'sample_title', 'stool_biopsy', 'studyID', 'uc_cd']

# Pulling path of files 
all_files = glob('/content/drive/MyDrive/AI in Microbiome/Filtered normalized data/phyla_all*.csv')
biop_files = glob('/content/drive/MyDrive/AI in Microbiome/Filtered normalized data/phyla_biop*.csv')
stool_files = glob('/content/drive/MyDrive/AI in Microbiome/Filtered normalized data/phyla_stool*.csv')

ls_files_data = {}
ls_files_met = {}

#ALL tables
for i in range(len(all_files)) :
  path = all_files[i]
  tbl_name_s = path.find("phyla")
  tbl_name_e = path.find(".")
  tbl_name = path[tbl_name_s:tbl_name_e]
 
  tbl = pd.read_csv(path)
  tbl = tbl.set_index('Unnamed: 0')

  ls_files_data[tbl_name] = tbl.drop(metadata, axis=1)
  ls_files_met[tbl_name] = tbl[metadata]

#BIOP tables
for i in range(len(biop_files)) :
  path = biop_files[i]
  tbl_name_s = path.find("phyla")
  tbl_name_e = path.find(".")
  tbl_name = path[tbl_name_s:tbl_name_e]
 
  tbl = pd.read_csv(path)
  tbl = tbl.set_index('Unnamed: 0')

  ls_files_data[tbl_name] = tbl.drop(metadata, axis=1)
  ls_files_met[tbl_name] = tbl[metadata]

#STOOL tables
for i in range(len(stool_files)) :
  path = stool_files[i]
  tbl_name_s = path.find("phyla")
  tbl_name_e = path.find(".")
  tbl_name = path[tbl_name_s:tbl_name_e]
 
  tbl = pd.read_csv(path)
  tbl = tbl.set_index('Unnamed: 0')

  ls_files_data[tbl_name] = tbl.drop(metadata, axis=1)
  ls_files_met[tbl_name] = tbl[metadata]

"""## Standardization"""

standardized_files = {}
training_params = {}

#For training data
for name in ls_files_data :
  
  # sub_name will be used to store params for specific data
  start = name.find("_") + 1
  end = name.find("_", start) #finding the second _
  sub_name = name[start:end]

  tmp = ls_files_data[name]

  if "75p" in name :
    scaler = StandardScaler()
    tmp_stand = scaler.fit_transform(tmp)
    training_params[sub_name] = {'mean' : scaler.mean_, 'scale' : scaler.scale_}
    standardized_files[name] = pd.DataFrame(tmp_stand, index = tmp.index, columns = tmp.columns)

#For validation and test (made a separate loop to ensure training_params exists before)
for name in ls_files_data :

  # sub_name will be used to store params for specific data
  start = name.find("_") + 1
  end = name.find("_", start) #finding the second _
  sub_name = name[start:end]

  tmp = ls_files_data[name]
  mean_ls = training_params[sub_name]['mean']
  std_ls = training_params[sub_name]['scale']

  if "75p"  not in name :
    tmp = (tmp - mean_ls)/std_ls
    standardized_files[name] = tmp

# Adding metadata back
for name in standardized_files :
  standardized_files[name] = pd.concat([standardized_files[name], ls_files_met[name] ], axis=1)

"""## Min-Max"""

minmax_files = {}
training_params = {}

#For training data 
for name in ls_files_data :

  # sub_name will be used to store params for specific data
  start = name.find("_") + 1
  end = name.find("_", start) #finding the second _
  sub_name = name[start:end]

  tmp = ls_files_data[name]
  ls_min = []
  ls_max = []

  if "75p" in name :
    for f in tmp.columns :
      feature_min = min(tmp[f])
      feature_max = max(tmp[f])
      ls_min.append(feature_min)
      ls_max.append(feature_max)

      tmp.loc[:, f] = (tmp.loc[:, f] - feature_min)/(feature_max - feature_min)

    training_params[sub_name] = {'min' : ls_min, 'max': ls_max}
    minmax_files[name] = tmp

#For validation and test (made a separate loop to ensure training_params exists before)
for name in ls_files_data :
  # sub_name will be used to store params for specific data
  start = name.find("_") + 1
  end = name.find("_", start) #finding the second _
  sub_name = name[start:end]

  tmp = ls_files_data[name]
  min_ls = np.array(training_params[sub_name]['min'])
  max_ls = np.array(training_params[sub_name]['max'])

  if "75p" not in name :
    tmp = (tmp - min_ls) / (max_ls - min_ls)
    minmax_files[name] = tmp

# Adding metadata back
for name in minmax_files :
  minmax_files[name] = pd.concat([minmax_files[name], ls_files_met[name] ], axis=1)

"""## MAD

"""

mad_files = {}
training_params = {}

#For training data 
for name in ls_files_data :

  # sub_name will be used to store params for specific data
  start = name.find("_") + 1
  end = name.find("_", start) #finding the second _
  sub_name = name[start:end]

  tmp = ls_files_data[name]
  ls_med = []
  ls_mad = []

  if "75p" in name :
    for f in tmp.columns :
      feature_median = median(tmp[f])
      feature_mad = median_absolute_deviation(tmp[f], scale=1)
      ls_med.append(feature_median)
      ls_mad.append(feature_mad)

      tmp.loc[:, f] = (tmp.loc[:, f] - feature_median)/feature_mad
    training_params[sub_name] = {'median' : ls_med, 'mad': ls_mad}
    mad_files[name] = tmp

#For validation and test (made a separate loop to ensure training_params exists before)
for name in ls_files_data :
  # sub_name will be used to store params for specific data
  start = name.find("_") + 1
  end = name.find("_", start) #finding the second _
  sub_name = name[start:end]

  tmp = ls_files_data[name]
  median_ls = np.array(training_params[sub_name]['median'])
  mad_ls = np.array(training_params[sub_name]['mad'])
  
  if "75p" not in name :
    tmp = (tmp - median_ls )/mad_ls
    mad_files[name] = tmp

# Adding metadata back
for name in mad_files :
  mad_files[name] = pd.concat([mad_files[name], ls_files_met[name] ], axis=1)

"""# Writing files as csv files"""

# Standardization 
path_to_folder = '/content/drive/MyDrive/AI in Microbiome/Filtered normalized data/noBE_removal_clr_and_other_normalizations/standardization/'
for name in standardized_files :
  tbl = standardized_files[name]
  nrows = str(tbl.shape[0])
  ncols = str(tbl.shape[1])
  tbl.to_csv(path_to_folder + name + '_standardized.csv')

# Min-Max 
path_to_folder = '/content/drive/MyDrive/AI in Microbiome/Filtered normalized data/noBE_removal_clr_and_other_normalizations/min-max/'
for name in minmax_files :
  tbl = minmax_files[name]
  nrows = str(tbl.shape[0])
  ncols = str(tbl.shape[1])
  tbl.to_csv(path_to_folder + name + '_min-max.csv')

# MAD
path_to_folder = path_to_folder = '/content/drive/MyDrive/AI in Microbiome/Filtered normalized data/noBE_removal_clr_and_other_normalizations/MAD/'
for name in mad_files :
  tbl = mad_files[name]
  nrows = str(tbl.shape[0])
  ncols = str(tbl.shape[1])
  tbl.to_csv(path_to_folder + name + '_MAD.csv')
