#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 17:24:06 2021

@teammembers: abayega, mali0303, aikenchung

PCA for normalized microbiome readcount data

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

phyla_data_df = pd.read_csv('phyla_dataset_d3_normalized.csv')
phyla_columnName = []
# iterating the columns 
for col in phyla_data_df.columns: 
    phyla_columnName.append(col)

# Extract data value
data_value_df = phyla_data_df.iloc[:, 0:1177].copy()
# Remove the outliers, there are total 5 of them.
data_value_df_trimmed = data_value_df.drop([964, 1183, 1243, 1284, 3237]).copy()
# Extract metadata columns
metaData_df_ori = phyla_data_df.iloc[:, 1177:1184].copy()
# Remove the outliers, there are total 5 of them.
metaData_df = metaData_df_ori.drop([964, 1183, 1243, 1284,3237]).copy()

# metadata
col_site_list = []
diagnosis_list = []
stool_biopsy_list = []
studyID_list =[]
uc_id_list = []
# metadata of interest
metadata_col_list = ['col_site', 'diagnosis', 'stool_biopsy', 'studyID', 'uc_cd']
metadata_list = [col_site_list, diagnosis_list, stool_biopsy_list, studyID_list, uc_id_list]
# Specify the index of the metadata to look into 
col_data_select_Index = 0

# extract metadata from dataframe to lists
row_num, col_num = metaData_df.shape
for index, row in metaData_df.iterrows():
    col_site_list.append(row['col_site'])
    if row['diagnosis'] > 0:
        diagnosis_list.append('Yes')
    else:
        diagnosis_list.append('No')
    stool_biopsy_list.append(row['stool_biopsy'])
    studyID_list.append(row['studyID'])
    uc_id_list.append(row['uc_cd'])
    

# Standardizing the matrix data
norm_data = StandardScaler().fit_transform(data_value_df_trimmed)

# Principal Component Analysis
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(norm_data)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC-1', 'PC-2', 'PC-3'])

# extract variance percentage
pca_variance = pca.explained_variance_ratio_

# appending metadata for plotting
data_group_list = []
for i in range(row_num):
       data_group_list.append(metadata_list[col_data_select_Index][i])

group_name_list = ['col_site_Group', 'diagnosis_Group', 'stool_biopsy_Group', 'studyID_Group', 'uc_cd_Group']
principalDf[group_name_list[col_data_select_Index]] = data_group_list

# parameters for PCA result plotting
targets = []
targets_label = []
colors = []
color_universe = ['blue', 'red','darkgreen','darksalmon', 
                  'slateblue', 'lightcoral', 'g', 'orange',
                  'mediumblue', 'darkred', 'springgreen', 'orangered',
                  'steelblue', 'tomato', 'limegreen', 'orchid']

val_count = Counter(data_group_list)
targets = list(set(data_group_list))

for i in range(len(targets)):
    targets_label.append(targets[i]+ " ("+str(val_count[targets[i]])+")")
    colors.append(color_universe[i])

# for 3D ploting
fig = plt.figure(figsize = (10,8))

ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('\nPC-1 ('+str(round(pca_variance[0],2))+')', fontsize = 15)
ax.set_ylabel('\nPC-2 ('+str(round(pca_variance[1],2))+')', fontsize = 15)
ax.set_zlabel('\nPC-3 ('+str(round(pca_variance[2],2))+')', fontsize = 15)
ax.tick_params(axis='x', which='major', pad=0.01)
plt.rc('xtick',labelsize=14)

ax.tick_params(axis='y', which='major', pad=0.01)
ax.tick_params(axis='z', which='major', pad=0.01)

ax.zaxis.labelpad = -5
ax.set_title(('PCA - ' + metadata_col_list[col_data_select_Index]), fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = principalDf[group_name_list[col_data_select_Index]] == target
    ax.scatter(principalDf.loc[indicesToKeep, 'PC-1']
               , principalDf.loc[indicesToKeep, 'PC-2']
               , principalDf.loc[indicesToKeep, 'PC-3']
               , c = color
               , s = 5
               , depthshade = False)

# If you want to label data point with its index value, uncomment this part of codes    
# =============================================================================
# for index in range(row_num):
#     ax.text(principalDf.loc[index, 'PC-1'],principalDf.loc[index, 'PC-2'],
#             principalDf.loc[index, 'PC-3'],  '%s' % (str(index)), size=8, zorder=6,  
#             color='k')
# =============================================================================

ax.legend(targets_label,markerscale=4)
ax.grid()
ax.dist = 50

ax.view_init(15, 195)
fig


