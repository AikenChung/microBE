#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 8 22:09:02 2021

@author: aikenchung
Adopting median ratio normalization method to phyla dataset.
This step should be excuted before proceeding to PCA.
"""

import pandas as pd 
import numpy as np
import time
from statistics import geometric_mean

phyla_data_df = pd.read_csv('phyla_dataset_d3.csv')
#phyla_columnName = phyla_data_df.loc[0,:].copy()
phyla_columnName = []
# iterating the columns 
for col in phyla_data_df.columns: 
    phyla_columnName.append(col)
    
data_df = phyla_data_df.iloc[:, 1:1178].copy()

# DESeq2 median of ration normalization method
geo_mean_list = []
# Calculate geometric mean of read count for each OTU
for column in data_df:
    df_col_val_list = data_df[column].values.flatten().tolist()
    nonZero_list = list(filter(lambda a: a != 0, df_col_val_list))
    if len(nonZero_list) > 0:
        geo_mean_val = round(geometric_mean(nonZero_list), 2)
        geo_mean_list.append(geo_mean_val)
    else:
        geo_mean_list.append(0)


# starting time
start = time.time()
col_num = np.shape(data_df)[1]
# normalize read count with geometric mean for each OTU
for index, row in data_df.iterrows():
    for i in range(1, col_num):
        if row[[i]].values !=0:
            adj_factor_val = round((row[[i]].values/geo_mean_list[i])[0],2)
            if adj_factor_val > 0:
                row[[i]] = round(row[[i]].values[0]/adj_factor_val,2)

# add "run_accession" column to normed_data    
data_df[phyla_columnName[0]] = phyla_data_df[phyla_columnName[0]]
# add the rest of 6 meta-column data into normalized dataframe
for index in range(1178,1184):
    data_df[phyla_columnName[index]] = phyla_data_df[phyla_columnName[index]]


# end time
end = time.time()
totalSeconds = round(end - start)
# total time taken
print(f"Runtime of the program is {totalSeconds} seconds")            
data_df.to_csv('phyla_dataset_d3_normalized.csv', index=False, encoding='utf-8')
