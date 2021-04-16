# -*- coding: utf-8 -*-
"""DeepMicro_Pre_Processing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nLM_6B0evwCPl2MTA5kti31w1NrVBp2o

## DeepMicro
"""

import numpy as np
import pandas as pd
from skbio.stats.composition import clr

# Pulling all tables in, and concatenating
all_train = pd.read_csv("/home/laura/Documents/AI_Genomics/Phyla Project/MAD/phyla_all_3949x1177_pmi_0_clr_75p_MAD.csv")
all_val = pd.read_csv("/home/laura/Documents/AI_Genomics/Phyla Project/MAD/phyla_all_565x1177_pmi_0_clr_10p_MAD.csv") 
all_test = pd.read_csv("/home/laura/Documents/AI_Genomics/Phyla Project/MAD/phyla_all_753x1177_pmi_0_clr_15p_MAD.csv")
all_df = pd.concat([all_train, all_val], axis=0) 
all_df = pd.concat([all_df, all_test], axis =0)

biop_train = pd.read_csv("/home/laura/Documents/AI_Genomics/Phyla Project/MAD/phyla_biopsy_1109x1177_pmi_0_clr_75p_MAD.csv")
biop_val = pd.read_csv("/home/laura/Documents/AI_Genomics/Phyla Project/MAD/phyla_biopsy_164x1177_pmi_0_clr_10p_MAD.csv")
biop_test = pd.read_csv("/home/laura/Documents/AI_Genomics/Phyla Project/MAD/phyla_biopsy_213x1177_pmi_0_clr_15p_MAD.csv")
biop_df = pd.concat([biop_train, biop_val], axis=0) 
biop_df = pd.concat([biop_df, biop_test], axis =0)

stool_train = pd.read_csv("/home/laura/Documents/AI_Genomics/Phyla Project/MAD/phyla_stool_2840x1177_pmi_0_clr_75p_MAD.csv")
stool_val = pd.read_csv("/home/laura/Documents/AI_Genomics/Phyla Project/MAD/phyla_stool_401x1177_pmi_0_clr_10p_MAD.csv")
stool_test = pd.read_csv("/home/laura/Documents/AI_Genomics/Phyla Project/MAD/phyla_stool_540x1177_pmi_0_clr_15p_MAD.csv")
stool_df = pd.concat([stool_train, stool_val], axis=0) 
stool_df = pd.concat([stool_df, stool_test], axis =0)

metadata = ['col_site', 'diagnosis', 'sample_title', 'stool_biopsy', 'studyID', 'uc_cd']

all_data = all_df.drop(metadata, axis=1).dropna()
all_diag = all_df['diagnosis'].dropna()

biop_data = biop_df.drop(metadata, axis=1).dropna()
biop_diag = biop_df['diagnosis'].dropna()

stool_data = stool_df.drop(metadata, axis=1).dropna()
stool_diag = stool_df['diagnosis'].dropna()

# Writing count data tables and diagnosis tables to DeepMicro/data
all_data.to_csv("/home/laura/Documents/AI_Genomics/Code/DeepMicro/data/all_data.csv", header=False, index=False)
all_diag.to_csv("/home/laura/Documents/AI_Genomics/Code/DeepMicro/data/all_diag.csv", header=False, index=False)

biop_data.to_csv("/home/laura/Documents/AI_Genomics/Code/DeepMicro/data/biop_data.csv", header=False, index=False)
biop_diag.to_csv("/home/laura/Documents/AI_Genomics/Code/DeepMicro/data/biop_diag.csv", header=False, index=False)

stool_data.to_csv("/home/laura/Documents/AI_Genomics/Code/DeepMicro/data/stool_data.csv", header=False, index=False)
stool_diag.to_csv("/home/laura/Documents/AI_Genomics/Code/DeepMicro/data/stool_diag.csv", header=False, index=False)

# Run if you want to see options
# 2>/dev/null supresses warnings
!python DM.py -h 2>/dev/null