
import numpy as np

from scipy.stats.mstats import gmean

'''
Point-wise Mutual Information. Thanks to Ahmad's advise
'''
def pmi(matrix, smooth_val=1):
    if 0 < smooth_val: matrix = matrix.todense() + smooth_val
    matrix = matrix / np.sum(matrix)
    sc, sr = np.sum(matrix, 0), np.sum(matrix, 1)
    return np.log10(matrix / (sr * sc))

'''
Centered log-ratio normalization
'''
def clr(x):
    return np.log(x) - np.log(gmean(x))
    

