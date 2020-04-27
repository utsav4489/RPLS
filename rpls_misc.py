import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import datetime as dt
import math
from scipy.stats import chi2

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut

class rpls_misc:
    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def scaling(self, data, cols):
        for col in cols:
            data[[col]] = data[[col]].apply(lambda x: (x - float(self.mean.loc[col]))/float(self.scale.loc[col]))
        return(data)

    def shuffling(self, data1, data2):
        d_sub = data1.iloc[:self.n_add].copy()
        data1 = data1.iloc[self.n_add:].copy()
        data1 = pd.concat([data1, data2], axis=0)
        return(data1, d_sub)

    def data_structure(self, data, cols):
        data = self.scaling(data.copy(), cols)
        data = data.assign(intercept = 1/np.sqrt(self.n_base-1))
        data = data[self.cols_total]
        return(data)
