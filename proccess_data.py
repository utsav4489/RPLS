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

### Loading Dataset
class process_data():

    shift_params = pd.read_csv('config/shift_params.csv')
    cols_init = shift_params['variables'].to_list()
    shift_params = shift_params.set_index('variables')
    ycols = ['y_target']
    xcols_init = [col for col in cols_init if 'y_target' not in col]
    xcols_t1 = [col + '_t1' for col in xcols_init]
    xcols_t2 = [col + '_t2' for col in xcols_init]
    xcols = xcols_t1 + xcols_t2
    xcols_model = xcols + ['intercept']
    cols_total = xcols_model + ycols
    cols_process = xcols + ycols

    td = ['t1', 't2']

    def read(self, data):
        data[['Datetime']] = data[['Datetime']].apply(pd.to_datetime, dayfirst = True, errors = 'coerce')
        data[data.columns.drop('Datetime')] = data[data.columns.drop('Datetime')].apply(pd.to_numeric, errors = 'coerce')
        data = data.set_index('Datetime')
        data = data[self.cols_init]
        return(data)

    def time_shift(self, data):
        for col in self.xcols_init:
            data[col + '_t1'] = data[col].shift(int(self.shift_params.loc[col, self.td[0]]))
            data[col + '_t2'] = data[col].shift(int(self.shift_params.loc[col, self.td[1]]))
        data = data[self.xcols + self.ycols]
        return(data)

    def load_dataset(self, data):
        data = self.read(data)
        data = self.time_shift(data)
        data = data.dropna()
        return(data)

    def params(self, data, n_base):
        data = data[self.cols_process].iloc[:n_base]
        mean = data.mean(axis=0, skipna = True)
        scale = data.mean(axis=0, skipna = True)
        return(mean, scale)
