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

from rpls_main import rpls_main
from rpls_misc import rpls_misc
from rpls_statistics import rpls_statistics

class rpls_master:

    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)

        self.fn = rpls_misc(params)
        self.algo = rpls_main(params)
        self.stats = rpls_statistics(params)

    def build_init(self, data):
        d_base = data.iloc[:self.n_base].copy()
        d_sm = data.iloc[self.n_base:(self.n_base + self.n_sm)].copy()

        d_base_s = self.fn.data_structure(d_base, self.cols_process)
        d_sm_s = self.fn.data_structure(d_sm, self.cols_process)

        T_base, P_base, Q_base, W_base, B_base = self.algo.rpls_algo(d_base_s)
        L_baseT = pd.DataFrame(P_base.T, columns=self.xcols_model)
        R_baseT = pd.DataFrame(B_base@Q_base.T, columns=self.ycols)

        T_sm, P_sm, Q_sm, W_sm, B_sm = self.algo.rpls_algo(d_sm_s)
        L_smT = pd.DataFrame(P_sm.T, columns=self.xcols_model)
        R_smT = pd.DataFrame(B_sm@Q_sm.T, columns=self.ycols)

        ## Combining Base and SM Model
        d_model = pd.concat([pd.concat([L_baseT, L_smT], axis=0), pd.concat([R_baseT, R_smT], axis=0)], axis=1)

        self.error, self.LV = self.algo.rpls_cv(d_model)

        ## Predictor Model Development
        T_c, P_c, Q_c, W_c, B_c = self.algo.rpls_algo(d_model, LV = self.LV)

        return(T_c, P_c, Q_c, W_c, B_c, d_model, d_base_s, d_sm_s)

    ## sript functions
    def model_update(self, d_model, d_sm_s, d_add_s):
        d_sm_s, d_sub_s = self.fn.shuffling(d_sm_s, d_add_s)
        d_base_s = pd.concat([d_model, d_sub_s], axis=0, sort=False)

        T_base, P_base, Q_base, W_base, B_base = self.algo.rpls_algo(d_base_s)
        L_baseT = pd.DataFrame(P_base.T, columns=self.xcols_model)
        R_baseT = pd.DataFrame(B_base@Q_base.T, columns=self.ycols)

        T_sm, P_sm, Q_sm, W_sm, B_sm = self.algo.rpls_algo(d_sm_s)
        L_smT = pd.DataFrame(P_sm.T, columns=self.xcols_model)
        R_smT = pd.DataFrame(B_sm@Q_sm.T, columns=self.ycols)

        ## Combining Base and SM Model
        d_model = pd.concat([pd.concat([L_baseT, L_smT], axis=0), pd.concat([R_baseT, R_smT], axis=0)], axis=1)

        self.error, self.LV = self.algo.rpls_cv(d_model)

        ## Predictor Model Development
        T_c, P_c, Q_c, W_c, B_c = self.algo.rpls_algo(d_model, LV = self.LV)

        return(T_c, P_c, Q_c, W_c, B_c, d_model, d_sm_s)

    def cache_limit(self, l, l_lim):
        if (len(l) == l_lim):
            l = l[1:]
        return(l)

    def save_data(self, data, f_name, folder = 'output', f  = '.csv'):
        ff_name = os.path.join(folder, str(f_name)) + str(f)
        with open(ff_name, 'a', newline='') as write:
            w = csv.writer(write, dialect='excel')
            w.writerow(data)
