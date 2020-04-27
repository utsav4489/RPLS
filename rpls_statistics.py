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

# rpls statistics
class rpls_statistics:

    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)

        self.algo = rpls_main(params)

    def hotelings_tsq(self, P, W, tsq_cache, obs):
        data = pd.concat([tsq_cache, obs], axis=0)

        x_rotation = np.dot(W, np.linalg.inv(P.T@W))
        x_scores = data[self.xcols_model].values@x_rotation
        x_var = np.var(x_scores, axis=0)

        t1sq_by_Sisq = x_scores**2/x_var
        tsq = np.sum(t1sq_by_Sisq, axis = 1)

        return(float(tsq[-1]))

    def spe(self, P, W, data):

        x_rotation = np.dot(W, np.linalg.inv(P.T@W))
        x_scores = data[self.xcols_model].values@x_rotation
        q_residual = data[self.xcols_model].values - np.matmul(x_scores, P.T)

        spe_var = np.square(q_residual)
        spe_tot = np.sum(np.square(q_residual), axis=1)

#         y_scores = np.dot(x_scores, B)
#         q_residualy = data[self.ycols].values - np.matmul(y_scores, Q.T)
#         spe_target = np.sum(np.square(q_residualy), axis=1)

        return(float(spe_tot[-1]), spe_var[-1].tolist())

    def confidence_limit(self, data, alpha = 0.05):
        mean = np.mean(data)
        var = np.var(data)

        g = var/(2*mean)
        h = (2*mean**2)/var

        lim = g*chi2.ppf((1-alpha), h)
        return(lim)

    def pred_interval(self, data, obs, T, P, Q, W, B):

        pred = self.algo.predict(data, T, P, Q, W, B)
        actual = data[self.ycols].values
        error = pred - actual
        sd = np.std(error)
        se = sd/np.sqrt(error.shape[0])

        X = data[self.xcols_model].values
        x_obs = obs[self.xcols_model].values
        h0 = x_obs@np.linalg.inv(X.T@X)@x_obs.T
        s = se*np.sqrt(1+h0+1/error.shape[0])

        interval = 1.96*s*float(self.scale.loc[self.ycols])

        return(interval)
