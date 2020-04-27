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

### Main RPLS Algorithm

class rpls_main:

    def __init__(self, params):
        for key, value in params.items():
            setattr(self, key, value)

    def rpls_algo(self, data, LV = None, max_iter = 500, tol = 1e-6):
        # data is in dataframe format
        # xcols and ycols in list format

        if (LV == None):
            LV = len(self.xcols_model)
        else:
            LV = LV

        n_vars = len(self.xcols_model)
        n_target = len(self.ycols)
        n_obs = data.shape[0]
        i = 0

        W = np.zeros(shape=(n_vars,LV))
        Q = np.zeros(shape=(n_target,LV))
        P = np.zeros(shape=(n_vars,LV))
        B = list()
        T = np.zeros(shape=(n_obs,LV))

        X = data[self.xcols_model].values
        y = data[self.ycols].values
        E = X.copy()
        F = y.copy()
        u_old = F[:,0].reshape(-1,1)

        iter_n = 0

        while (i < LV):
            while(iter_n <= max_iter):
                iter_n+=1
                w = (E.T@u_old)/(u_old.T@u_old)
                t = E@w/np.linalg.norm(E@w)
                q = F.T@t/np.linalg.norm(F.T@t)
                u_new = F@q
                tol_n = np.abs(np.linalg.norm(u_new - u_old))
                u_old = u_new
                if (tol_n <= tol):
                    break

            p = E.T@t
            b = u_old.T@t
            E_deflated = E - t@p.T
            E = E_deflated

            W[:,i] = w.reshape(n_vars)
            Q[:,i] = q.reshape(n_target)
            T[:,i] = t.reshape(n_obs)
            P[:,i] = p.reshape(n_vars)
            B.append(float(b))

            i+=1

        B = np.diag(B)

        return(T, P, Q, W, B)

    ### Cross Validation
    def rpls_cv(self, d_model):

        train = d_model.copy()
        train = train.reset_index(drop = True)

        ### PLS LOO CV
        loo = LeaveOneOut()

        error_trainCV_ncomp = []
        error_testCV_ncomp = []

        for ncomp in range(1,(len(self.xcols_model)+1)):
            error_trainCV = []
            error_testCV = []

            for train_index, test_index in loo.split(train):
                traincv, testcv = train.iloc[train_index], train.iloc[test_index]

                y_traincv = traincv[self.ycols].values
                y_testcv = testcv[self.ycols].values

                T_cv, P_cv, Q_cv, W_cv, B_cv = self.rpls_algo(traincv, LV = ncomp, max_iter = 500, tol = 1e-6)

                ypred_train = self.predict(traincv, T_cv, P_cv, Q_cv, W_cv, B_cv)
                ypred_test = self.predict(testcv, T_cv, P_cv, Q_cv, W_cv, B_cv)

                ypred_train = self.rescale(ypred_train, self.ycols)
                ypred_test = self.rescale(ypred_test, self.ycols)

                error_trainCV.append(mean_squared_error(ypred_train, y_traincv))
                error_testCV.append(mean_squared_error(ypred_test, y_testcv))

            error_trainCV_ncomp.append(sum(error_trainCV)/len(error_trainCV))
            error_testCV_ncomp.append(sum(error_testCV)/len(error_testCV))

        #converting list to dataframe
        error_LOO = pd.DataFrame(
        {'error_train': error_trainCV_ncomp,
         'error_test': error_testCV_ncomp
        }, index = range(1,(len(self.xcols_model)+1)))

        LV = error_LOO['error_test'].idxmin()
        return(error_LOO, LV)

    def predict(self, data, T, P, Q, W, B):
        coeff = W@(np.linalg.inv(P.T@W))@B@Q.T
        pred = data[self.xcols_model].values@coeff
        pred = self.rescale(pred, self.ycols)
        return(pred)

    def rescale(self, data, cols):
        data_rescale = (data * float(self.scale.loc[cols]))+ float(self.mean.loc[cols])
        return(data_rescale)
