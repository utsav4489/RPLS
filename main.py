"""
main python script to run the script

@author: utsav
"""
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import datetime as dt
import math
import time

from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut

from proccess_data import process_data
from rpls_main import rpls_main
from rpls_master import rpls_master
from rpls_misc import rpls_misc
from rpls_statistics import rpls_statistics

# to control the monitoring iteration
print_iter = 1000
n_iters = 75000

##### reading dataset #####
prdata = pd.read_excel('config/Dataset/process_data.xlsx', skiprows=[0])
print('prdata loaded')
#####

##### params loading and loading dataset #####
# dict that will have all the hyperparamaters and colnames
params = dict()

# reading hyper params from connfig file
hparams = pd.read_csv('config/hyperparams.csv')
hparams = hparams.set_index('hyperparams')

n_base = int(hparams.loc['n_base', 'value'])
n_sm = int(hparams.loc['n_sm', 'value'])
n_add = int(hparams.loc['n_add', 'value'])
w_spe = int(hparams.loc['w_spe', 'value'])
w_tsq = int(hparams.loc['w_tsq', 'value'])

### loading dataset
pr = process_data()
data_main = pr.load_dataset(prdata)
###

# calculating mean and scale of model based on input data and hyperparams i.e. n_base
mean, scale = pr.params(data_main, n_base)

# reading all column names
xcols_model = pr.xcols_model
xcols = pr.xcols
ycols = pr.ycols
cols_process = pr.cols_process
cols_total = pr.cols_total

# column names dictionary
cparams = dict({'xcols_model': xcols_model,
                'xcols': xcols,
                'ycols': ycols,
                'cols_process': cols_process,
                'cols_total': cols_total})

# hyperparams dictionary
mparams = dict({'scale': scale,
                'mean': mean,
                'n_base': n_base,
                'n_sm': n_sm,
                'n_add': n_add,
                'w_spe': w_spe,
                'w_tsq': w_tsq})

# total params concatenated
params = {**cparams, **mparams}

##### end of section #####

##### loading classes #####
# loading classes
algo = rpls_main(params)
fn = rpls_misc(params)
stats = rpls_statistics(params)
#####

##### building first model instance #####
master = rpls_master(params)
T_c, P_c, Q_c, W_c, B_c, d_model, d_base_s, d_sm_s = master.build_init(data_main)
#####

##### preparing for model iteration #####
# cache for calculating variance in scores required for tsq
# implemented in moving window format
d_cache = pd.concat([d_base_s, d_sm_s], axis=0)
tsq_cache = d_cache.iloc[-(n_base + n_sm):]

# data left after utilizing it for developing first instance
# data is scaled together as scaling params are constant
data_iter = data_main.iloc[(n_base + n_sm):].copy()
data_iterS = fn.data_structure(data_iter, cols_process)

# d_add that will be used to store next sequence of data until model updation is active
# it reset after model updation and the sequence repeats
d_add_s = pd.DataFrame(columns = cols_total)

# storing actual values and timestamps
actual_ts = data_iter[ycols].values.reshape(-1,).tolist()
ts = data_iter.index.tolist()

### creating files if they don't exist
# checking dir if it exists - clearing the dir if it exists
folder = 'output'
if not os.path.isdir(folder):
    os.mkdir(folder)
else:
    filelist = [ f for f in os.listdir(folder)]
    for f in filelist:
        os.remove(os.path.join(folder, f))

# files to be created
filenames = ['actual_ts', 'pred_pi_ts', 'tsq_analysis', 'spetot_analysis',
            'spevar_analysis', 'spevarlim_analysis', 'error_lv_model']


for f in filenames:
    ext = '.csv'
    if not os.path.exists(os.path.join(folder, f) + ext):
        with open(f_name, 'w'): pass

# storing timestamp and actual target data
for _ts, _actual_ts in zip(ts, actual_ts):
    master.save_data([_ts, _actual_ts], 'actual_ts')

 ###

 ##### end-of-section #####

 #### data iteration for calculating prediction, pi, tsq, tsq_lim, spe_tot, spe_tot_lim
 # spe_var, spe_var_lim, (error_model, lv) when model is updated

tsq_ts = []
spe_tot_ts = []
spe_var_ts = []

# timing the code
timeperloop = []
start_time = time.time()
# for row in range(data_iterS.shape[0]):
for row in range(n_iters):

    # for monitoring progress
    ttop_time = time.time()
    if (row % print_iter == 0):
        print('current row:', row)

### model updation
    if (d_add_s.shape[0] == n_add):
        T_c, P_c, Q_c, W_c, B_c, d_model, d_sm_s = master.model_update(d_model, d_sm_s, d_add_s)
        d_add_s = pd.DataFrame(columns = cols_total)
        error_model = master.error.loc[master.LV].tolist()
        # saving the data
        master.save_data(error_model + [master.LV], 'error_lv_model')
        savename = os.path.join(folder, 'error') + str(row)
        master.error.to_csv(savename)
        print('model updated at row:', row)
###

### current obs and appending it to d_add
    data = data_iterS.iloc[[row]]
    d_add_s = d_add_s.append(data, ignore_index = False, sort = False)
###

### prediction and pi calculation
    pred = algo.predict(data, T_c, P_c, Q_c, W_c, B_c)
    pi = stats.pred_interval(d_model, data, T_c, P_c, Q_c, W_c, B_c)
    # saving the data
    to_save = [float(pred)] + [float(pi)]
    master.save_data(to_save, 'pred_pi_ts')
###

### tsq calculation
    tsq = stats.hotelings_tsq(P_c, W_c, tsq_cache, data)
    # data as per window size preserved for lim calculation
    tsq_ts = master.cache_limit(tsq_ts, w_tsq)
    tsq_ts.append(tsq)

    # tsq_cache for calculating scores' variance
    tsq_cache = tsq_cache.iloc[1:]
    tsq_cache = pd.concat([tsq_cache, data], axis=0, sort=False)

    # tsq lim calculation
    if (len(tsq_ts) >= w_tsq):
        tsq_lim = stats.confidence_limit(tsq_ts[-w_tsq:])
    else:
        tsq_lim = -1000
    # saving the data
    to_save = [tsq] + [tsq_lim]
    master.save_data(to_save, 'tsq_analysis')
###

### spe calculation
    spe_tot, spe_var = stats.spe(P_c, W_c, data)
    # data as per window size preserved for lim calculation
    spe_tot_ts = master.cache_limit(spe_tot_ts, w_spe)
    spe_var_ts = master.cache_limit(spe_var_ts, w_spe)

    spe_tot_ts.append(spe_tot)
    spe_var_ts.append(spe_var)

    # spe lim calculation
    if (len(spe_tot_ts) >= w_spe):
        spe_tot_lim = stats.confidence_limit(spe_tot_ts[-w_spe:])
    else:
        spe_tot_lim = -1000
    # saving data
    to_save = [spe_tot] + [spe_tot_lim]
    master.save_data(to_save, 'spetot_analysis')

    # spe for each variable
    spe_var_ts_narray = np.array(spe_var_ts).copy()

    # calculating lim for each column i.e. variable
    spe_var_lim = []
    if (spe_var_ts_narray.shape[0] >= w_spe):
        for icol in range(spe_var_ts_narray.shape[1]):
            temp = stats.confidence_limit(spe_var_ts_narray[-w_spe:,icol])
            spe_var_lim.append(temp)
    else:
        for icol in range(spe_var_ts_narray.shape[1]):
            temp = -1000
            spe_var_lim.append(temp)
    # saving data
    # to_save = spe_var spe_var_lim
    master.save_data(spe_var, 'spevar_analysis')
    master.save_data(spe_var_lim, 'spevarlim_analysis')

    # each loop time
    tbot_time = time.time()
    timeperloop.append((tbot_time - ttop_time)/60)

end_time = time.time()
print('total_time:', (end_time - start_time)/60)

for _tloop in timeperloop:
    master.save_data([_tloop], 'timeperloop')
###

##### end-of-section #####
