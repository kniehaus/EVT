# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 16:01:36 2016

@author: trin2441
"""


# This script provides some sample data and illustration of the EVT methods



###############################################################################
# 
### Import
# 
###############################################################################

import numpy as np
import pandas as pd
import datetime 
import matplotlib.pylab as plt

import kn_handyFuncs as knh
from Toolbox_EVT import *

# (Change this)
pathStart = '/local/trin2441/Dropbox/Toolboxes/'   # path to where data lives

#%%
###############################################################################
# 
### Example with deduplication of lab values
# 
###############################################################################

# Load up sample data
# here, dates must be in datetime format
exFile = '{0}sampleLabData.csv'.format(pathStart)
labDFh = knh.readcsvAndReturnDF_generic(exFile)
labDFh['value'] = labDFh['value'].astype(float)  # make sure in correct format
labDFh['date'] = [datetime.datetime.strptime(a[:-1], '%Y-%m-%d %H:%M:%S') for a in labDFh['date']]
ptUIs = np.unique(labDFh['UI'].values)

# Options - adjust these
# Choose patient to look at
ptInd = 1
# Choose lab to look at ('CRP', 'Haemoglobin')
lab = 'Haemoglobin'
# Set cut-off threshold
cutoff = 11
# Set option for 'shortfall' or 'exceed' 
opt = 'shortfall'

# Get patient's data
df1 = labDFh.loc[(labDFh['UI']==ptUIs[ptInd]) & (labDFh['testname']==lab)]
df1 = df1.sort_values(by='date')  # to make graphing lines work out

# perform deduplication
df_dedup = deduplicateValues(df1, cutoff, numDays=10, opt=opt)

# plot to illustrate the difference
plt.figure(figsize=[12,5])
plt.subplot(1,2,1)
plt.plot(df1['date'], df1['value'], '.')
plt.plot(df1['date'], df1['value'])
plt.xlabel('Date'); plt.ylabel('Value'); plt.title('Raw')
plt.subplot(1,2,2)
plt.plot(df_dedup['date'], df_dedup['value'], '.')
plt.plot(df_dedup['date'], df_dedup['value'])
plt.xlabel('Date'); plt.ylabel('Value'); plt.title('Deduplicated')



#%%
###############################################################################
# 
### Example of fitting EVT model
# 
###############################################################################

# Load data
# here, dates must be in float format (time from some event)
#exFile2 = '{0}sampleLabData_4paramsFit_White Cells.csv'.format(pathStart)
exFile2 = '{0}sampleLabData_4paramsFit_Haemoglobin.csv'.format(pathStart)
singleLabDF = knh.readcsvAndReturnDF_generic(exFile2)
singleLabDF['value'] = singleLabDF['value'].astype(float)  # make sure in correct format
singleLabDF['time'] = singleLabDF['time'].astype(float)  # make sure in correct format



# Options - adjust these
k = 20    # number of samples to use in a block; 
timeBlock = 1    # length of time block in years
u = 11  # cutoff value
minMax = [0,40]  # min and max values allowed (for outlier screening)
negativeMult = -1  # =1 if exceedances; =-1 if shortfalls
logOpt = 1  # =1 if take log; =0 if not
blockVecsOpt = 1   # =1 if should take block vectors; =0 if not
columnsOpt = ['UI', 'value', 'time']  # names for relevant columns; to correspond with ['UI', 'value', 'time']

# Manipulate values as according to settings above
vals, u_trans, offset = processDataPreEVTfit(singleLabDF, u, k, timeBlock, minMax, negativeMult=negativeMult, logOpt=logOpt, blockVecsOpt=blockVecsOpt, columnsOpt = columnsOpt)

# Fit model with cross-validation
optParams, optNLL = xvalidatePPfit(vals, k, u_trans, timeBlock, numFolds=5)
# optParams: [lambda, mu, sigma, sigma_p, xi]

# make full set of parameters (need to record transformations performed above as well)
paramsFull = optParams + [u, offset, negativeMult, logOpt]

# NOTE: if change the block size later, need to adjust lambda to correct ratio



#%%
###############################################################################
# 
### Example of using Janossy density model
# 
# NOTE: assumes some fitted model parameters (e.g. above)
###############################################################################

# (use df1 as defined and deduplicated above)

# set time zero as first lab measurement
timeZero = min(df1['date'])  
# convert dates into time-from-event
df1['time'] = [timeDiffYears(tN,timeZero) for tN in df1['date']]

# Use patient's data to construct anomaly score with parameters fit above
anomDF = calcJanossyAnomalyScore(df1['value'], df1['time'], paramsFull, 0, np.max(df1['time']), windowSize=1.0, step=1/12.0)

# plot
plotAnomalyScores(df1, anomDF, u, '{0} value'.format(lab), ptInd)


### If want to change block size

# Adjust below
multWindowSize = 3   # multiplier by 1 year to adjust window size
step = 1/12.

# poisson lambda then needs to be updated as below 
paramsHere = copy.deepcopy(paramsFull)
paramsHere[0] = paramsHere[0]*multWindowSize
anomDF = calcJanossyAnomalyScore(df1['value'], df1['time'], paramsFull, 0, np.max(df1['time']), windowSize=multWindowSize, step=step)

# plot
plotAnomalyScores(df1, anomDF, u, '{0} value'.format(lab), ptInd)








