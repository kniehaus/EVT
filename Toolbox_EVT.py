# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 15:30:08 2016

@author: trin2441
"""

#### Set of functions for implementing extreme value theory methods

# Functions included: 
# - de-duplicating function
# - PP model fitting function
# - classical EVT evaluating function
# - Janossy density evaluating function


import numpy as np
import pandas as pd
import scipy.optimize
from copy import deepcopy
from sklearn.cross_validation import KFold
import scipy.stats as ss
import copy
import matplotlib.pylab as plt


def timeDiffYears(time1, time2):
    """
    Return the difference of time1-time2 (both are datetime
    objects) in terms of years
    
    (time1-time2).days/365.25
    
    """
    return (time1-time2).days/365.25

                
        
        
def deduplicateValues(df, cutoff, numDays=10, opt='exceed'):
    """
    This function will deduplicate the values in the df so that only the 
    maximum within a time-window of numDays remains.  This is the 
    'declustering' step often mentioned in the PP literature
    
    This is a recursive function.
    
    ----------
    
    Input
    
    df: pandas dataframe
        dataframe with the following columns: 
        - 'value': float values
        - 'date': date in datetime format
        (- 'UI': unique identifier for subject)
        (- 'testname': name for unique lab/biomarker)

    cutoff: float
        threshold to define an exceedance or shortfall

    numDays: int
        number of days on either side of max val to remove

    opt: string
        whether to remove values above the cutoff (='exceed') or below 
        (='shortfall')
        
    ----------
    
    Output
    
    df_dedup: pandas dataframe
        dataframe with 'value' and 'date' columns; datapoints within
        +/-numDays have been removed
    
    
    Written by KN, 3-Aug-2016
    
    [Original in R, from KN, 12-Jan-2016]
    
    """
    # sort df
    df = df.sort_values(by='date')
    # check whether any exceedances (or shortfalls)
    if (opt=='exceed' and np.sum(df['value']>cutoff)>1) or (opt=='shortfall' and np.sum(df['value']<cutoff)>1):
        if opt=='exceed':
            # get just the exceedances
            spikeInd = np.linspace(0,len(df)-1, len(df))[df['value'].values>cutoff].astype(int)
            spikeDF = df.loc[df['value']>cutoff]
            # sort them
            spikeInd = spikeInd[np.argsort(spikeDF['value'].values)]
            spikeInd = spikeInd[::-1]   # need to reverse the sorting
        elif opt=='shortfall':
            # get just the shortfalls
            spikeInd = []
            spikeInd = np.linspace(0,len(df)-1, len(df))[df['value'].values<cutoff].astype(int)
            spikeDF = df.loc[df['value']<cutoff]
            # sort them
            spikeInd = spikeInd[np.argsort(spikeDF['value'].values)]
        else:
            print('unrecognized option for \'opt\' specified')
            return df
        # remove within numDays
        for i in range(0,len(spikeInd)):
            indexN = spikeInd[i]
            # check before
            if indexN>0 and indexN<(len(df)-1):
                timeDiff = (df.iloc[indexN]['date'] - df.iloc[indexN-1]['date']).days
                if timeDiff < numDays:
                    df = df.drop(df.index[[indexN-1]])
                    df = deduplicateValues(df, cutoff, numDays=numDays, opt=opt)
                    if (opt=='exceed' and np.sum(df['value']>cutoff)<=1) or (opt=='shortfall' and np.sum(df['value']<cutoff)<=1):
                        return df
            # check after
            if indexN<(len(df)-1):
                timeDiff = (df.iloc[indexN+1]['date'] - df.iloc[indexN]['date']).days
                if timeDiff < numDays:
                    df = df.drop(df.index[[indexN+1]])
                    df = deduplicateValues(df, cutoff, numDays=numDays, opt=opt)
                    if (opt=='exceed' and np.sum(df['value']>cutoff)<=1) or (opt=='shortfall' and np.sum(df['value']<cutoff)<=1):
                        return df
        return df
    else:
        return df




def calcPPlikelihood(params, data, k, u, timeBlock=1):
    """
	This function calculates the negative log-likelihood of a poisson process with the
	given parameters, given the input data
	LL obtained through derivation presented by Coles - pg. 134, take LL of
	eq. 7.9
 
     ----------
	
	Input
 
	params : list of floats
         parameters: [mu sigma xi]
	
	data : vector
         vector of input data, Nx1 (both above & below threshold u)
         
	k : int
         block size
	
	u: float
         threshold
	
	timeBlock : float 
         Not yet implemented
	
     ----------
 
	Output
 
	NLL : float
         PP NLL of data, given params
	
 
	Written by KN, 3-Aug-2016.
 
	[Original in matlab, from KN, 8-Jan-2015.  
	Based largely upon lik_pp function by Stijn Luca (Aug 2014)]
 
    """
    # get above-threshold data
    aboveThreshInd = np.where(data>u)[0]
    # make params more readable
    mu = np.float(params[0])
    sigma = np.float(params[1])
    xi = np.float(params[2])
    # normalize
    u_norm = (u-mu)/np.float(sigma)
    data_norm = (data-mu)/np.float(sigma)
    # useful
    numSamples = len(data)
    numExceed = len(aboveThreshInd)
	
    # check for conditions
    conditionsMet  = 1
    if xi==0: conditionsMet=0
    if sigma<0: conditionsMet=0
    checkVals = 1+xi*data_norm
    if len(np.where(checkVals<0)[0])>0: conditionsMet=0
    if (1+xi*u_norm)<0: conditionsMet = 0
	
	# Calculate LL
    if conditionsMet==1:
        sumPrev = np.sum(np.log(1+xi*data_norm)[aboveThreshInd])
        NLL = (numSamples/np.float(k))*timeBlock*(1+xi*u_norm)**(-1/xi) + numExceed*np.log(sigma) + (1/xi + 1)*sumPrev
    else:
        NLL = 100000
	
    return NLL
	
	
	
	
def fitPP(data, k, u, timeBlock=1):
     """
     Finds the parameters that minimize the PP NLL. 	
    
     ----------
     
	Args:
     
	data : vector
         1-d vector of input data
 
	k : int
         block-size of model
 
	u : float
         threshold defining as extreme
 
	timeBlock : float
         length of time - TODO not implemented yet
    
     ---------
     
	Returns :
 
	pp_params: list
         list of fitted parameters: [lambda, mu, sigma, sigma_p, xi]
 
	NLL : float
         final NLL value
	
 
	Written by KN, 3-Aug-2016.
 
	[Original in matlab, from KN, 8-Jan-2015.  
	Based largely upon lik_pp function by Stijn Luca (Aug 2014)]
 
     """
     # set initial guesses
     guess_mu = np.mean(data)
     guess_sigma = np.std(data)
     guess_xi = 0.01
     params0 = [guess_mu, guess_sigma, guess_xi]
	
     # min search
     f = lambda params: calcPPlikelihood(params, data, k, u, timeBlock)
     optParams = scipy.optimize.fmin(func=f, x0 = params0)
	
     # corresponding LL
     NLL = calcPPlikelihood(optParams, data, k, u, timeBlock)
	
     # calculate lambda
     mu = np.float(optParams[0]); sigma = np.float(optParams[1]); xi = np.float(optParams[2])
     lambda_pp = (1 + xi*(u-mu)/sigma)**(-1/xi)
 
     # calculate sigma_p
     sigma_p = sigma + xi*(u-mu)
 
     pp_params = [lambda_pp, mu, sigma, sigma_p, xi]

     return pp_params, NLL




def xvalidatePPfit(data, k, u, timeBlock, numFolds=5):
    """
    Cross-validates with subsets of the data to determine the PP parameters.
    
    ----------
    
    Input
    
    data: vector
        1-d vector of input values
        
    k: int
        Number of samples per block
        
    u: float
        Cutoff threshold
        
    timeBlock: float
        Length of time represented by one block
        
    numFolds: int (default=5)
        number of folds of x-validation to run
        
        
    --------
    
    Output
    
    pp_params: list
        list of optimal PP parameters, as fit thru x-validation
        [lambda, mu, sigma, sigma_p, xi]
        
    NLL: optimal NLL
    
    
    Written by KN, 4-Aug-2016
    
    [Adapted from matlab function by KN, 16-Mar-2015]
    
    """
    # get folds
    skf = KFold(len(data), numFolds)
    test_NLLs = []
    trainParams = []
    for ktrainInd, ktestInd in skf:
        ktrain = data[ktrainInd]
        ktest = data[ktestInd]
        pp_params, NLL = fitPP(ktrain, k, u, timeBlock)
        # look at held out test set
        test_NLLs.append(calcPPlikelihood(pp_params, ktest, k, u, timeBlock))
        trainParams.append(pp_params)
    
    # find best
    test_NLLs = np.array(test_NLLs)
    minInd = np.argmin(test_NLLs)
    optParams = trainParams[minInd]
    optNLL = test_NLLs[minInd]
    return optParams, optNLL
    


def processDataPreEVTfit(df, u, k, timeBlock, minMax, negativeMult=1, logOpt=1, blockVecsOpt=1, columnsOpt = ['UI', 'value', 'time']):
    """
    This function performs the necessary pre-processing steps before the PP 
    model is fit.  These are: 
    
    * removing outliers
    * transforming (translation + negation) if looking at shortfalls
    * log transform 
    * formation of block vectors of time
    
    ----------
    
    Input
    
    df: pandas dataframe
        DF with a 'value' and a 'time' column.  'time' must be either floats 
        of years from some event [or datetime.datetime objects - not yet]
        
    u: float
        cutoff threshold
        
    k: int
        Number of samples per block
        
    timeBlock: float
        Length of time represented by one block (e.g. 0.5 year or 1.0 year)
    
    minMax: list or None
        [min max] range allowed for outliers
        If no outlier removal is to be performed, pass None

    negativeMult: int (default 1)
        Whether shortfalls should be analyzed (=-1), or exceedances (=1)

    logOpt: int (default 1)
        Whether the log should be taken (=1) or not (=0)
    
    blockVecsOpt: int (default 1)
        Whether to pad the data with extra samples to simulate more frequent
        sampling (=1) or not (=0)
    
    ----------
    
    Output
    
    vals: vector
        vector of data values, transformed as specified above
        
    u_trans: float
        value of u, the cutoff, transformed as the data has been
        
    offset: float
        size of numerical translation performed (=0 if exceedances)
    
    Written by KN, 4-Aug-2016
    [Consolidation of matlab code by KN, from 12-Oct-2015]
    
    """
    # get into format
    df['UI'] = df[columnsOpt[0]]
    df['value'] = df[columnsOpt[1]]
    df['time'] = df[columnsOpt[2]]
    df['value'] = df['value'].astype(float)
    try:
        df['time'] = df['time'].astype(float)    
    except:
        print('Need to put \'time\' column into appropriate format')
    
    # remove outliers
    if minMax is not None:
        df = df.loc[df['value']>minMax[0]]   
        df = df.loc[df['value']<minMax[1]]  
    
    # set fillval
    if negativeMult==1:
        fillVal = u*1.2
    else:
        fillVal = u*0.8
        
    u_trans = deepcopy(u)
    
    # transform if shortfalls
    offset = 0
    if negativeMult==-1:
        offset = np.ceil(1.5*np.max(df['value']))
        df['value'] = -df['value'] + offset
        # fill val
        fillVal = -fillVal + offset
        u_trans = -u_trans + offset
        
    # take log if desired
    if logOpt==1:
        eps = 0.00001
        df['value'] = np.log(df['value']+eps)
        # fillval
        fillVal = np.log(fillVal + eps)
        u_trans = np.log(u_trans)

    # pad to make block vectors
    if blockVecsOpt==1:
        vals = makeBlockedVectors(df, fillVal, k, timeBlock)
    else:
        vals = df['value'].values
        
    # permute
    vals = np.random.permutation(vals)
    
    return vals, u_trans, offset
    




def makeBlockedVectors(df, fillVal, k, timeBlock):
    """
    Return the input data with additional padded samples to approximate if the 
    data had been sampled more frequently

    ----------
    
    Input
    
    df: pandas dataframe
        dataframe with a 'value', 'time', and 'UI' column
        
    fillVal: float
        value of padding 
        
    k: int
        number of samples per time block
    
    timeBlock: float
        length (in time units - usu years) of time block
        
    
    ----------
    
    Output
    
    vals: vector
        vector of block-padded data values
    
    
    Written by KN, 4-Aug-2016
    
    [From matlab function by KN, 15-Apr-2016]
    
    """
    # get list of patients
    ptList = np.unique(df['UI'])
    
    dataBlocked = []
    # go through each patient
    for ui in ptList:
        subDF = df.loc[df['UI']==ui]
        times = subDF['time'].values
#        if isinstance(times[0], np.datetime64) or isinstance(times[0], datetime):
#            minTime = np.min(times)
#            times = [((t-minTime).days)/365.25 for t in times]  # returns days difference
#            times = times/365.25  # convert to years
        # get expected number of samples
        timeRange = np.max(times) - np.min(times)
        expN_samples = np.ceil((k/np.float(timeBlock))*timeRange)
        numAdd_samples = expN_samples - len(subDF)
        if numAdd_samples<0:
            numAdd_samples = 0
        # add to list
        dataBlocked.extend(subDF['value'].values)
        data2add = np.ravel(np.tile(fillVal, [1,numAdd_samples]))
        dataBlocked.extend(data2add)

    return dataBlocked




def calcJanossyAnomalyScore(values, times, paramsFull_o, firstTime, lastTime, windowSize=1.0, step=1/12.0):
    """
    Return the Janossy anomaly score, based upon fitted model parameters
    
    ----------
    
    Input
    
    values: vector of floats
        contains numerical values of lab or other metric
        
    times: vector of floats
        contains corresponding times 
        
    paramsFull: list
        [lambda, mu, sigma, sigma_p, xi, cutoff, offset, negativeMult, logOpt]
        data is transformed as follows: values*negative + offset
        So, offset=0, negative=1 if looking at exceedances
        Offset must be set and negative=-1 if looking at shortfalls
    
    firstTime: float
        time point at which calculation should begin 
        
    lastTime: float
        time point at which calculation should end
        
    windowSize: float (default = 1 year = 1.0)
        size of window over which score is calculated
        
    step: float (default = 1 month = 1/12.0)
        size of steps for score calculation 
        
    NOTE: firstTime and lastTime are the midpoints for the time window over 
    which the score will be calculated
    ----------
    
    Output
    
    anomDF: pandas dataframe
        dataframe with columns: 'time', 'c_anomScore', 'j_anomScore'
        time is the time mid-point at which the score is calculated
        c_anomalyScore is the anomaly score as calculated through classical EVT
        j_anomalyScore is the anomaly score using the Janossy model
        
    
    Written by KN, 10-Aug-2016
    
    [From python EVT function by KN, 9-Apr-2015]

    """
    halfWindowSize = np.float(windowSize)/2 
    epsilon = 0.0000001   # value to replace when taking logs of 0
    
    # transform values
    paramsFull = copy.deepcopy(paramsFull_o)
    offset = paramsFull[6]; negativeIndic = paramsFull[7]; u = paramsFull[5]; logOpt = paramsFull[8]
    vals_t = negativeIndic*values + offset
    paramsFull[5] = negativeIndic*u + offset
    if logOpt==1:
        vals_t = np.log(vals_t+epsilon)
        paramsFull[5] = np.log(paramsFull[5])

    # iterate through dates
    newTime = firstTime-halfWindowSize  # start of first window
    lastTime= lastTime-halfWindowSize   # start of last window
    
    timeVec = []; c_anomScore = []; j_anomScore = []
    while (lastTime-newTime)>0:
        # get values
        endTime = newTime+windowSize  # end point for window
        valSubset = [val for val, time in zip(vals_t, times) if (time-newTime)>0 and (time-endTime)<0]

        # calculate classical EVT score
        poissonScore, gpdScore, gevScore = calcClassicAnomalyScore(valSubset, paramsFull)
        c_anomScore.append(np.sum([poissonScore, gpdScore, gevScore]))

        # calculate Janossy model score
        pdf_val, cdf_val, janossy_density = calcJanossyAnomalyProbabilities(valSubset, paramsFull)
        j_anomScore.append(-np.log(pdf_val))

        # add date to timeVec
        timeVec.append(newTime+halfWindowSize)

        # iterate
        newTime = newTime+step

    anomDF = pd.DataFrame(timeVec, columns=['time'])
    anomDF['c_anomScore'] = c_anomScore
    anomDF['j_anomScore'] = j_anomScore
    
    return anomDF




def calcClassicAnomalyScore(vals, paramsFull):
    """ 
    
    Return the -log of poisson pmf, gev probability, and mean of gpd 
    probability for the Poisson Process model
    
    Note!!: assumes data and cutoff already transformed; 
    looking only at exceedances
    
    ----------
    
    Input
    
    vals: vector of floats
        contains numerical values of lab or other metric from time window 
        of interest
        
    paramsFull: list
        [lambda, mu, sigma, sigma_p, xi, cutoff, offset, negativeMult, logOpt]
        last 3 params not used here

    ----------
    
    Output
    
    poissonScore: float
        -log(poisson PMF)    
    
    gpdScore: float
        -log(mean(GPD prob))
    
    gevScore: float
        -log(GEV prob)
    
    Written by KN, 10-Aug-2016
    
    [From python EVT function by KN, ~9-Apr-2015]
    
    """
    # define params    
    mylambda = paramsFull[0]; mu = paramsFull[1]; sigma = paramsFull[2]
    sigma_p = paramsFull[3]; xi = paramsFull[4]; cutoff = paramsFull[5]
    # get extremes
    vals = np.array(vals)
    indsKeep = np.where(vals>cutoff)[0]
    valsProcessed = vals[indsKeep]
    # calculate scores
    if len(valsProcessed)>=1:
        # poisson
        poissonPMF = (ss.poisson.pmf(len(valsProcessed), mylambda))
        if poissonPMF==0:
            poissonScore = 0
        else:
            poissonScore = -np.log(poissonPMF)
        # gpd
        gpdProb = 1-np.mean(ss.genpareto.cdf(valsProcessed, c=xi, loc=mu, scale=sigma_p))   
        if gpdProb==0:
            gpdScore = 0
        else:
            gpdScore = -np.log(gpdProb)
        # gev
        gevProb = 1-ss.genextreme.cdf(max(vals), c=xi, loc=mu, scale=sigma)
        if gevProb==0:
            gevScore = 0
        else:
            gevScore = -np.log(gevProb)
        return poissonScore, gpdScore, gevScore
    else:
        return 0, 0, 0
        
        
        
        
def calcJanossyAnomalyProbabilities(vals, params):
    """ 
    Return the PDF and CDF values for a patient's given set of lab values, 
    with the input parameters to describe the Janossies
    
    
    Not yet public  
    
    """
    # (Not public)
    return pdf_val, cdf_val, janossy_density
    








def plotAnomalyScores(df1, anomDF, u, ylab, ptIndice):
    """
    
    Plot the raw data and classical and janossy density anomaly scores
    
    ----------
    
    Input
    
    df1: pandas dataframe
        dataframe containing 'time' and 'value' columns, containing the 
        raw data
        
    anomDF: pandas dataframe
        dataframe containing 'time' and 'c_anomScore' and 'j_anomScore'
        columns, containing the fitted anomaly scores
        
    u: float
        cutoff threshold for fitting PP model (will be plotted as horizontal
        line on graph)
        
    ylab: string
        label for y-axis
        
    pIndice: float or int or string
        identifier for given subject that is being plotted
        
        
    ----------
    
    Output
    
    (none): graph
        Will produce a 1x2 subplot graph with the raw data on the left and 
        the anomaly scores on the right

    
    Written by KN, 12-Aug-2016
    
    """
    
    plt.figure(figsize=[12,5])
    
    # raw data
    plt.subplot(1,2,1)
    plt.plot(df1['time'], df1['value'])
    plt.xlabel('Time from start of labs')
    plt.ylabel(ylab)
    plt.title('Pt {0} raw data'.format(ptIndice))
    plt.hlines(u, np.min(df1['time']), np.max(df1['time']), linestyle='--')
    
    # anomaly scores
    plt.subplot(1,2,2)
    plt.plot(anomDF['time'], anomDF['c_anomScore'], label='Classical')
    plt.plot(anomDF['time'], anomDF['j_anomScore'], label='Janossy')
    plt.xlabel('Time from start of labs')
    plt.ylabel('Anomaly score')
    plt.legend(loc='best')
    plt.title('Pt {0} anomaly scores'.format(ptIndice))
    plt.tight_layout()



