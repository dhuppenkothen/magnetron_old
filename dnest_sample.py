
import numpy as np

import burstmodel
from pylab import *
import scipy.stats

def read_dnest_results(filename, dnestdir="./"):

    """
    Read output from RJObject/DNest3 run and return in a format more
    friendly to post-processing.

    filename: filename with posterior sample (probably sample.txt)

    NOTE: parameters (amplitudes + background) are in COUNTS space, not COUNT RATE!
    """


    #options = burstmodel.conversion("%sOPTIONS.txt" %dnestdir)

    dfile = open("%s%s" %(dnestdir, filename), "r")
    data = dfile.readlines()

    alldata = []
    for d in data[2:]:
        alldata.append(np.array([float(t) for t in d.split()[:-1]]))

    alldata = np.array(alldata)


    ## background parameter
    bkg = alldata[:,0]
    #bkg = np.array([float(t) for t in data[0]])

    ## dimensions of parameter space of individual  model components
    burst_dims =  alldata[:,1]
    burst_dims = list(set(burst_dims))[0]

    ## total number of model components permissible in the model
    compmax = alldata[:,2]
    compmax = list(set(compmax))[0]

    ## hyper-parameter (mean) of the exponential distribution used
    ## as prior for the spike amplitudes
    ## NOTE: IN LINEAR SPACE, NOT LOG
    hyper_mean_amplitude = alldata[:,3]

    ## hyper-parameter (mean) for the exponential distribution used
    ## as prior for the spike rise time
    ## NOTE: IN LINEAR SPACE, NOT LOG
    hyper_mean_risetime = alldata[:,4]

    ## hyper-parameters for the lower and upper limits of the uniform
    ## distribution osed as a prior for the skew
    hyper_lowerlimit_skew = alldata[:,5]
    hyper_upperlimit_skew = alldata[:,6]

    ## distribution over number of model components
    nbursts = alldata[:, 7]

    ## peak positions for all model components, some will be zero
    pos_all = alldata[:, 8:8+compmax]

    ## amplitudes for all model components, some will be zero
    amp_all = alldata[:, 8+compmax:8+2*compmax]

    ## rise times for all model components, some will be zero
    scale_all = alldata[:, 8+2*compmax:8+3*compmax]

    ## skew parameters for all model components, some will be zero
    skew_all = alldata[:, 8+3*compmax:8+4*compmax]

    ## pull out the ones that are not zero
    paras_real = []

    for p,a,sc,sk in zip(pos_all, amp_all, scale_all, skew_all):
        paras_real.append([(pos,amp,scale,skew) for pos,amp,scale,skew in zip(p,a,sc,sk) if pos != 0.0])




    sample_dict = {"bkg":bkg, "cdim":burst_dims, "cmax":compmax, "parameters":paras_real}

    return sample_dict


def extract_real_spikes(sample_dict, min_scale=1.0e-4):

    paras = sample_dict["parameters"]
    bkg = sample_dict["bkg"]

    ### make a Poisson distribution with the mean of the background parameter
    ### as distribution mean
    pois = scipy.stats.poisson(np.mean(bkg))

    ### extract 0.99 quantiles
    min_bkg, max_bkg = pois.interval(0.99)

    paras_filtered = []
    for par in paras:
        p_temp = [p for p in par if p[2] >= min_scale/5.0 and  p[1] >= max_bkg]
        if len(p_temp) > 0:
            paras_filtered.append(p_temp)
        else:
            continue


    sample_dict["filtered parameters"] = paras_filtered

    return sample_dict