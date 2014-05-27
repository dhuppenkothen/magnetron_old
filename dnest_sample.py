
import numpy as np
import glob

import burstmodel
import parameters
import word

from pylab import *
import scipy.stats

def plot_posterior_lightcurves(datadir="./", nsims=10):

    files = glob.glob("%s*posterior*"%datadir)

    for f in files:
        fsplit = f.split("_")
        data = loadtxt("%s_%s_all_data_new.dat"%(fsplit[0], fsplit[1]))
        fig = figure(figsize=(12,9))
        plot(data[:,0], data[:,1], lw=2, color="black", linestyle="steps-mid")
        sample = atleast_2d(loadtxt(f))
        print(sample.shape)
        ind = np.random.choice(np.arange(len(sample)), replace=False, size=10)
        for i in ind:
            print("shape data: " + str(len(data[:,0])))
            print("shape sample: " + str(len(sample[i,-data.shape[0]:])))
            plot(data[:,0], sample[i,-data.shape[0]:], lw=1)
        xlabel("Time since trigger [s]", fontsize=20)
        ylabel("Counts per bin", fontsize=20)
        savefig("%s_%s_lc.png"%(fsplit[0], fsplit[1]), format="png")
        close()

    return


def






##### OLD CODE: NEED TO CHECK THIS! ##########


def read_dnest_results(filename, datadir="./"):

    """
    Read output from RJObject/DNest3 run and return in a format more
    friendly to post-processing.

    filename: filename with posterior sample (posterior_sample.txt)

    NOTE: parameters (amplitudes + background) are in COUNTS space, not COUNT RATE!
    """


    #options = burstmodel.conversion("%sOPTIONS.txt" %dnestdir)

    dfile = "%s%s" %(datadir, filename)
    alldata = np.loadtxt(dfile)

    niter = len(alldata)

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
        paras_real.append([(pos,scale,amp,skew) for pos,amp,scale,skew in zip(p,a,sc,sk) if pos != 0.0])


    sample_dict = {"bkg":bkg, "cdim":burst_dims, "nbursts":nbursts, "cmax":compmax, "parameters":paras_real}

    return sample_dict

def extract_param(sample_dict, i, filtered=True):

    if filtered:
        params = sample_dict["filtered parameters"]
    else:
        params = sample_dict["parameters"]

    output_par = []
    for par in params:
        par_temp = [p[i] for p in par]
        output_par.append(par_temp)

    return np.array(output_par)

def flatten_param(par):

    par_new = []
    for p in par:
        par_new.extend(p)

    return np.array(par_new)

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
        p_temp = [p for p in par if p[2] >= min_scale/5.0 and p[1] >= max_bkg]
        if len(p_temp) > 0:
            paras_filtered.append(p_temp)
        else:
            continue


    sample_dict["filtered parameters"] = paras_filtered

    return sample_dict


def position_histogram(sample_dict, btimes, tsearch=0.01, tfine=0.001, niter=1000):

    """
    Make a histogram of the distribution of position parameters.
    sample_dict: dictionary with output from Dnest run
    btimes: burst start and end times
    tsearch: length of segment over which to integrate probability
    tfine: histogram bin size
    niter: number of iterations in the DNest run
    """

    ## extract parameters
    positions = []
    if "filtered parameters" in sample_dict.keys():
        fparams = sample_dict["filtered parameters"]
    else:
        fparams = sample_dict["parameters"]

    ## extract positions only
    for fpar in fparams:
        ptemp = [f[0] for f in fpar]
        positions.extend(ptemp)

    ## burst duration

    tstart = btimes[0]
    tend = btimes[1]

    tseg = tend - tstart

    ## number of bins in histogram
    nbins = int(tseg/tfine)+1
    range = np.arange(nbins)*tfine + tstart[0]

    ## make histogram
    n, bins = np.histogram(positions, bins=range, normed=True)
    n_normed = n*np.diff(bins)

    ## number of histogram bins to integrate
    nsearch = int(tsearch/tfine)

    nsum_all = []
    for i in xrange(len(n[:-nsearch])):
        n_temp = n_normed[i:i+nsearch]
        nsum_all.append(np.sum(n_temp))


    return


def parameter_sample(filename, datadir="./"):

    ### extract parameters from file
    sample_dict = read_dnest_results(filename, datadir=datadir)


    ### I need the parameters, the number of components, and the background parameter
    pars_all = sample_dict["parameters"]
    nbursts_all = sample_dict["nbursts"]
    bkg_all = sample_dict["bkg"]

    parameters_all = []
    for pars,nbursts,bkg in zip(pars_all, nbursts_all, bkg_all):

        pars_flat = np.array(pars).flatten()
        pars_flat = list(pars_flat)
        pars_flat.extend([bkg])
        pars_flat = np.array(pars_flat)

        p = parameters.TwoExpCombined(pars_flat, int(nbursts), log=False, bkg=True)

        parameters_all.append(p)

    return parameters_all


def make_model_lightcurves(samplefile, times=None, datadir="./"):

    if times is None:
        fsplit = samplefile.split("_")
        datafile = "%s%s_%s_data.dat"%(datadir, fsplit[0], fsplit[1])

        ### load data
        times, counts = burstmodel.read_gbm_lightcurves(datafile)

    else:
        counts = np.ones(len(times))

    parameters_all = parameter_sample(samplefile, datadir)

    model_counts_all = []

    for p in parameters_all:
        ncomp = len(p.all)
        wordlist = [word.TwoExp for i in xrange(ncomp)]
        bd = burstmodel.BurstDict(times, counts, wordlist)
        model_counts = bd.model_means(p)

        model_counts_all.append(model_counts)

    return model_counts_all
