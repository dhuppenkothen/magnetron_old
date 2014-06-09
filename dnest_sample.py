
import numpy as np
import glob

import burstmodel
import parameters
import word

from pylab import *
import matplotlib.cm as cm
import scipy.stats

def plot_posterior_lightcurves(datadir="./", nsims=10):

    files = glob.glob("%s*posterior*"%datadir)

    for f in files:
        fsplit = f.split("_")
        data = loadtxt("%s_%s_all_data.dat"%(fsplit[0], fsplit[1]))
        fig = figure(figsize=(24,9))
        ax = fig.add_subplot(121)
        plot(data[:,0], data[:,1], lw=2, color="black", linestyle="steps-mid")
        sample = atleast_2d(loadtxt(f))

        print(f)
        print(sample.shape)

        ind = np.random.choice(np.arange(len(sample)), replace=False, size=nsims3)
        for i in ind:
            #print("shape data: " + str(len(data[:,0])))
            #print("shape sample: " + str(len(sample[i,-data.shape[0]:])))
            plot(data[:,0], sample[i,-data.shape[0]:], lw=1)
        xlabel("Time since trigger [s]", fontsize=20)
        ylabel("Counts per bin", fontsize=20)

        ax = fig.add_subplot(122)
        nbursts = sample[:, 7]

        hist(nbursts, bins=30, range=[np.min(nbursts), np.max(nbursts)], histtype='stepfilled')
        xlabel("Number of spikes per burst", fontsize=20)
        ylabel("N(samples)", fontsize=20)
        savefig("%s_%s_lc.png"%(fsplit[0], fsplit[1]), format="png")
        close()

    return


def extract_sample(datadir="./", nsims=5, trigfile=None):

    files = glob.glob("%s*posterior*"%datadir)
    print("files: " + str(files))

    all_parameters, bids, nsamples = [], [], []
    for f in files:
        #parameters = parameter_sample(f, trigfile=trigfile)
        fname = f.split("/")[-1]
        bid = fname.split("_")[0]
        bids.append(bid)
        parameters = parameter_sample(f, trigfile=trigfile)

        all_parameters.append(parameters)
        nsamples.append(len(parameters))

    if nsims > np.min(nsamples):
        nsims = np.min(nsamples)
        print("Number of desired simulations larger than smallest posterior sample.")
        print("Resetting nsims to %i" %nsims)

    parameters_red = np.array([np.random.choice(p, replace=False, size=nsims) for p in all_parameters])
    print("shape of reduced parameter array: " + str(parameters_red.shape))

    return parameters_red, bids

def risetime_amplitude(datadir="./", nsims=5, dt=0.0005):


    parameters_red, bids = extract_sample(datadir, nsims)
    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)

    risetime_sample, amplitude_sample = [], []

    for i in xrange(nsims):

        sample = parameters_red[:,i]
        risetime_all = np.array([np.array([a.scale for a in s.all]) for s in sample])

        #risetime_all = risetime_all.flatten()
        amplitude_all = np.array([np.array([a.amp for a in s.all]) for s in sample])
        #amplitude_all = amplitude_all.flatten()

        risetime, amplitude = [], []
        for r,a in zip(risetime_all, amplitude_all):
            risetime.extend(r)
            amplitude.extend(a)

        risetime_sample.append(risetime)
        amplitude_sample.append(amplitude)


    sp_all = []

    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    for i,(r,a) in enumerate(zip(risetime_sample, amplitude_sample)):
        a = np.array(a)/0.0005
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)
        logr = np.log10(r)
        loga = np.log10(a)
        scatter(logr,loga, color=cm.jet(i*50))

    axis([np.min([np.min(np.log10(r)) for r in risetime_sample]),
          np.max([np.max(np.log10(r)) for r in risetime_sample]),
          np.min([np.min(np.log10(a)) for a in amplitude_sample]),
          np.max([np.max(np.log10(a)) for a in amplitude_sample])])

    xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
    ylabel("log(spike amplitude)", fontsize=20)
    title("spike amplitude versus rise time")
    savefig("risetime_amplitude.png", format="png")
    close()

    return risetime_sample, amplitude_sample, sp_all


def risetime_energy(datadir="./", nsims=5, dt=0.0005):


    parameters_red,bids = extract_sample(datadir, nsims)
    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    risetime_sample, energy_sample = [], []

    for i in xrange(nsims):

        sample = parameters_red[:,i]
        risetime_all = np.array([np.array([a.scale for a in s.all]) for s in sample])

        #risetime_all = risetime_all.flatten()
        energy_all = np.array([np.array([a.energy for a in s.all]) for s in sample])
        #amplitude_all = amplitude_all.flatten()

        risetime, energy = [], []
        for r,a in zip(risetime_all, energy_all):
            risetime.extend(r)
            energy.extend(a)

        risetime_sample.append(risetime)
        energy_sample.append(energy)

    sp_all = []

    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    for i,(r,a) in enumerate(zip(risetime_sample, energy_sample)):
        a = np.array(a)
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)
        scatter(np.log10(r),np.log10(a), color=cm.jet(i*20))

    axis([np.min([np.min(np.log10(r)) for r in risetime_sample]),
          np.max([np.max(np.log10(r)) for r in risetime_sample]),
          np.min([np.min(np.log10(a)) for r in energy_sample]),
          np.max([np.max(np.log10(a)) for a in energy_sample])])

    xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
    ylabel("total number of counts in a spike", fontsize=20)
    title("total number of counts in a spike versus rise time")
    savefig("risetime_energy.png", format="png")
    close()

    return risetime_sample, energy_sample, sp_all

def risetime_skewness(datadir="./", nsims=5):


    parameters_red,bids = extract_sample(datadir, nsims)
    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    risetime_sample, skewness_sample = [], []

    for i in xrange(nsims):

        sample = parameters_red[:,i]
        risetime_all = np.array([np.array([a.scale for a in s.all]) for s in sample])

        #risetime_all = risetime_all.flatten()
        skewness_all = np.array([np.array([a.skew for a in s.all]) for s in sample])
        #amplitude_all = amplitude_all.flatten()

        risetime, skewness = [], []
        for r,a in zip(risetime_all, skewness_all):
            risetime.extend(r)
            skewness.extend(a)

        risetime_sample.append(risetime)
        skewness_sample.append(skewness)

    sp_all = []

    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    for i,(r,a) in enumerate(zip(risetime_sample, skewness_sample)):
        a = np.array(a)/0.0005
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)
        scatter(np.log10(r),np.log10(a), color=cm.jet(i*20))

    axis([np.min([np.min(np.log10(r)) for r in risetime_sample]),
          np.max([np.max(np.log10(r)) for r in risetime_sample]),
          np.min([np.min(np.log10(a)) for r in skewness_sample]),
          np.max([np.max(np.log10(a)) for a in skewness_sample])])

    xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
    ylabel("skewness parameter", fontsize=20)
    title("skewness versus rise time")
    savefig("risetime_skewness.png", format="png")
    close()

    return risetime_sample, skewness_sample, sp_all



def waiting_times(datadir="./", nsims=10, trigfile=None):

    parameters_red, bids = extract_sample(datadir, nsims)
    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)

    waitingtime_sample = []
    print("nsims: %i"%nsims)

    if not trigfile is None:
        data = burstmodel.conversion(trigfile)
        bid_ttrig = np.array([t for t in data[0]])
        ttrig_all = np.array([float(t) for t in data[1]])


    for i in xrange(nsims):

        sample = parameters_red[:,i]

        t0_all = np.array([np.array([a.t0 for a in s.all]) for s in sample])

        t0_all_corrected = []
        if not trigfile is None:
            for j,t in enumerate(t0_all):
                bid_ind = np.where(bid_ttrig == bids[j])[0]
                ttrig = ttrig_all[bid_ind]
                t = t + ttrig
                t0_all_corrected.append(t)

        else:
            t0_all_corrected = t0_all

        t0 = []
        for t in t0_all_corrected:
            t0.extend(t)

        t0_sort = np.sort(np.array(t0))
        print(t0_sort)

        waitingtime = t0_sort[1:] - t0_sort[:-1]
        waitingtime_sample.append(waitingtime)


    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    n_all = []
    for i,w in enumerate(waitingtime_sample):

        n,bins, patches = hist(log10(w), bins=30, range=[np.log10(0.0001), np.log10(330.0)],
                               color=cm.jet(i*20),alpha=0.6, normed=True)
        n_all.append(n)

    axis([np.log10(0.0001), np.log10(330.0), np.min([np.min(n) for n in n_all]), np.max([np.max(n) for n in n_all])])

    xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
    ylabel("total number of counts in a spike", fontsize=20)
    title("total number of counts in a spike versus rise time")
    savefig("waitingtimes.png", format="png")
    close()


    return waitingtime_sample


def risetime_duration(datadir="./", nsims=10):

    parameters_red,bids = extract_sample(datadir, nsims)
    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    risetime_sample, skewness_sample, duration_sample, bkg_sample, amp_sample = [], [], [], [], []

    for i in xrange(nsims):

        sample = parameters_red[:,i]
        risetime_all = np.array([np.array([a.scale for a in s.all]) for s in sample])

        #risetime_all = risetime_all.flatten()
        skewness_all = np.array([np.array([a.skew for a in s.all]) for s in sample])
        #amplitude_all = amplitude_all.flatten()

        amplitude_all = np.array([np.array([a.amp for a in s.all]) for s in sample])

        bkg_all = np.array([s.bkg for s in sample])


        risetime, skewness = [], []
        for r,a in zip(risetime_all, skewness_all):
            risetime.extend(r)
            skewness.extend(a)

        risetime_sample.append(risetime)
        skewness_sample.append(skewness)

    sp_all = []

    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    for i,(r,a) in enumerate(zip(risetime_sample, skewness_sample)):
        a = np.array(a)/0.0005
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)
        scatter(np.log10(r),np.log10(a), color=cm.jet(i*20))

    axis([np.min([np.min(np.log10(r)) for r in risetime_sample]),
          np.max([np.max(np.log10(r)) for r in risetime_sample]),
          np.min([np.min(np.log10(a)) for r in skewness_sample]),
          np.max([np.max(np.log10(a)) for a in skewness_sample])])

    xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
    ylabel("skewness parameter", fontsize=20)
    title("skewness versus rise time")
    savefig("risetime_skewness.png", format="png")
    close()



    return

def skewness_dist(datadir="./", nsims=10):

    parameters_red = extract_sample(datadir, nsims)
    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)

    skewness_sample = []
    for i in xrange(nsims):

        sample = parameters_red[:,i]
        skewness_all = np.array([np.array([a.skew for a in s.all]) for s in sample])

        skewness = []
        for t in skewness_all:
            skewness.extend(t)

        skewness_sample.append(skewness)

    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)

    for i,w in enumerate(skewness_sample):

        n,bins, patches = hist(log10(w), bins=30,
                               color=cm.jet(i*20),alpha=0.6, normed=True)
        #n_all.append(n)

    #axis([, np.log10(1000.0), np.min([np.min(n) for n in n_all]), np.max([np.min(n) for n in n_all])])

    xlabel(r"$\log{(\mathrm{skewness})}$ [s]", fontsize=20)
    ylabel("p(skewness)", fontsize=20)
    title("skewness parameter for a large number of spikes")
    savefig("skewness_dist.png", format="png")
    close()


    return skewness_sample


##### OLD CODE: NEED TO CHECK THIS! ##########


def read_dnest_results(filename, datadir="./", trigfile=None):

    """
    Read output from RJObject/DNest3 run and return in a format more
    friendly to post-processing.

    filename: filename with posterior sample (posterior_sample.txt)

    NOTE: parameters (amplitudes + background) are in COUNTS space, not COUNT RATE!
    """


    #options = burstmodel.conversion("%sOPTIONS.txt" %dnestdir)

    dfile = "%s%s" %(datadir, filename)
    alldata = np.loadtxt(dfile)
    print("filename: " + str(filename))
    print("shape alldata: " + str(alldata.shape))



    if not trigfile is None:
        trigdata = burstmodel.conversion("%s%s"%(datadir,trigfile))
        bids = np.array(trigdata[0])
        ttrig_all = np.array([float(t) for t in trigdata[1]])

        bid_data = filename.split("_")[0][2:]
        #print("bid_data: " + str(bid_data))
        bind = np.where(bids == bid_data)[0]
        ttrig = ttrig_all[bind]
        #print("ttrig: " + str(ttrig))

    else:
        ttrig = 0


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
    pos_all = np.array(alldata[:, 8:8+compmax]) + ttrig

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


def parameter_sample(filename, datadir="./", trigfile=None):

    ### extract parameters from file
    sample_dict = read_dnest_results(filename, datadir=datadir, trigfile=trigfile)


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
        e_all = p.compute_energy()

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
