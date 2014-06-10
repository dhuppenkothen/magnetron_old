
import numpy as np
import glob

import burstmodel
import parameters
import word

from pylab import *
rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)

import matplotlib.cm as cm
import scipy.stats
import scipy.optimize

def plot_posterior_lightcurves(datadir="./", nsims=10):

    files = glob.glob("%s*posterior*"%datadir)

    for f in files:
        fsplit = f.split("_")
        data = loadtxt("%s_%s_all_data.dat"%(fsplit[0], fsplit[1]))
        fig = figure(figsize=(24,9))
        ax = fig.add_subplot(121)
        plot(data[:,0], data[:,1], lw=2, color="black", linestyle="steps-mid")
        sample = atleast_2d(loadtxt(f))

        print(sample.shape)
        ind = np.random.choice(np.arange(len(sample)), replace=False, size=10)
        for i in ind:
            print("shape data: " + str(len(data[:,0])))
            print("shape sample: " + str(len(sample[i,-data.shape[0]:])))
            plot(data[:,0], sample[i,-data.shape[0]:], lw=1)
            plot(data[:,0], np.ones(len(data[:,0]))*sample[i,0], lw=2)
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


def extract_sample(datadir="./", nsims=50, filter_weak=False):

    files = glob.glob("%s*posterior*"%datadir)
    print("files: " + str(files))

    all_parameters, bids, nsamples = [], [], []
    for f in files:
        fname = f.split("/")[-1]
        bid = fname.split("_")[0]
        bids.append(bid)
        parameters = parameter_sample(f, filter_weak=filter_weak)
        all_parameters.append(parameters)
        nsamples.append(len(parameters))

    if nsims > np.min(nsamples):
        nsims = np.min(nsamples)
        print("Number of desired simulations larger than smallest posterior sample.")
        print("Resetting nsims to %i" %nsims)

    parameters_red = np.array([np.random.choice(p, replace=False, size=nsims) for p in all_parameters])
    print("shape of reduced parameter array: " + str(parameters_red.shape))

    return parameters_red, bids

def risetime_amplitude(sample=None, datadir="./", nsims=5, dt=0.0005, makeplot=True):

    if sample is None:
        parameters_red, bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample

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
    popt_all, pcov_all = [], []

    for i,(r,a) in enumerate(zip(risetime_sample, amplitude_sample)):
        a = np.array(a)/dt
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)

        popt, pcov = scipy.optimize.curve_fit(straight, np.log10(r), np.log10(a), p0=None, sigma=None)
        popt_all.append(popt)
        pcov_all.append(pcov)

    if makeplot:
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        for i,(r,a) in enumerate(zip(risetime_sample, amplitude_sample)):
            a = np.array(a)/dt
            #sp = scipy.stats.spearmanr(r,a)
            #sp_all.append(sp)
            logr = np.log10(r)
            loga = np.log10(a)
            scatter(logr,loga, color=cm.jet(i*50))

        axis([np.min([np.min(np.log10(r)) for r in risetime_sample]),
              np.max([np.max(np.log10(r)) for r in risetime_sample]),
              np.min([np.min(np.log10(np.array(a)/dt)) for a in amplitude_sample]),
              np.max([np.max(np.log10(np.array(a)/dt)) for a in amplitude_sample])])

        xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
        ylabel("log(spike amplitude)", fontsize=20)
        title("spike amplitude versus rise time")
        savefig("risetime_amplitude.png", format="png")
        close()

        return risetime_sample, amplitude_sample, sp_all, popt_all


def risetime_energy(sample=None, datadir="./", nsims=5, dt=0.0005, makeplot=True):

    if sample is None:
        parameters_red,bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample

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


    ### compute lower limit for rise times
    rx = np.logspace(np.min([np.min(np.log10(r)) for r in risetime_sample]),
                     np.max([np.max(np.log10(r)) for r in risetime_sample]),
                     num=1000)

    min_energy = (1.0/dt)*rx


    sp_all = []
    sp_all = []


    popt_all, pcov_all = [], []


    for i,(r,a) in enumerate(zip(risetime_sample, energy_sample)):
        a = np.array(a)/dt
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)

        popt, pcov = scipy.optimize.curve_fit(straight, np.log10(r), np.log10(a), p0=None, sigma=None)
        popt_all.append(popt)
        pcov_all.append(pcov)


    if makeplot:
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        for i,(r,a) in enumerate(zip(risetime_sample, energy_sample)):
            a = np.array(a)/dt
            #sp = scipy.stats.spearmanr(r,a)
            #sp_all.append(sp)
            scatter(np.log10(r),np.log10(a), color=cm.jet(i*20))

        axis([np.min([np.min(np.log10(r)) for r in risetime_sample]),
              np.max([np.max(np.log10(r)) for r in risetime_sample]),
              np.min([np.min(np.log10(np.array(a)/dt)) for a in energy_sample]),
              np.max([np.max(np.log10(np.array(a)/dt)) for a in energy_sample])])

        plot(np.log10(rx), np.log10(min_energy), lw=2, color="black", ls="dashed")

        xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
        ylabel("total number of counts in a spike", fontsize=20)
        title("total number of counts in a spike versus rise time")
        savefig("risetime_energy.png", format="png")
        close()

    return risetime_sample, energy_sample, sp_all, popt_all

def risetime_skewness(sample=None, datadir="./", nsims=5, makeplot=True):

    if sample is None:
        parameters_red,bids = extract_sample(datadir, nsims)

    else:
        parameters_red = sample


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
    popt_all, pcov_all = [], []

    for i,(r,a) in enumerate(zip(risetime_sample, skewness_sample)):
        a = np.array(a)
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)

        popt, pcov = scipy.optimize.curve_fit(straight, np.log10(r), np.log10(a), p0=None, sigma=None)
        popt_all.append(popt)
        pcov_all.append(pcov)


    if makeplot:
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        for i,(r,a) in enumerate(zip(risetime_sample, skewness_sample)):
            a = np.array(a)
            #sp = scipy.stats.spearmanr(r,a)
            #sp_all.append(sp)
            scatter(np.log10(r),np.log10(a), color=cm.jet(i*20))

        axis([np.min([np.min(np.log10(r)) for r in risetime_sample]),
              np.max([np.max(np.log10(r)) for r in risetime_sample]),
              np.min([np.min(np.log10(a)) for a in skewness_sample]),
              np.max([np.max(np.log10(a)) for a in skewness_sample])])

        xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
        ylabel("skewness parameter", fontsize=20)
        title("skewness versus rise time")
        savefig("risetime_skewness.png", format="png")
        close()

    return risetime_sample, skewness_sample, sp_all, popt_all



def waiting_times(sample=None, bids=None, datadir="./", nsims=10, trigfile=None, makeplot=True):

    if sample is None and bids is None:
        parameters_red, bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)

    waitingtime_sample = []
    #print("nsims: %i"%nsims)

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
        #print(t0_sort)

        waitingtime = t0_sort[1:] - t0_sort[:-1]
        waitingtime_sample.append(waitingtime)


    if makeplot:
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        n_all = []
        for i,w in enumerate(waitingtime_sample):

            n,bins, patches = hist(log10(w), bins=30, range=[np.log10(0.0001), np.log10(330.0)],
                                   color=cm.jet(i*20),alpha=0.6, normed=True)
            n_all.append(n)

        axis([np.log10(0.0001), np.log10(330.0), np.min([np.min(n) for n in n_all]), np.max([np.max(n) for n in n_all])])

        xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)
        ylabel("p(waiting time)", fontsize=20)
        title("waiting time distribution")
        savefig("waitingtimes.png", format="png")
        close()


    return waitingtime_sample

def waitingtime_energy(sample=None, bids=None, datadir="./", nsims=10, trigfile=None, makeplot=True, dt=0.0005):

    if sample is None and bids is None:
        parameters_red, bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)

    waitingtime_sample, energy_sample = [], []
    #print("nsims: %i"%nsims)

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


        energy_all = np.array([np.array([a.energy for a in s.all]) for s in sample])
        #amplitude_all = amplitude_all.flatten()

        energy = []
        for a in energy_all:
            energy.extend(a)

        sample_sort = sorted(zip(t0, energy))
        t0_sort = np.array(sample_sort)[:,0]
        energy_sort = np.array(sample_sort)[:,1]
        #print(t0_sort)

        waitingtime = t0_sort[1:] - t0_sort[:-1]
        print("len(waitingtime): " + str(len(waitingtime)))
        print("len(energy): " + str(len(energy)))
        waitingtime_sample.append(waitingtime)

        energy_sample.append(energy[:-1])


    sp_plus_all = []
    sp_minus_all = []
    popt_plus_all, popt_minus_all = [], []

    for i,(r,a) in enumerate(zip(waitingtime_sample, energy_sample)):
        a = np.array(a)/dt
        sp_plus = scipy.stats.spearmanr(r,a)
        sp_minus = scipy.stats.spearmanr(r[:-1],a[1:])
        sp_plus_all.append(sp_plus)
        sp_minus_all.append(sp_minus)

        popt_plus, pcov_plus = scipy.optimize.curve_fit(straight, np.log10(r), np.log10(a), p0=None, sigma=None)
        popt_minus, pcov_minus = scipy.optimize.curve_fit(straight, np.log10(r[:-1]), np.log10(a[1:]), p0=None, sigma=None)

        popt_plus_all.append(popt_plus)
        popt_minus_all.append(popt_minus)


    if makeplot:
        fig = figure(figsize=(24,9))
        for i,(r,a) in enumerate(zip(waitingtime_sample, energy_sample)):
            a = np.array(a)/dt
            #sp = scipy.stats.spearmanr(r,a)
            #sp_all.append(sp)
            ax1 = fig.add_subplot(121)
            ax1.scatter(np.log10(r), np.log10(a), color=cm.jet(i*20), label=r"$dt_+$")\

            #ax1.axis([np.min([np.min(np.log10(r)) for r in waitingtime_sample]),
            #  np.max([np.max(np.log10(r)) for r in waitingtime_sample]),
            #  np.min([np.min(np.log10(a)) for a in energy_sample]),
            #  np.max([np.max(np.log10(a)) for a in energy_sample])])

            xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)
            ylabel("total number of counts", fontsize=20)

            ax2 = fig.add_subplot(122)

            ax2.scatter(np.log10(r[:-1]), np.log10(a[1:]),color=cm.jet(i*20), label=r"$dt_-$")

            #axis([np.min([np.min(np.log10(r)) for r in waitingtime_sample]),
            #      np.max([np.max(np.log10(r)) for r in waitingtime_sample]),
            #      np.min([np.min(np.log10(a)) for a in energy_sample]),
            #      np.max([np.max(np.log10(a)) for a in energy_sample])])

            legend()
            xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)
        fig.suptitle("Waiting time versus energy", fontsize=26)
        #    title("energy versus waiting time")
        savefig("waitingtime_energy.png", format="png")
        close()


    return waitingtime_sample, energy_sample

def waitingtime_amplitude(sample=None, bids=None, datadir="./", nsims=10, trigfile=None, makeplot=True, dt=0.0005):

    if sample is None and bids is None:
        parameters_red, bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)

    waitingtime_sample, amplitude_sample = [], []
    #print("nsims: %i"%nsims)

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


        amplitude_all = np.array([np.array([a.amp for a in s.all]) for s in sample])
        #amplitude_all = amplitude_all.flatten()

        amplitude = []
        for a in amplitude_all:
            amplitude.extend(a)

        sample_sort = sorted(zip(t0, amplitude))
        t0_sort = np.array(sample_sort)[:,0]
        energy_sort = np.array(sample_sort)[:,1]
        #print(t0_sort)

        waitingtime = t0_sort[1:] - t0_sort[:-1]
        print("len(waitingtime): " + str(len(waitingtime)))
        print("len(energy): " + str(len(amplitude)))
        waitingtime_sample.append(waitingtime)

        amplitude_sample.append(amplitude[:-1])


    sp_plus_all = []
    sp_minus_all = []
    popt_plus_all, popt_minus_all = [], []

    for i,(r,a) in enumerate(zip(waitingtime_sample, amplitude_sample)):
        a = np.array(a)/dt
        sp_plus = scipy.stats.spearmanr(r,a)
        sp_minus = scipy.stats.spearmanr(r[:-1],a[1:])
        sp_plus_all.append(sp_plus)
        sp_minus_all.append(sp_minus)

        popt_plus, pcov_plus = scipy.optimize.curve_fit(straight, np.log10(r), np.log10(a), p0=None, sigma=None)
        popt_minus, pcov_minus = scipy.optimize.curve_fit(straight, np.log10(r[:-1]), np.log10(a[1:]), p0=None, sigma=None)

        popt_plus_all.append(popt_plus)
        popt_minus_all.append(popt_minus)


    if makeplot:
        fig = figure(figsize=(24,9))
        for i,(r,a) in enumerate(zip(waitingtime_sample, amplitude_sample)):
            a = np.array(a)/dt
            #sp = scipy.stats.spearmanr(r,a)
            #sp_all.append(sp)
            ax1 = fig.add_subplot(121)
            ax1.scatter(np.log10(r), np.log10(a), color=cm.jet(i*20), label=r"$dt_+$")\

            #ax1.axis([np.min([np.min(np.log10(r)) for r in waitingtime_sample]),
            #  np.max([np.max(np.log10(r)) for r in waitingtime_sample]),
            #  np.min([np.min(np.log10(a)) for a in energy_sample]),
            #  np.max([np.max(np.log10(a)) for a in energy_sample])])

            xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)
            ylabel("total number of counts", fontsize=20)

            ax2 = fig.add_subplot(122)

            ax2.scatter(np.log10(r[:-1]), np.log10(a[1:]),color=cm.jet(i*20), label=r"$dt_-$")

            #axis([np.min([np.min(np.log10(r)) for r in waitingtime_sample]),
            #      np.max([np.max(np.log10(r)) for r in waitingtime_sample]),
            #      np.min([np.min(np.log10(a)) for a in energy_sample]),
            #      np.max([np.max(np.log10(a)) for a in energy_sample])])

            legend()
            xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)

        fig.suptitle("Waiting time versus amplitude", fontsize=26)
        #    title("energy versus waiting time")
        savefig("waitingtime_amplitude.png", format="png")
        close()


    return waitingtime_sample, amplitude_sample


def risetime_duration(sample=None, datadir="./", nsims=10, makeplot=True):


    if sample is None:
        parameters_red,bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    risetime_sample, duration_sample = [], []

    for i in xrange(nsims):

        sample = parameters_red[:,i]
        risetime_all = np.array([np.array([a.scale for a in s.all]) for s in sample])

        #risetime_all = risetime_all.flatten()
        duration_all = np.array([np.array([a.duration for a in s.all]) for s in sample])
        #amplitude_all = amplitude_all.flatten()


        risetime, duration = [], []
        for r,a in zip(risetime_all, duration_all):
            risetime.extend(r)
            duration.extend(a)

        risetime_sample.append(risetime)
        duration_sample.append(duration)

    sp_all = []
    popt_all, pcov_all = [], []

    for i,(r,a) in enumerate(zip(risetime_sample, duration_sample)):
        a = np.array(a)
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)

        popt, pcov = scipy.optimize.curve_fit(straight, np.log10(r), np.log10(a), p0=None, sigma=None)
        popt_all.append(popt)
        pcov_all.append(pcov)


    if makeplot:
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        for i,(r,a) in enumerate(zip(risetime_sample, duration_sample)):
            a = np.array(a)
            #sp = scipy.stats.spearmanr(r,a)
            #sp_all.append(sp)
            scatter(np.log10(r),np.log10(a), color=cm.jet(i*20))

        axis([np.min([np.min(np.log10(r)) for r in risetime_sample]),
              np.max([np.max(np.log10(r)) for r in risetime_sample]),
              np.min([np.min(np.log10(a)) for a in duration_sample]),
              np.max([np.max(np.log10(a)) for a in duration_sample])])

        xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
        ylabel("spike duration", fontsize=20)
        title("rise time versus total duration")
        savefig("risetime_duration.png", format="png")
        close()

    return risetime_sample, duration_sample, sp_all, popt_all

def skewness_dist(sample=None, datadir="./", nsims=10, makeplot=True):

    if sample is None:
        parameters_red = extract_sample(datadir, nsims)
    else:
        parameters_red = sample

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

    if makeplot:
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



def parameter_evolution(sample=None, datadir="./", nsims=50, nspikes=10):

    if sample is None:
        parameters_red,bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    risetime_all, amplitude_all, energy_all, duration_all, waitingtime_all = [], [], [], [], []

    for pars in parameters_red:

        risetime = np.array([[a.scale for a in p.all] for p in pars])
        duration = np.array([[a.duration for a in p.all] for p in pars])
        t0 = np.array([[a.t0 for a in p.all] for p in pars])
        amplitude = np.array([[a.amp for a in p.all] for p in pars])

        waiting_times = np.array([t[1:]-t[:-1] for t in t0])

        risetime_all.append(risetime)
        duration_all.append(duration)
        amplitude_all.append(amplitude)
        waitingtime_all.append(waiting_times)

    ### I NEED TO FINISH THIS FUNCTION

    return


def compare_samples(p1, p2, bids1, bids2, froot="test", label1="p1", label2="p2", dt=0.0005):
    """
    Compare different properties for two different parameter samples p1 and p2,
    where p1 and p2, and bids1 and bids2 are the outputs of extract_sample for
    various parameter sets

    """

    fig = figure(figsize=(24,8))
    subplots_adjust(top=0.9, bottom=0.1, left=0.03, right=0.97, wspace=0.15, hspace=0.2)

    ax1 = fig.add_subplot(131)

    risetime1, energy1, sp_all1, popt_all1 = risetime_energy(p1,nsims=len(p1), makeplot=False)
    risetime2, energy2, sp_all2, popt_all2 = risetime_energy(p2,nsims=len(p2), makeplot=False)


    popt_all1 = np.array(popt_all1)
    print(popt_all1)
    popt_mean1 = np.mean(popt_all1, axis=0)
    popt_std1 = np.std(popt_all1, axis=0)

    popt_all2 = np.array(popt_all2)
    print(popt_all2)
    popt_mean2 = np.mean(popt_all2, axis=0)
    popt_std2 = np.std(popt_all2, axis=0)


    emodel1 = straight(np.log10(np.sort(risetime1[0])), *popt_mean1)
    emodel2 = straight(np.log10(np.sort(risetime2[0])), *popt_mean2)


    e1 = np.log10(np.array(energy1[0])/dt)
    e2 = np.log10(np.array(energy2[0])/dt)


    ax1.scatter(np.log10(risetime1[0]), e1, color="blue", marker="o", edgecolor="blue", label=label1)
    ax1.scatter(np.log10(risetime2[0]), e2, color="red", marker="o", edgecolor="red", label=label2)

    ax1.plot(np.sort(np.log10(risetime1[0])), emodel1, lw=4, color="navy", ls="dashed")
    ax1.plot(np.sort(np.log10(risetime2[0])), emodel2, lw=4, color="darkred", ls="dashed")

       ### compute lower limit for rise times
    rx = np.logspace(np.min(np.log10(risetime1[0])), np.max(np.log10(risetime1[0])), num=100)
    min_energy = (1.0/dt)*rx
    ax1.plot(np.log10(rx), np.log10(min_energy), lw=2, color="black", ls="dashed")

    ax1.set_xlim([np.min(np.log10(risetime1[0])), np.max(np.log10(risetime1[0]))])
    ax1.set_ylim([np.min(e1), np.max(e1)])

    ax1.text(-1.5,0.7, r"power law index $\gamma_1 = %.2f \pm %.2f$"%(popt_mean1[0],popt_std1[0]),
            verticalalignment='center', horizontalalignment='center', color='blue',
            fontsize=16)

    ax1.text(-1.5,0.5, r"power law index $\gamma_2 = %.2f \pm %.2f$"%(popt_mean2[0], popt_std2[0]),
            verticalalignment='center', horizontalalignment='center', color='red',
            fontsize=16)

    legend(prop={"size":16})

    xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
    ylabel("total number of counts in a spike", fontsize=20)
    title("total number of counts in a spike versus rise time")


    risetime1, duration1, sp_all1, popt_all1 = risetime_duration(p1, nsims=len(p1), makeplot=False)
    risetime2, duration2, sp_all2, popt_all2 = risetime_duration(p2, nsims=len(p2), makeplot=False)

    popt_all1 = np.array(popt_all1)
    print(np.shape(popt_all1))
    popt_mean1 = np.mean(popt_all1, axis=0)
    popt_std1 = np.std(popt_all1, axis=0)
    print(np.shape(popt_mean1))

    popt_all2 = np.array(popt_all2)
    print(np.shape(popt_all2))
    popt_mean2 = np.mean(popt_all2, axis=0)
    popt_std2 = np.std(popt_all2, axis=0)
    print(np.shape(popt_mean2))


    emodel1 = straight(np.log10(np.sort(risetime1[0])), *popt_mean1)
    emodel2 = straight(np.log10(np.sort(risetime2[0])), *popt_mean2)

    ax2 = fig.add_subplot(132)

    ax2.scatter(np.log10(risetime1[0]), np.log10(duration1[0]), color="blue", marker="o", edgecolor="blue", label=label1)
    ax2.scatter(np.log10(risetime2[0]), np.log10(duration2[0]), color="red", marker="o", edgecolor="red", label=label2)

    ax2.plot(np.sort(np.log10(risetime1[0])), emodel1, lw=4, color="navy", ls="dashed")
    ax2.plot(np.sort(np.log10(risetime2[0])), emodel2, lw=4, color="darkred", ls="dashed")

    ax2.set_xlim([np.min(np.log10(risetime1[0])), np.max(np.log10(risetime1[0]))])
    ax2.set_ylim([np.min(np.log10(duration1[0])), np.max(np.log10(duration1[0]))])
    ax2.legend(prop={"size":16})

    ax2.text(-1.5,-3.9, r"power law index $\gamma_1 = %.2f \pm %.2f$"%(popt_mean1[0],popt_std1[0]),
            verticalalignment='center', horizontalalignment='center', color='blue',
            fontsize=16)

    ax2.text(-1.5,-4.2, r"power law index $\gamma_2 = %.2f \pm %.2f$"%(popt_mean2[0], popt_std2[0]),
            verticalalignment='center', horizontalalignment='center', color='red',
            fontsize=16)


    xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
    ylabel("Duration of the spike [s]", fontsize=20)
    title("total number of counts in a spike versus rise time")


    waitingtimes1 = waiting_times(p1, bids=bids1,nsims=len(p1), trigfile="sgr1550_ttrig.dat", makeplot=False)
    waitingtimes2 = waiting_times(p2, bids=bids2,nsims=len(p2), trigfile="sgr1550_ttrig.dat", makeplot=False)


    ax3 = fig.add_subplot(133)

    n1, bins1, patches1 = ax3.hist(np.log10(waitingtimes1[0]), bins=50, range=[np.log10(0.0001), np.log10(10.0)],
                                   color="blue", alpha=0.7, histtype="stepfilled", normed=True, label=label1)
    n2, bins2, patches2 = ax3.hist(np.log10(waitingtimes2[0]), bins=50, range=[np.log10(0.0001), np.log10(10.0)],
                                   color="red", alpha=0.7, histtype="stepfilled", normed=True, label=label2)

    ax3.set_xlim([np.log10(0.0001), np.log10(10.0)])
    ax3.set_ylim([0.0, np.max([np.max(n1), np.max(n2)])+0.1])
    ax3.legend(prop={"size":16})

    xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)
    ylabel("p(waiting time)", fontsize=20)
    title("waiting time distribution")

    savefig("%s_comparison.png"%froot, format="png")
    close()

    return




def extract_brightest_bursts(min_countrate=100000.0):

    files = glob.glob("*data.dat")
    posterior_files = glob.glob("*posterior_sample*")

    brightest = []
    brightest_posterior = []
    for f in files:
        fsplit = f.split("_")
        if "%s_%s_posterior_sample.txt"%(fsplit[0], fsplit[1]) in posterior_files:
            times, counts = burstmodel.read_gbm_lightcurves(f)
            dt = times[1] -times[0]
            print(dt)
            countrate = np.array(counts)/dt
            maxc = np.max(countrate)
            print(maxc)
            if min_countrate <= maxc <= 280000.0:
                brightest.append(f)
                brightest_posterior.append("%s_%s_posterior_sample.txt"%(fsplit[0], fsplit[1]))
            else:
                continue
        else:
            continue

    return brightest, brightest_posterior



def straight(x,a,b):
    return a*x + b

def pl(x, a, b):
    return b*np.array(x)**a


def fit_distribution(func, x, y, p0):

    xy_sorted = sorted(zip(x,y))
    xy_sorted = np.array(xy_sorted)
    x = xy_sorted[:,0]
    y = xy_sorted[:,1]


    popt, pcov = scipy.optimize.curve_fit(func, x, y, p0=None, sigma=None, absolute_sigma=False)



    return popt


##### OLD CODE: NEED TO CHECK THIS! ##########


def read_dnest_results(filename, datadir="./", filter_smallest=False):

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


def parameter_sample(filename, datadir="./", filter_weak=False):

    ### extract parameters from file
    sample_dict = read_dnest_results(filename, datadir=datadir)

    print("filter_weak " + str(filter_weak))

    ### I need the parameters, the number of components, and the background parameter
    pars_all = sample_dict["parameters"]

    nbursts_all = sample_dict["nbursts"]
    bkg_all = sample_dict["bkg"]

    parameters_all = []
    for pars,nbursts,bkg in zip(pars_all, nbursts_all, bkg_all):

        if filter_weak:
            pars_filtered = [p for p in pars if p[2] > bkg]
        else:
            pars_filtered = pars

        #print("len pars %i"%len(pars))
        #print("len filtered pars %i"%len(pars_filtered))

        nbursts = len(pars_filtered)

        pars_flat = np.array(pars_filtered).flatten()
        pars_flat = list(pars_flat)
        pars_flat.extend([bkg])
        pars_flat = np.array(pars_flat)

        p = parameters.TwoExpCombined(pars_flat, int(nbursts), log=False, bkg=True)
        e_all = p.compute_energy()
        d_all = p.compute_duration()

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
