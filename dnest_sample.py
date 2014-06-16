
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
        data = loadtxt("%s_%s_data.dat"%(fsplit[0], fsplit[1]))
        fig = figure(figsize=(24,9))
        ax = fig.add_subplot(121)
        plot(data[:,0], data[:,1], lw=2, color="black", linestyle="steps-mid")
        sample = atleast_2d(loadtxt(f))

        print(f)
        print(sample.shape)

        ind = np.random.choice(np.arange(len(sample)), replace=False, size=nsims)
        for i in ind:
            #print("shape data: " + str(len(data[:,0])))
            #print("shape sample: " + str(len(sample[i,-data.shape[0]:])))
            plot(data[:,0], sample[i,-data.shape[0]:], lw=1)
            #plot(data[:,0], np.ones(len(data[:,0]))*sample[i,0], lw=2)
        xlabel("Time since trigger [s]", fontsize=20)
        ylabel("Counts per bin", fontsize=20)
        xlim([0.0, data[-1,0]-data[0,0]])


        ax = fig.add_subplot(122)
        nbursts = sample[:, 7]

        hist(nbursts, bins=30, range=[np.min(nbursts), np.max(nbursts)], histtype='stepfilled')
        xlabel("Number of spikes per burst", fontsize=20)
        ylabel("N(samples)", fontsize=20)
        savefig("%s_%s_lc.png"%(fsplit[0], fsplit[1]), format="png")
        close()

    return


def extract_sample(datadir="./", nsims=50, filter_weak=False, trigfile="sgr1550_ttrig.dat"):

    files = glob.glob("%s*posterior*"%datadir)
    #print("files: " + str(files))

    all_parameters, bids, nsamples = [], [], []
    for f in files:
        fname = f.split("/")[-1]
        bid = fname.split("_")[0]
        bids.append(bid)
        parameters = parameter_sample(f, filter_weak=filter_weak, trigfile=trigfile)
        all_parameters.append(parameters)
        nsamples.append(len(parameters))

    if nsims > np.min(nsamples):
        nsims = np.min(nsamples)
        print("Number of desired simulations larger than smallest posterior sample.")
        print("Resetting nsims to %i" %nsims)

    parameters_red = np.array([np.random.choice(p, replace=False, size=nsims) for p in all_parameters])
    #print("shape of reduced parameter array: " + str(parameters_red.shape))

    return parameters_red, bids

def risetime_amplitude(sample=None, datadir="./", nsims=5, dt=0.0005, makeplot=True, froot="test"):

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

    popt_mean = np.mean(np.array(popt_all), axis=0)
    popt_std = np.std(np.array(popt_all), axis=0)



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


        ax.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_mean[0],popt_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax.transAxes,
                fontsize=16)


        xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
        ylabel("log(spike amplitude)", fontsize=20)
        title("spike amplitude versus rise time")
        savefig("%s_risetime_amplitude.png"%froot, format="png")
        close()

        return risetime_sample, amplitude_sample, sp_all, popt_all


def risetime_energy(sample=None, datadir="./", nsims=5, dt=0.0005, makeplot=True, froot="test"):

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

    popt_mean = np.mean(np.array(popt_all), axis=0)
    popt_std = np.std(np.array(popt_all), axis=0)


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


        ax.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_mean[0],popt_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax.transAxes,
                fontsize=16)



        plot(np.log10(rx), np.log10(min_energy), lw=2, color="black", ls="dashed")

        xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
        ylabel("total number of counts in a spike", fontsize=20)
        title("total number of counts in a spike versus rise time")
        savefig("%s_risetime_energy.png"%froot, format="png")
        close()

    return risetime_sample, energy_sample, sp_all, popt_all

def risetime_skewness(sample=None, datadir="./", nsims=5, makeplot=True, froot="test"):

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

    popt_mean = np.mean(popt_all, axis=0)
    popt_std = np.std(popt_all, axis=0)

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

        ax.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_mean[0],popt_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax.transAxes,
                fontsize=16)


        xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
        ylabel("skewness parameter", fontsize=20)
        title("skewness versus rise time")
        savefig("%s_risetime_skewness.png"%froot, format="png")
        close()

    return risetime_sample, skewness_sample, sp_all, popt_all



def waiting_times(sample=None, bids=None, datadir="./", nsims=10, trigfile=None, makeplot=True, froot="test"):

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
        savefig("%s_waitingtimes.png"%froot, format="png")
        close()


    return waitingtime_sample

def waitingtime_energy(sample=None, bids=None, datadir="./", nsims=10, trigfile=None, makeplot=True,
                       dt=0.0005, froot="test"):

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

    popt_plus_all = np.array(popt_plus_all)
    popt_minus_all = np.array(popt_minus_all)

    popt_plus_mean = np.mean(popt_plus_all, axis=0)
    popt_plus_std = np.std(popt_plus_all, axis=0)

    popt_minus_mean = np.mean(popt_minus_all, axis=0)
    popt_minus_std = np.std(popt_minus_all, axis=0)

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

            ax1.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_plus_mean[0],popt_plus_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax1.transAxes,
                fontsize=16)


            xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)
            ylabel("total number of counts", fontsize=20)

            ax2 = fig.add_subplot(122)

            ax2.scatter(np.log10(r[:-1]), np.log10(a[1:]),color=cm.jet(i*20), label=r"$dt_-$")

            ax2.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_minus_mean[0],popt_minus_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax2.transAxes,
                fontsize=16)

            #axis([np.min([np.min(np.log10(r)) for r in waitingtime_sample]),
            #      np.max([np.max(np.log10(r)) for r in waitingtime_sample]),
            #      np.min([np.min(np.log10(a)) for a in energy_sample]),
            #      np.max([np.max(np.log10(a)) for a in energy_sample])])

            #legend()
            xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)
        fig.suptitle("Waiting time versus energy", fontsize=26)
        #    title("energy versus waiting time")
        savefig("%s_waitingtime_energy.png"%froot, format="png")
        close()


    return waitingtime_sample, energy_sample

def waitingtime_amplitude(sample=None, bids=None, datadir="./", nsims=10, trigfile=None, makeplot=True,
                          dt=0.0005, froot="test"):

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


    popt_plus_all = np.array(popt_plus_all)
    popt_minus_all = np.array(popt_minus_all)

    popt_plus_mean = np.mean(popt_plus_all, axis=0)
    popt_plus_std = np.std(popt_plus_all, axis=0)

    popt_minus_mean = np.mean(popt_minus_all, axis=0)
    popt_minus_std = np.std(popt_minus_all, axis=0)

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

            ax1.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_plus_mean[0],popt_plus_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax1.transAxes,
                fontsize=16)


            xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)
            ylabel("total number of counts", fontsize=20)

            ax2 = fig.add_subplot(122)

            ax2.scatter(np.log10(r[:-1]), np.log10(a[1:]),color=cm.jet(i*20), label=r"$dt_-$")
            ax2.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_minus_mean[0],popt_minus_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax2.transAxes,
                fontsize=16)


            #axis([np.min([np.min(np.log10(r)) for r in waitingtime_sample]),
            #      np.max([np.max(np.log10(r)) for r in waitingtime_sample]),
            #      np.min([np.min(np.log10(a)) for a in energy_sample]),
            #      np.max([np.max(np.log10(a)) for a in energy_sample])])

            #legend()
            xlabel(r"$\log{(\mathrm{waiting\; time})}$ [s]", fontsize=20)

        fig.suptitle("Waiting time versus amplitude", fontsize=26)
        #    title("energy versus waiting time")
        savefig("%s_waitingtime_amplitude.png"%froot, format="png")
        close()


    return waitingtime_sample, amplitude_sample


def risetime_duration(sample=None, datadir="./", nsims=10, makeplot=True, froot="test", dt=0.0005):


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
        risetime_all = np.array([np.array([a.scale for a in s.all if a.duration > 0.0]) for s in sample])

        print('risetime_all: ' + str(risetime_all[0]))
        #risetime_all = risetime_all.flatten()
        duration_all = np.array([np.array([a.duration for a in s.all if a.duration > 0.0]) for s in sample])
        #amplitude_all = amplitude_all.flatten()

        print("duration_all: " + str(duration_all[0]))

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
        #print("len(a): " + str(a))
        #print("len(r): " + str(r))
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)

        popt, pcov = scipy.optimize.curve_fit(straight, np.log10(r), np.log10(a), p0=None, sigma=None)
        popt_all.append(popt)
        pcov_all.append(pcov)

    popt_mean = np.mean(popt_all, axis=0)
    popt_std = np.std(popt_all, axis=0)

    if makeplot:
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        for i,(r,a) in enumerate(zip(risetime_sample, duration_sample)):
            a = np.array(a)
            #sp = scipy.stats.spearmanr(r,a)
            #sp_all.append(sp)
            scatter(np.log10(r),np.log10(a), color=cm.jet(i*20))

        ### minimum values for duration and rise time are set by prior:
        ### rise time cannot be shorter than dt/10.0, thus neither can the duration
        axis([np.log10(dt/10.0),
              np.max([np.max(np.log10(r)) for r in risetime_sample]),
              np.log10(dt/10.0),
              np.max([np.max(np.log10(a)) for a in duration_sample])])

        ax.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_mean[0],popt_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax.transAxes,
                fontsize=16)


        xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
        ylabel("spike duration", fontsize=20)
        title("rise time versus total duration")
        savefig("%s_risetime_duration.png"%froot, format="png")
        close()

    return risetime_sample, duration_sample, sp_all, popt_all



def energy_duration(sample=None, datadir="./", nsims=10, makeplot=True, dt=0.0005, p0=[1.5, 1.0], froot="test"):


    if sample is None:
        parameters_red,bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    energy_sample, duration_sample = [], []

    for i in xrange(nsims):

        sample = parameters_red[:,i]
        energy_all = np.array([np.array([a.scale for a in s.all if a.duration > 0.0]) for s in sample])

        #energy_all = energy_all.flatten()
        duration_all = np.array([np.array([a.duration for a in s.all if a.duration > 0.0]) for s in sample])
        #amplitude_all = amplitude_all.flatten()


        energy, duration = [], []
        for e,d in zip(energy_all, duration_all):
            energy.extend(e)
            duration.extend(d)

        energy_sample.append(energy)
        duration_sample.append(duration)

    sp_all = []
    popt_all, pcov_all = [], []

    for i,(r,a) in enumerate(zip(duration_sample, energy_sample)):
        a = np.array(a)/dt
        sp = scipy.stats.spearmanr(r,a)
        sp_all.append(sp)

        popt, pcov = scipy.optimize.curve_fit(straight, np.log10(r), np.log10(a), p0=[0.5,1.0], sigma=None)
        popt_all.append(popt)
        pcov_all.append(pcov)


    popt_mean = np.mean(popt_all, axis=0)
    popt_std = np.std(popt_all, axis=0)

    if makeplot:
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        for i,(r,a) in enumerate(zip(duration_sample, energy_sample)):
            a = np.array(a)/dt
            #sp = scipy.stats.spearmanr(r,a)
            #sp_all.append(sp)
            scatter(np.log10(r),np.log10(a), color=cm.jet(i*20))

        axis([np.log10(dt/10.0),
              np.max([np.max(np.log10(r)) for r in duration_sample]),
              np.min([np.min(np.log10(np.array(a)/dt)) for a in energy_sample]),
              np.max([np.max(np.log10(np.array(a)/dt)) for a in energy_sample])])

        ax.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_mean[0],popt_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax.transAxes,
                fontsize=16)


        xlabel(r"$\log_{10}{(\mathrm{duration})}$ [s]", fontsize=20)
        ylabel(r"$\log_{10}{(\mathrm{spike\; energy})}$", fontsize=20)
        title("duration versus total energy")
        savefig("%s_duration_energy.png"%froot, format="png")
        close()

    return duration_sample, energy_sample, sp_all, popt_all


def skewness_dist(sample=None, datadir="./", nsims=10, makeplot=True, froot="test"):

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
        savefig("%s_skewness_dist.png"%froot, format="png")
        close()


    return skewness_sample


def nspike_dist(sample=None, datadir="./", nsims=10, makeplot=True, froot="sgr1550"):

    if sample is None:
        parameters_red,bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    nspikes_sample = []
    for i in xrange(nsims):

        sample = parameters_red[:,i]
        nspikes_all = np.array([len(s.all) for s in sample])
        nspikes_sample.append(nspikes_all)

    if makeplot:
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        n_all = []
        for i,w in enumerate(nspikes_sample):

            n,bins, patches = hist(w, bins=50, range=[1, 50],
                                   color=cm.jet(i*20),alpha=0.6, normed=True)
            n_all.append(n)

        axis([1, 50, np.min([np.min(n) for n in n_all]), np.max([np.max(n) for n in n_all])])

        xlabel(r"number of components", fontsize=24)
        ylabel("p(number of components)", fontsize=24)
        title("distribution of the number of components per burst")
        savefig("%s_nspikes.png"%froot, format="png")
        close()

    return nspikes_sample


def nspikes_energy(sample=None, datadir="./", nsims=10, makeplot=True, froot="sgr1550", dt=0.0005):


    if sample is None:
        parameters_red,bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    nspikes_sample, energy_sample = [], []
    for i in xrange(nsims):

        sample = parameters_red[:,i]
        nspikes_all = np.array([len(s.all) for s in sample])
        nspikes_sample.append(nspikes_all)

        energy_all = np.array([np.sum(np.array([a.energy for a in s.all if a.energy > 0.0])) for s in sample])

        #print('energy_all: ' + str(energy_all))

        nspikes_sample.append(nspikes_all)
        energy_sample.append(energy_all)

    #print("nspikes_sample: " + str(energy_sample))

    if makeplot:
        fig = figure(figsize=(12,9))
        ax = fig.add_subplot(111)
        for i,(r,a) in enumerate(zip(nspikes_sample, energy_sample)):
            a = np.array(a)/dt
            #sp = scipy.stats.spearmanr(r,a)
            #sp_all.append(sp)
            scatter(r,np.log10(a), color=cm.jet(i*20))

        min_energy = np.array([np.min(np.log10(np.array([a_temp for a_temp in a if a_temp>0])/dt)) for a in energy_sample])
        max_energy = np.array([np.max(np.log10(np.array([a_temp for a_temp in a if a_temp>0])/dt)) for a in energy_sample])

        #print("min_energy: " + str(min_energy))
        #print("max_energy: " + str(max_energy))

        axis([0,20, np.min(min_energy), np.max(max_energy)])

        #ax.text(0.8,0.1, r"power law index $\gamma = %.2f \pm %.2f$"%(popt_mean[0],popt_std[0]),
        #        verticalalignment='center', horizontalalignment='center', color='black', transform=ax.transAxes,
        #        fontsize=16)


        xlabel(r"$n_{\mathrm{spikes}}$", fontsize=24)
        ylabel(r"$\log_{10}{(\mathrm{burst\; energy})}$", fontsize=24)
        title("duration versus total energy")
        savefig("%s_nspikes_energy.png"%froot, format="png")
        close()


    return nspikes_sample, energy_sample






def all_correlations(sample=None, bids=None, datadir="./", trigfile="sgr1550_ttrig.dat", nsims=10, dt=0.0005,
                     makeplot=True, froot="sgr1550"):

    if sample is None and bids is None:
        sample, bids = extract_sample(datadir, nsims)

    risetime, amplitude, sp_all, popt_all = risetime_amplitude(sample, nsims=nsims, makeplot=makeplot, froot=froot)
    risetime, energy, sp_all, popt_all = risetime_energy(sample, nsims=nsims, makeplot=makeplot, froot=froot)
    risetime, skewness, sp_all, popt_all = risetime_skewness(sample, nsims=nsims, makeplot=makeplot, froot=froot)
    risetime, duration, sp_all, popt_all = risetime_duration(sample, nsims=nsims, makeplot=makeplot,
                                                             froot=froot, dt=dt)

    waitingtimes = waiting_times(sample, bids,nsims=nsims, trigfile= trigfile, makeplot=makeplot, froot=froot)
    waitingtime, energy= waitingtime_energy(sample, bids, nsims=nsims,trigfile=trigfile, makeplot=makeplot, froot=froot)
    waitingtime, amplitude = waitingtime_amplitude(sample, bids, nsims=nsims,trigfile=trigfile,
                                                   makeplot=makeplot, froot=froot)

    duration, energy, sp_all, popt_all = energy_duration(sample, nsims=nsims, makeplot=makeplot, froot=froot,
                                                         dt=dt)

    nspikes = nspike_dist(sample, nsims=nsims, datadir=datadir, makeplot=makeplot, froot=froot)
    nspikes, energies = nspikes_energy(sample, datadir=datadir, nsims=nsims, makeplot=makeplot, froot=froot, dt=dt)

    return



def parameter_evolution(sample=None, datadir="./", nsims=50, nspikes=10, dt=0.0005, froot="sgr1550"):

    if sample is None:
        parameters_red,bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    sorted_data_all = []

    for pars in parameters_red:

        risetime = np.array([[a.scale for a in p.all] for p in pars])
        duration = np.array([[a.duration for a in p.all] for p in pars])
        t0 = np.array([[a.t0 for a in p.all] for p in pars])
        amplitude = np.array([[a.amp for a in p.all] for p in pars])
        energy = np.array([[a.energy for a in p.all] for p in pars])
        waiting_times = np.array([np.array(t[1:])-np.array(t[:-1]) for t in t0])


        #risetime_all.append(risetime)
        #duration_all.append(duration)
        #amplitude_all.append(amplitude)
        #waitingtime_all.append(waiting_times)

        sorted_data = [sorted(zip(t,r,d,a,e,w)) for t,r,d,a,e,w
                       in zip(t0, risetime, duration, amplitude, energy, waiting_times)]

        sorted_data_all.append(sorted_data)

    sorted_data_all = np.array(sorted_data_all)


    ### columns and rows for plot
    ncolumns = 3
    nrows = int(nspikes/ncolumns)


    ### if nspikes is not divisible by 3, I need another row
    if float(nspikes/ncolumns) - nrows > 0:
        nrows += 1


    fig_rise = figure(figsize=(ncolumns*6.0,nrows*6.0))
    ax_rise_top = fig_rise.add_subplot(111)
    ax_rise_top.spines['top'].set_color('none')
    ax_rise_top.spines['bottom'].set_color('none')
    ax_rise_top.spines['left'].set_color('none')
    ax_rise_top.spines['right'].set_color('none')
    ax_rise_top.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')


    fig_amp = figure(figsize=(ncolumns*6.0,nrows*6.0))
    ax_amp_top = fig_amp.add_subplot(111)

    ax_amp_top.spines['top'].set_color('none')
    ax_amp_top.spines['bottom'].set_color('none')
    ax_amp_top.spines['left'].set_color('none')
    ax_amp_top.spines['right'].set_color('none')
    ax_amp_top.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')



    fig_dt = figure(figsize=(ncolumns*6.0,nrows*6.0))
    ax_dt_top = fig_dt.add_subplot(111)

    ax_dt_top.spines['top'].set_color('none')
    ax_dt_top.spines['bottom'].set_color('none')
    ax_dt_top.spines['left'].set_color('none')
    ax_dt_top.spines['right'].set_color('none')
    ax_dt_top.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')



    fig_duration = figure(figsize=(ncolumns*6.0,nrows*6.0))
    ax_duration_top = fig_duration.add_subplot(111)

    ax_duration_top.spines['top'].set_color('none')
    ax_duration_top.spines['bottom'].set_color('none')
    ax_duration_top.spines['left'].set_color('none')
    ax_duration_top.spines['right'].set_color('none')
    ax_duration_top.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')



    fig_energy = figure(figsize=(ncolumns*6.0,nrows*6.0))
    ax_energy_top = fig_energy.add_subplot(111)

    ax_energy_top.spines['top'].set_color('none')
    ax_energy_top.spines['bottom'].set_color('none')
    ax_energy_top.spines['left'].set_color('none')
    ax_energy_top.spines['right'].set_color('none')
    ax_energy_top.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')


    #fig_rise, ax_rise_top = subplots(111, figsize=(ncolumns*6.0,nrows*6.0))
    #fig_amp, ax_amp_top = subplots(111, figsize=(ncolumns*6.0,nrows*6.0))
    #fig_dt, ax_dt_top = subplots(111,figsize=(ncolumns*6.0,nrows*6.0))
    #fig_duration, ax_duration_top = subplots(111,figsize=(ncolumns*6.0,nrows*6.0))
    #fig_energy, ax_energy_top = subplots(111,figsize=(ncolumns*6.0,nrows*6.0))
    #fig_skew = subplots(111, figsize=(ncolumns*6.0,nrows*6.0))


    for n in xrange(nspikes):
        ax_rise = fig_rise.add_subplot(nrows, ncolumns, n+1)
        ax_amp = fig_amp.add_subplot(nrows, ncolumns, n+1)
        ax_dt = fig_dt.add_subplot(nrows, ncolumns, n+1)
        ax_duration= fig_duration.add_subplot(nrows, ncolumns, n+1)
        ax_energy = fig_energy.add_subplot(nrows, ncolumns, n+1)
        #ax_skew = fig_skew.add_subplot(nrows, ncolumns, n)

        for i in xrange(nsims):
            samp = sorted_data_all[:,i]

            rise = np.array([s[n][1] for s in samp if len(s)>n])
            duration = np.array([s[n][2] for s in samp if len(s)>n])
            amp = np.array([s[n][3] for s in samp if len(s)>n])/dt
            energy = np.array([s[n][4] for s in samp if len(s)>n])/dt
            waitingtime = np.array([s[n][5] for s in samp if len(s)>n])

            ax_rise.hist(np.log10(rise), range=[np.log10(0.00005), np.log10(2.5)], bins=40,
                         normed=True, alpha=0.6, color=cm.jet(i*20.0))

            ax_amp.hist(np.log10(amp), range=[np.log10(1.0/dt), np.log10(3.5e5)], bins=40,
                         normed=True, alpha=0.6, color=cm.jet(i*20.0))

            ax_energy.hist(np.log10(energy), range=[np.log10(1.0/dt), np.log10(3.5e6)], bins=40,
                         normed=True, alpha=0.6, color=cm.jet(i*20.0))

            ax_duration.hist(np.log10(duration), range=[np.log10(0.00005), np.log10(2.5)], bins=40,
                         normed=True, alpha=0.6, color=cm.jet(i*20.0))

            ax_dt.hist(np.log10(waitingtime), range=[np.log10(0.0005), np.log10(330.0)], bins=40,
                         normed=True, alpha=0.6, color=cm.jet(i*20.0))

        ax_rise.set_title("Spike %i"%(n+1))
        ax_amp.set_title("Spike %i"%(n+1))
        ax_energy.set_title("Spike %i"%(n+1))
        ax_duration.set_title("Spike %i"%(n+1))
        ax_dt.set_title("Spike %i"%(n+1))


    ax_dt_top.set_xlabel(r"$\log_{10}{(\mathrm{waiting \; time})}$", fontsize=20)
    ax_dt_top.set_ylabel(r"$p(\log_{10}{(\mathrm{waiting \; time})})$", fontsize=20)
    savefig("%s_dt_evolution.png"%froot, format="png")
    close()

    ax_duration_top.set_xlabel(r"$\log_{10}{(\mathrm{duration})}$", fontsize=20)
    ax_duration_top.set_ylabel(r"$p(\log_{10}{(\mathrm{duration})})$", fontsize=20)
    savefig("%s_duration_evolution.png"%froot, format="png")
    close()

    ax_energy_top.set_xlabel(r"$\log_{10}{(\mathrm{energy})}$", fontsize=20)
    ax_energy_top.set_ylabel(r"$p(\log_{10}{(\mathrm{energy})})$", fontsize=20)
    savefig("%s_energy_evolution.png"%froot, format="png")
    close()

    ax_amp_top.set_xlabel(r"$\log_{10}{(\mathrm{amplitude})}$", fontsize=20)
    ax_amp_top.set_ylabel(r"$p(\log_{10}{(\mathrm{amplitude})})$", fontsize=20)
    savefig("%s_amplitude_evolution.png"%froot, format="png")
    close()

    ax_rise_top.set_xlabel(r"$\log_{10}{(\mathrm{rise \; time})}$", fontsize=20)
    ax_rise_top.set_ylabel(r"$p(\log_{10}{(\mathrm{rise \; time})})$", fontsize=20)
    savefig("%s_risetime_evolution.png"%froot, format='png')
    close()


    return


def differential_distributions(sample=None, datadir="./", nsims=10, makeplot=True, dt=0.0005, froot="test"):

    if sample is None:
        parameters_red,bids = extract_sample(datadir, nsims)
    else:
        parameters_red = sample


    if nsims > parameters_red.shape[1]:
        print("Number of available parameter sets smaller than nsims.")
        nsims = parameters_red.shape[1]
        print("Resetting nsims to %i."%nsims)


    energy_sample, duration_sample, amp_sample = [], [],  []

    for i in xrange(nsims):

        sample = parameters_red[:,i]
        energy_all = np.array([np.array([a.scale for a in s.all if a.duration > 0.0]) for s in sample])

        #energy_all = energy_all.flatten()
        duration_all = np.array([np.array([a.duration for a in s.all if a.duration > 0.0]) for s in sample])
        #amplitude_all = amplitude_all.flatten()

        amp_all = np.array([np.array([a.amp for a in s.all if a.duration > 0.0]) for s in sample])

        duration, energy, amp = [], [], []
        for d,e,a in zip(duration_all, energy_all, amp_all):
            duration.extend(d)
            energy.extend(e)
            amp.extend(a)


        energy_sample.append(energy)
        duration_sample.append(duration)
        amp_sample.append(amp)


    if makeplot:
        fig = figure(figsize=(24,8))

        ### first subplot: differential duration distribution
        ax = fig.add_subplot(131)
        n_all = []
        for i,d in enumerate(duration_sample):

            n,bins, patches = ax.hist(log10(d), bins=40, range=[np.log10(0.0005/10.0), np.log10(2.0)],
                                   color=cm.jet(i*20),alpha=0.6, normed=False)
            n_all.append(n)

        axis([np.log10(0.0005/10.0), np.log10(2.0), np.min([np.min(n) for n in n_all]), np.max([np.max(n) for n in n_all])])
        ax.set_xlabel(r"$\log_{10}{(\mathrm{Duration})}$", fontsize=24)
        ax.set_ylabel("p($\log_{10}{(\mathrm{Duration})}$)", fontsize=24)
        ax.set_title("Differential Duration Distribution", fontsize=24)

        ax1 = fig.add_subplot(132)
        n_all = []
        min_a, max_a = [], []
        for i,a in enumerate(amp_sample):
            a = np.array(a)/dt
            min_a.append(np.min(np.log10(a)))
            max_a.append(np.max(np.log10(a)))
            n,bins, patches = ax1.hist(log10(a), bins=40, range=[3.0, 6.0],
                                   color=cm.jet(i*20),alpha=0.6, normed=False)
            n_all.append(n)

        axis([np.min(min_a), np.max(max_a), np.min([np.min(n) for n in n_all]), np.max([np.max(n) for n in n_all])])
        ax1.set_xlabel(r"$\log_{10}{(\mathrm{Amplitude})}$", fontsize=24)
        ax1.set_ylabel("p($\log_{10}{(\mathrm{Amplitude})}$)", fontsize=24)
        ax1.set_title("Differential Amplitude Distribution", fontsize=24)

        ax2 = fig.add_subplot(133)
        n_all = []
        min_e, max_e = [], []
        for i,e in enumerate(energy_sample):
            e = np.array(e)/dt

            min_e.append(np.min(np.log10(e)))
            max_e.append(np.max(np.log10(e)))

            n,bins, patches = ax2.hist(e, bins=40, range=[-1.0, 4.0],
                                   color=cm.jet(i*20),alpha=0.6, normed=False)
            n_all.append(n)

        axis([np.min(min_e), np.max(max_e), np.min([np.min(n) for n in n_all]), np.max([np.max(n) for n in n_all])])
        ax2.set_xlabel(r"$\log_{10}{(\mathrm{Energy})}$", fontsize=24)
        ax2.set_ylabel("p($\log_{10}{(\mathrm{Energy})}$)", fontsize=24)
        ax2.set_title("Differential Energy Distribution", fontsize=24)


        savefig("%s_diff_dist.png"%froot, format="png")
        close()


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
    #print(popt_all1)
    popt_mean1 = np.mean(popt_all1, axis=0)
    popt_std1 = np.std(popt_all1, axis=0)

    popt_all2 = np.array(popt_all2)
    #print(popt_all2)
    popt_mean2 = np.mean(popt_all2, axis=0)
    popt_std2 = np.std(popt_all2, axis=0)


    emodel_energy1 = straight(np.log10(np.sort(risetime1[0])), *popt_mean1)
    emodel_energy2 = straight(np.log10(np.sort(risetime2[0])), *popt_mean2)


    e1 = np.log10(np.array(energy1[0])/dt)
    e2 = np.log10(np.array(energy2[0])/dt)


    ax1.scatter(np.log10(risetime1[0]), e1, color="blue", marker="o", edgecolor="blue", label=label1)
    ax1.scatter(np.log10(risetime2[0]), e2, color="red", marker="o", edgecolor="red", label=label2)

    ax1.plot(np.sort(np.log10(risetime1[0])), emodel_energy1, lw=4, color="navy", ls="dashed")
    ax1.plot(np.sort(np.log10(risetime2[0])), emodel_energy2, lw=4, color="darkred", ls="dashed")

    ### compute lower limit for rise times
    rx = np.logspace(np.min(np.log10(risetime1[0])), np.max(np.log10(risetime1[0])), num=100)
    min_energy = (1.0/dt)*rx
    ax1.plot(np.log10(rx), np.log10(min_energy), lw=2, color="black", ls="dashed")

    ax1.set_xlim([np.min(np.log10(risetime1[0])), np.max(np.log10(risetime1[0]))])
    ax1.set_ylim([np.min(e1), np.max(e1)])

    ax1.text(0.6,0.08, r"power law index $\gamma_1 = %.2f \pm %.2f$"%(popt_mean1[0],popt_std1[0]),
            verticalalignment='center', horizontalalignment='center', color='blue',transform=ax1.transAxes,
            fontsize=16)

    ax1.text(0.6,0.05, r"power law index $\gamma_2 = %.2f \pm %.2f$"%(popt_mean2[0], popt_std2[0]),
            verticalalignment='center', horizontalalignment='center', color='red',transform=ax1.transAxes,
            fontsize=16)

    legend(prop={"size":16})

    xlabel(r"$\log{(\mathrm{rise\; time})}$ [s]", fontsize=20)
    ylabel("total number of counts in a spike", fontsize=20)
    title("total number of counts in a spike versus rise time")


    duration1, energy1, sp_all1, popt_all1 = energy_duration(p1, nsims=len(p1), makeplot=False)
    duration2, energy2, sp_all2, popt_all2 = energy_duration(p2, nsims=len(p2), makeplot=False)

    #print("shape duration1: " + str(np.shape(duration1[0])))
    #print("shape duration2: " + str(np.shape(duration2[0])))
    #print("shape energy1: " + str(np.shape(energy1[0])))
    #print("shape energy2: " + str(np.shape(energy2[0])))

    popt_all1 = np.array(popt_all1)
    #print(np.shape(popt_all1))
    popt_mean1 = np.mean(popt_all1, axis=0)
    popt_std1 = np.std(popt_all1, axis=0)
    #print(np.shape(popt_mean1))

    popt_all2 = np.array(popt_all2)
    #print(np.shape(popt_all2))
    popt_mean2 = np.mean(popt_all2, axis=0)
    popt_std2 = np.std(popt_all2, axis=0)
    #print(np.shape(popt_mean2))

    print("popt_all1: " + str(popt_all1))
    print("popt_mean1: " + str(popt_mean1))
    print("popt_mean2: " + str(popt_mean2))

    emodel_duration1 = straight(np.log10(np.sort(duration1[0])), *popt_mean1)
    emodel_duration2 = straight(np.log10(np.sort(duration2[0])), *popt_mean2)

    print(np.min(emodel_duration2))
    print(np.max(emodel_duration2))

    ax2 = fig.add_subplot(132)

    ax2.scatter(np.log10(np.array(duration1[0])), np.log10(np.array(energy1[0])/dt), color="blue", marker="o", edgecolor="blue", label=label1)
    ax2.scatter(np.log10(np.array(duration2[0])), np.log10(np.array(energy2[0])/dt), color="red", marker="o", edgecolor="red", label=label2)

    ax2.plot(np.sort(np.log10(duration1[0])), emodel_duration1, lw=4, color="navy", ls="dashed")
    ax2.plot(np.sort(np.log10(duration2[0])), emodel_duration2, lw=4, color="darkred", ls="dashed")

    #ax2.set_xlim([np.min(np.log10(np.array(duration1[0]))), np.max(np.log10(np.array(duration1[0])))])
    #ax2.set_ylim([np.min(np.log10(np.array(energy1[0])/dt)), np.max(np.log10(np.array(energy1[0])/dt))])
    ax2.legend(prop={"size":16})

    ax2.text(0.6, 0.08, r"power law index $\gamma_1 = %.2f \pm %.2f$"%(popt_mean1[0],popt_std1[0]),
            verticalalignment='center', horizontalalignment='center', color='blue', transform=ax2.transAxes,
            fontsize=16)

    ax2.text(0.6, 0.05, r"power law index $\gamma_2 = %.2f \pm %.2f$"%(popt_mean2[0], popt_std2[0]),
            verticalalignment='center', horizontalalignment='center', color='red', transform=ax2.transAxes,
            fontsize=16)


    xlabel(r"$\log_{10}{(\mathrm{duration})}$ [s]", fontsize=20)
    ylabel(r"$\log_{10}{(\mathrm{energy})}$ [s]", fontsize=20)
    title("duration versus energy")


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


def read_dnest_results(filename, datadir="./", filter_smallest=False, trigfile="sgr1550_ttrig.dat"):

    """
    Read output from RJObject/DNest3 run and return in a format more
    friendly to post-processing.

    filename: filename with posterior sample (posterior_sample.txt)

    NOTE: parameters (amplitudes + background) are in COUNTS space, not COUNT RATE!
    """


    #options = burstmodel.conversion("%sOPTIONS.txt" %dnestdir)

    alldata = np.loadtxt(filename)
    #print("filename: " + str(filename))
    #print("shape alldata: " + str(alldata.shape))



    if not trigfile is None:
        trigdata = burstmodel.conversion("%s%s"%(datadir,trigfile))
        bids = np.array(trigdata[0])
        #print("bids: " + str(bids))
        ttrig_all = np.array([float(t) for t in trigdata[1]])

        #print("ttrig_all: " + str(ttrig_all))
        fsplit = filename.split("/")[-1]
        bid_data = fsplit.split("_")[0]
        #print("bid_data: " + str(bid_data))
        bind = np.where(bids == bid_data)[0][0]
        #print("bind: " + str(bind))
        ttrig = ttrig_all[bind]
        #print("ttrig: " + str(ttrig))

    else:
        ttrig = 0

    #print("ttrig: " + str(ttrig))
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
    #print("compmax: " + str(compmax))

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
    #print(nbursts)

    ## peak positions for all model components, some will be zero
    pos_all = np.array(alldata[:, 8:8+compmax])


    ## amplitudes for all model components, some will be zero
    amp_all = alldata[:, 8+compmax:8+2*compmax]

    ## rise times for all model components, some will be zero
    scale_all = alldata[:, 8+2*compmax:8+3*compmax]

    ## skew parameters for all model components, some will be zero
    skew_all = alldata[:, 8+3*compmax:8+4*compmax]

    ## pull out the ones that are not zero
    paras_real = []

    for p,a,sc,sk in zip(pos_all, amp_all, scale_all, skew_all):
        paras_real.append([(pos+ttrig,scale,amp,skew) for pos,amp,scale,skew in zip(p,a,sc,sk) if pos != 0.0])


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



def parameter_sample(filename, datadir="./", filter_weak=False, trigfile="sgr1550_ttrig.dat"):

    ### extract parameters from file
    sample_dict = read_dnest_results(filename, datadir=datadir, trigfile=trigfile)

    #print("filter_weak " + str(filter_weak))

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
