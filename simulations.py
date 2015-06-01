import glob
import numpy as np
import re

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
#sns.set_context("poster")

import parameters
import word
import dnest_sample
#import run_dnest
### Simulated light curves
from pylab import *

rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)

def singlepeak(bkg=None, datadir="./", trigfile="sgr1550_ttrig.dat"):

    """
    Makes simulations of light curves with peaks from the model we consider (exponential rise/decay with skewness
    parameter).

    NOTE: if bkg = None, then an estimate of the background is read from the sample.
    THIS REQUIRES FILES OF TYPE "%s*posterior*"%datadir and a file with trigger data (e.g. trigfile="sgr1550_ttrig.dat")
    TO WORK! Also, you need to have set your directory with the posterior and trigger file correctly!


    :param bkg:
    :return:
    """


    ## length of light curve
    tseg = 0.2
    dt = 0.0005
    nsims=100

    ### time stamps of time bins for a light curve of length 0.4 seconds with bin size dt
    times = np.arange(0.0, tseg, dt)

    ### if bkg keyword is not given, extract from sample
    if bkg is None:
        ### extract estimate of Fermi/GBM background counts/bin for sample
        pars, bids =  dnest_sample.extract_sample(datadir=datadir, nsims=nsims, prior="exp", trigfile=trigfile)
        bkg_sample = []
        for i in xrange(nsims):
            sample = pars[:,i]
            bkg= np.array([s.bkg for s in sample])
            bkg_sample.extend(bkg)

        ### make histogram of background parameter estimates
        n, bins = np.histogram(np.log10(bkg_sample), bins=100, range=[-5.0, np.log10(5.0)])

        ### find index of maximum of histogram
        b_index = np.where(n == np.max(n))[0]

        ### compute actual value for the background at the maximum of the model distribution
        bmax = bins[b_index] + (bins[b_index+1] - bins[b_index])/2.0
        bkg = 10.0**bmax

    ###### Now make light curves with single peak #########

    amp_all = [1,5, 10, 20]
    t0 = 0.06
    tau_rise = 0.005
    skew = 5.0

    print("background: " + str(bkg))

    for a in amp_all:
        p = parameters.TwoExpParameters(t0, tau_rise, a, skew, bkg=bkg, log=False)
        model_counts = word.TwoExp(times).model(p)
        poisson_counts = np.array([np.random.poisson(c) for c in model_counts])

        filename = "%sonespike_a=%i_data.txt"%(datadir,a)

        np.savetxt(filename, np.transpose(np.array([times, poisson_counts])))

    return


def singlepeak_results():

    #files = glob.glob("*onespike*_data.txt")
    posterior_files = glob.glob("*onespike*posterior_sample*")
    p_sorted = [posterior_files[1], posterior_files[3], posterior_files[0], posterior_files[2]]
    #r_all, a_all, n_all, s_all = [], [], [] ,[]


    #fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5,figsize = (24, 6))
    fig = plt.figure()
    ax1 = plt.subplot2grid((1,5), (0,0), colspan=1)

    ax2 = plt.subplot2grid((1,5), (0,1), colspan=4)


    fig.subplots_adjust(top=0.95, bottom=0.15, left=0.03, right=0.97, wspace=0.2)

    ncomp_all, amp_all, pos_all = [], [], []
    for f in p_sorted:

        #fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(26,7))
        amp = re.search(r'\d+', f).group()
        t0 = 0.06
        tau_rise = 0.005
        skew = 5.0

        pars = dnest_sample.parameter_sample(f, datadir="./", filter_weak=False,
                                             trigfile=None, bkg=None, prior="exp", efile=None)

        pars = pars[0]
        ncomp = [len(p.all) for p in pars]
        rise = np.array([np.array([a.scale for a in s.all]) for s in pars])
        amp = np.array([np.array([a.amp for a in s.all]) for s in pars])
        skew = np.array([np.array([a.skew for a in s.all]) for s in pars])
        pos = np.array([np.array([a.t0 for a in s.all]) for s in pars])

        risetime, amplitude, skewness, position = [], [], [], []
        for p,r,a,s in zip(pos, rise, amp, skew):
            position.extend(p)
            risetime.extend(r)
            amplitude.extend(a)
            skewness.extend(s)

        ncomp_all.append(ncomp)
        amp_all.append(amplitude)
        pos_all.append(position)

        #sns.kdeplot(data.values, shade=True, bw=(.5, 1), cmap="Purples");


        #h, bins = np.histogram(ncomp, bins=np.arange(-0.5, 10.5, 1.0))
        #ax2.hist(risetime, bins=20, range=[0.0, 0.03], color=c, alpha=0.5)
        #ax3.hist(amplitude, bins=20, range=[0,30], color=c, alpha=0.5)

        #ax5.scatter(ncomp, pos, color=c)
        #plt.title("$%s$"%f)
        #ax1.bar(bins[:-1], h, zs=z, zdir='y', color=c, alpha=0.8)

        #ax1.set_xlabel('X')
        #ax1.set_ylabel('Y')
        #ax1.set_zlabel('Z')

        #ax2.scatter(position, amplitude, color=c)
        #ax2.hexbin(position, amplitude, cmap=cm, alpha=0.5, gridsize=20)
        #plt.show()

    col = sns.color_palette()[:4]
    cmaps = [sns.dark_palette(c, reverse=True, as_cmap=True) for c in col]
    fs = 18

    sns.boxplot(ncomp_all, names=["0.6", "3.3", "6.5", "13.0"], ax=ax1)
    ax1.tick_params(axis='both', which='major', labelsize=fs)
    ax1.set_xlabel('SNR', fontsize=22)
    ax1.set_ylabel( ylabel="Number of spikes", fontsize=22)
    ax1.set_yscale("log")

    ax2.scatter(pos_all[0], amp_all[0], color=col[0])
    sns.kdeplot(np.transpose(np.array([pos_all[0],amp_all[0]])), shade=False, bw=(.01, .1), cmap=cmaps[0], ax=ax2)
    ax2.tick_params(axis='both', which='major', labelsize=fs)
    ax2.set_xlabel('Position since burst start [s]', fontsize=22, labelpad=10)
    ax2.set_ylabel("Amplitude [Counts/(0.005 seconds)]", fontsize=22)

    ax3.scatter(pos_all[1], amp_all[1], color=col[1])
    sns.kdeplot(np.transpose(np.array([pos_all[1],amp_all[1]])), shade=False, bw=(.01, .2), cmap=cmaps[1], ax=ax3)
    ax3.tick_params(axis='both', which='major', labelsize=fs)
    ax3.set_xlabel('Position since burst start [s]', fontsize=22, labelpad=10)

    ax4.scatter(pos_all[2], amp_all[2], color=col[2])
    sns.kdeplot(np.transpose(np.array([pos_all[2],amp_all[2]])), shade=False, bw=(.01, .2), cmap=cmaps[2], ax=ax4)
    ax4.tick_params(axis='both', which='major', labelsize=fs)
    ax4.set_xlabel('Position since burst start [s]', fontsize=22, labelpad=10)

    ax5.scatter(pos_all[3], amp_all[3], color=col[3])
    sns.kdeplot(np.transpose(np.array([pos_all[3],amp_all[3]])), shade=False, bw=(.01, 0.2), cmap=cmaps[3], ax=ax5)
    ax5.tick_params(axis='both', which='major', labelsize=fs)
    ax5.set_xlabel('Position since burst start [s]', fontsize=22, labelpad=10)

    plt.savefig("f3.pdf", format="pdf")
    plt.close()

    return

def multipeak(bkg=None, datadir="./", trigfile="sgr1550_ttrig.dat"):

    ## length of light curve
    tseg = 0.2
    dt = 0.0005
    nsims = 100

    ### time stamps of time bins for a light curve of length 0.4 seconds with bin size dt
    times = np.arange(0.0, tseg, dt)

    ### if bkg keyword is not given, extract from sample
    if bkg is None:
        ### extract estimate of Fermi/GBM background counts/bin for sample
        pars, bids =  dnest_sample.extract_sample(datadir=datadir, nsims=nsims, prior="exp", trigfile=trigfile)
        bkg_sample = []
        for i in xrange(nsims):
            sample = pars[:,i]
            bkg= np.array([s.bkg for s in sample])
            bkg_sample.extend(bkg)

        ### make histogram of background parameter estimates
        n, bins = np.histogram(np.log10(bkg_sample), bins=100, range=[-5.0, np.log10(5.0)])

        ### find index of maximum of histogram
        b_index = np.where(n == np.max(n))[0]

        ### compute actual value for the background at the maximum of the model distribution
        bmax = bins[b_index] + (bins[b_index+1] - bins[b_index])/2.0
        bkg = 10.0**bmax

    ###### Now make light curves with single peak #########

    nbursts = 10
    n_all = np.random.poisson(5, size=nbursts)

    #t0 = 0.06
    tau_rise = 0.005
    skew = 5.0

    print("background: " + str(bkg))

    for i,n in enumerate(n_all):
        print("I am here: %i"%i)
        t0_all = np.random.choice(times[:150], size=n)
        amp_all = np.random.uniform(0, 50, size=n)

        pars = np.array([[t, tau_rise, a, skew] for t,a in zip(t0_all, amp_all)])
        pflat = list(pars.flatten())
        pflat.append(bkg)

        p = parameters.TwoExpCombined(pflat,n, bkg=bkg, log=False)
        model_counts = word.CombinedWords(times, [word.TwoExp for w in range(n)]).model(p)
        poisson_counts = np.array([np.random.poisson(c) for c in model_counts])

        filename = "%smultipeak_%i_data.txt"%(datadir,i)

        np.savetxt(filename, np.transpose(np.array([times, poisson_counts])))

        f = open("%smultipeak_%i_parameters.txt"%(datadir,i), "w")
        for t,a in zip(t0_all, amp_all):
            f.write(str(t) + "\t" + str(tau_rise) + "\t" + str(a) + "\t" + str(skew))

        f.close()

    return



def make_sims(datadir="./"):

    ###estimate for Fermi/GBM sample
    log_bkg = 0.18606270394577695
    bkg = 10.0**log_bkg

    ### simulations with single peak
    singlepeak(datadir=datadir, bkg=bkg)

    return

def run_sims(datadir="../data/", dnest_dir="./", key="onespike", nsims=500):

    files = glob.glob("%s*%s*_data.txt"%(datadir,key))
    print("files: " + str(files))
    for f in files:
        run_dnest.run_burst(f, dnest_dir=dnest_dir, nsims=nsims)

    return

def plot_sims(datadir="./", key="onespike"):

    fig = figure(figsize=(24,9))
    subplots_adjust(top=0.9, bottom=0.1, left=0.06, right=0.97, wspace=0.15, hspace=0.1)

    amp_all = [1,5,10,50,100]

    colours = ["navy", "orange", "magenta", "mediumseagreen", "cyan"]

    for a,c in zip(amp_all, colours):

        data = np.loadtxt("%sonespike_a=%i_data.txt"%(datadir,a))
        sample = np.loadtxt("%sonespike_a=%i_posterior_sample.txt"%(datadir,a))

        ax = fig.add_subplot(121)
        plot(data[:,0]-data[0,0], (data[:,1]/0.0005)/1.e4, lw=2, color=c, linestyle="steps-mid",
             label="peak amplitude A = %.2e"%(a/0.0005))

        print(sample.shape)
        ind = np.random.choice(np.arange(len(sample)), replace=False, size=10)
        for i in ind:
            plot(data[:,0]-data[0,0], (sample[i,-data.shape[0]:]/0.0005)/1.e4, lw=1)
        xlabel("Time since burst start [s]", fontsize=24)
        ylabel(r"Count rate [$10^{4} \, \mathrm{counts} \, \mathrm{s}^{-1}$]", fontsize=24)


        ax = fig.add_subplot(122)
        nbursts = sample[:, 7]

        hist(nbursts, bins=14, range=[1, 15], histtype='stepfilled', color=c, alpha=0.6,
             label="peak amplitude A = %.2e"%(a/0.0005))

        legend()
    xlabel("Number of spikes per burst", fontsize=24)
    ylabel("N(samples)", fontsize=24)
    savefig("%s_sims.pdf"%key, format="pdf")
    close()


    return

def main():

    ## assume I'm running this code from the dnest directory in magnetron, and that the simulation files
    ## are in folder magnetron/data
    datadir = "../data/"
    dnest_dir = "./"

    ### make simulations
    make_sims(datadir=datadir)

    ### run simulations with one spike
    run_sims(datadir=datadir, dnest_dir=dnest_dir, key="onespike", nsims=250)

    return


if __name__ == "__main__":
    main()
