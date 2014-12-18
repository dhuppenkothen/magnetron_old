import numpy as np
import scipy.stats

import word
import burstmodel
import parameters
import dnest_sample

import seaborn as sns
sns.set_context("notebook", font_scale=2.5, rc={"axes.labelsize": 26})
import matplotlib.pyplot as plt
from pylab import *
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import scipy.ndimage

rc("font", size=24, family="serif", serif="Computer Sans")
rc("axes", titlesize=20, labelsize=20)
rc("text", usetex=True)


def example_word():
    """
    Makes Figure 2 in the paper

    """

    t = linspace(-10., 10., 10001)
    plot(t, word.TwoExp(t).model(parameters.TwoExpParameters(1.0, 0.5, 1., 3., log=False)), linewidth=2)
    xlabel('Time [seconds]')
    ylabel('Poisson Rate')
    ylim([0., 1.1])
    axvline(1., color='r', linestyle='--')
    #title('A Word')

    # Build an arrow.
    ar1 = FancyArrow(1., 1.01*exp(-1.), -0.5, 0., length_includes_head=True,
                    color='k', head_width=0.01, head_length=0.2, width=0.001, linewidth=1)
    ar2 = FancyArrow(1., 0.99*exp(-1.), 1.5, 0., length_includes_head=True,
                    color='k', head_width=0.01, head_length=0.2, width=0.001, linewidth=1)

    ar3 = FancyArrow(-3.,0.,0.,1.0, length_includes_head=True,
                    color='k', head_width=0.15, head_length=0.015, width=0.001, linewidth=1)
    ar4 = FancyArrow(-3.,1.,0.,-1.0, length_includes_head=True,
                  color='k', head_width=0.15, head_length=0.015, width=0.001, linewidth=1)

    ar5 = FancyArrow(0., 0.03, 1.0, 0., length_includes_head=True,
                    color='k', head_width=0.01, head_length=0.2, width=0.001, linewidth=1)
    ar6 = FancyArrow(1., 0.03, -1.0, 0., length_includes_head=True,
                    color='k', head_width=0.01, head_length=0.2, width=0.001, linewidth=1)


    ax = gca()
    # Add the arrow to the axes.
    ax.add_artist(ar1)
    ax.add_artist(ar2)
    ax.add_artist(ar3)
    ax.add_artist(ar4)
    ax.add_artist(ar5)
    ax.add_artist(ar6)

    # Add text
    text(-0.4, 1.*exp(-1.), r'$\tau$')
    text(2.7, 1.*exp(-1.), r'$\tau s$')
    text(-4.0, 0.5 , r"$A$")
    text(1.2, 0.01, r"$t_{\mathrm{offset}}$")

    savefig('documents/word.pdf', bbox_inches='tight')
#    show()
    close()

    return

def example_model():

    times = np.arange(1000)/1000.0
    w = word.TwoExp(times)

    t0 = 0.5
    scale = 0.05
    amp = 1.0
    skew = 5.0

    p = parameters.TwoExpParameters(t0=t0, scale=scale, amp=amp, skew=skew, log=False, bkg=None)

    word_counts = w.model(p)

    fig = plt.figure(figsize=(18,6))

    ax = fig.add_subplot(1,3,1)
    plot(times, word_counts, lw=2, color='black')
    xlabel(r"$\xi$")
    ylabel('Counts per bin in arbitrary units')
    ax.set_title(r'single word, $t=0.5$, $\tau=0.05$, $A=1$, $s=5$', fontsize=13)


    counts = np.ones(len(times))
    b = burstmodel.BurstDict(times, counts, [word.TwoExp for w in range(3)])

    wordparams = [0.1, np.log(0.05), np.log(60.0), np.log(5.0), 0.4, np.log(0.01),
                  np.log(100.0), np.log(1.0), 0.7, np.log(0.04), np.log(50), -2, np.log(10.0)]

    p = parameters.TwoExpCombined(wordparams, 3, scale_locked=False, skew_locked=False, log=True, bkg=True)

    model_counts = b.model_means(p)

    ax = fig.add_subplot(1,3,2)

    plot(times, model_counts, lw=2, color='black')
    xlabel(r"Time $t$ [s]")
    ylabel('Counts per bin in arbitrary units')
    ax.set_title(r'three words, $t=0.1,0.4,0.7$, $\tau=0.05, 0.01, 0.04$' + "\n" + r'$A=60, 100, 50$, $s=5,1,0.1$', fontsize=13)

    poisson_counts = np.array([np.random.poisson(c) for c in model_counts])

    ax = fig.add_subplot(1,3,3)
    plot(times, poisson_counts, lw=2, color='black')
    xlabel(r"Time $t$ [s]")
    ylabel('Counts per bin in arbitrary units')
    #title("\n".join(textwrap.wrap(r'three words, /w Poisson, $t=0.1,0.4,0.7$, $\tau=0.05, 0.01, 0.04$, $A=60, 100, 50$, $s=5,1,0.1$', 60)))
    ax.set_title(r'three words, /w Poisson, $t=0.1,0.4,0.7$, $\tau=0.05, 0.01, 0.04$,' + "\n" + r'$A=60, 100, 50$, $s=5,1,0.1$', fontsize=13)

    savefig("documents/example_words.pdf", format='pdf')
    close()

    return


def plot_example_bursts():
    """
    Makes Figure 1 of the paper

    """

    filenames = ["090122218_+048.206_all_data.dat", "090122194_+058.836_all_data.dat", "090122173_+241.347_all_data.dat",
                 "090122283_+131.840_all_data.dat", "090122283_+247.198_all_data.dat", "090122044_-000.043_all_data.dat"]

    alltimes, allcounts, allbintimes, allbincountrate= [], [], [], []
    for f in filenames:
        times, counts = burstmodel.read_gbm_lightcurves(f)
        countrate = np.array(counts)/(times[1] - times[0])
        bintimes, bincountrate = burstmodel.rebin_lightcurve(times, countrate, 10)
        bintimes = bintimes - bintimes[0]
        times = times - times[0]
        alltimes.append(times)
        allcounts.append(counts)
        allbintimes.append(bintimes)
        allbincountrate.append(bincountrate)

    fig = figure(figsize=(25,12))
    subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.97, wspace=0.1, hspace=0.2)

    sns.set_style("white")
    ax = fig.add_subplot(111)

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    sns.despine()

    sns.set_style("darkgrid")

    rc("font", size=20, family="serif", serif="Computer Sans")
    rc("axes", titlesize=16, labelsize=20)
    rc("text", usetex=True)


    for i in range(6):
        ax1 = fig.add_subplot(2,3,i)
        ax1.plot(allbintimes[i], allbincountrate[i]/10000.0, lw=2, color="black", linestyle='steps-mid')
        f = filenames[i].split("_")
        ax1.xaxis.set_ticks(np.arange(0.0, allbintimes[i][-1], 0.1))
        axis([allbintimes[i][0], allbintimes[i][-1], 0.1, 16.0])
 
        #xlabel("Time since trigger [s]")
        #ylabel(r"Count rate [$10^{4} \, \mathrm{cts}/\mathrm{s}$]")
        ax1.set_title("ObsID " + f[0] + r", $t_{\mathrm{start}} = $" + str(float(f[1])), fontsize=20)

    ax.set_xlabel("Time since burst start [s]", fontsize=24)
    ax.set_ylabel(r"Count rate [$10^{4} \, \mathrm{counts} \; \mathrm{s}^{-1}$]", fontsize=24)
    savefig("example_bursts.pdf", format='pdf')
    plt.close()

    return


def plot_example_dnest_lightcurve():

    data = loadtxt("090122173_+241.347_all_data.dat")
    fig = figure(figsize=(20,8))
    subplots_adjust(top=0.9, bottom=0.1, left=0.06, right=0.97, wspace=0.15, hspace=0.1)

    ax = fig.add_subplot(121)
    ax.plot(data[:,0]-data[0,0], (data[:,1]/0.0005)/1.e4, lw=2, color="black", linestyle="steps-mid")
    sample = atleast_2d(loadtxt("090122173_+241.347_posterior_sample.txt"))

    print(sample.shape)
    ind = np.random.choice(np.arange(len(sample)), replace=False, size=10)
    for i in ind:
        ax.plot(data[:,0]-data[0,0], (sample[i,-data.shape[0]:]/0.0005)/1.e4, lw=1)
    ax.set_ylim([0.01, 9.])
    ax.set_xlim([0.0,0.8])
    ax.set_xlabel("Time since burst start [s]", fontsize=24)
    ax.set_ylabel(r"Count rate [$10^{4} \, \mathrm{counts} \, \mathrm{s}^{-1}$]", fontsize=24, labelpad=-1)


    ax2 = fig.add_subplot(122)
    nbursts = sample[:, 7]

    ax2.hist(nbursts, bins=30, range=[np.min(nbursts), np.max(nbursts)], histtype='stepfilled')
    ax2.set_ylim([0.1, 25.])
    ax2.set_xlabel("Number of spikes per burst", fontsize=24)
    ax2.set_ylabel("N(samples)", fontsize=24, labelpad=-1)
    plt.savefig("example_dnest_result.pdf", format="pdf")
    plt.close()

    return



def make_lightcurve_grid():
 
    #files = glob.glob("*data.dat")
    #low_fluence, mid_fluence, high_fluence = [], [], []
    #for f in files:
    #    data = np.loadtxt(f)
    #    max_cr = np.max((data[:,1]/0.0005)/1.e4)
    #    fluence = np.sum(data[:,1])
    #    samplefile = glob.glob(f[:18] + "*posterior_sample.txt")[0]
    #    sample = np.atleast_2d(np.loadtxt(samplefile))
    #    nbursts = nbursts = sample[:, 7]
    #    lcs = (sample[:,-data.shape[0]:]/0.0005)/1.e4
    #    plt.figure(figsize=(12,6))
    #    plt.plot(data[:,0]-data[0,0], (data[:,1]/0.0005)/1.e4, lw=2)
    #    sampleinds = np.random.randint(0,len(lcs), 10)
    #    for s in sampleinds:
    #        plt.plot(data[:,0]-data[0,0],lcs[s, :], lw=1)
    #    bid = f.split("_")[0]
    #    bst = f.split("_")[1]
    #    plt.title("ObsID: %s, t_st = %s, nbursts = %i, fluence =%i"%(bid, bst, np.mean(nbursts), fluence))
    #    if max_cr < 3.:
    #        tag = "lf"
    #    elif 3. <= max_cr <= 8.:
    #        tag = "mf"
    #    else:
    #        tag = "hf"
    #    plt.savefig(tag + "_" + bid + "_" + bst + "lcs.pdf", format="pdf")
    #    plt.close()

    
    #plot_files = [["090122044_+288.53", "090122037a_+146.71", "090122104_+164.55"],
    #              ["090122052_-024.53", "090122037a_+143.09", "090122283_+131.84"],
    #              ["090122037a_+180.67", "090122044_+031.42", "090122187_+064.99"],
    #              ["090222540_-000.15", "090122194_+058.83", "090122173_+241.34"],
    #              ["090122052_+237.81", "090122037a_+142.46", "090122187_+292.95"]]

    #plot_nbursts = [[1, 2, 3],
    #                [4, 5, 5],
    #                [8, 10, 8],
    #                [15, 13 ,13],
    #                [48, 33, 13]]


    ### Picking bursts by maximum count rate
    ##
    ## 1,1: 081003377a, t_st = -000.048, nbursts = 1, fluence =1723
    ## 1,2: 090122037a, t_st = +143.096, nbursts = 5, fluence =1502
    ## 1,3: 090122037a, t_st = +155.722, nbursts = 6, fluence =1154
    ## 1,5: 090122037a, t_st = +142.468, nbursts = 33, fluence =1210
 
    ## LF: 137 bursts, 20 > 10 mean spikes, much higher values!

    ## 2,1: 081004050, t_st = -000.023, nbursts = 1, fluence =1057
    ## 2,2: 090202862, t_st = -000.048, nbursts = 5, fluence =3143
    ## 2,3: 090122052, t_st = +185.790, nbursts = 9, fluence =2409
    ## 2.4: 090122283, t_st = +145.752, nbursts = 13, fluence =7660 

    ## MF:
    ## only 5 out of 131 bursts with > 10 mean spikes

    ## 3,1: 090122104, t_st = -000.013, nbursts = 2, fluence =2446
    ## 3,2: 090210941, t_st = -000.032, nbursts = 5, fluence =2315
    ## 3,3: 090129588, t_st = +000.069, nbursts = 9, fluence =6519
    ## 3,5: 090122194, t_st = +058.836, nbursts = 13, fluence =4212

    ## 11 our of 63 bursts > 10, but none above 15 --> well constrained

    plot_files = [["081003377a_-000.04", "081004050_-000.02", "090122104_-000.013"],
                  ["090122037a_+143.09", "090202862_-000.04", "090210941_-000.032"],
                  ["090122037a_+155.72", "090122052_+185.79", "090129588_+000.06"],
                  ["090122037a_+142.46", "090122283_+145.75", "090122194_+058.83"]]

    plot_nbursts = [[1,1,2],
                    [5,5,5],
                    [6,9,9],
                    [33,13,13]]


    fig = plt.figure(figsize=(18, 20)) 
    plt.subplots_adjust(top=0.98, bottom=0.05, left=0.05, right=0.98, wspace=0.15, hspace=0.15)
    sns.set_style("white")
    ax = fig.add_subplot(111)
    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
    sns.despine()

    sns.set_style("darkgrid")
    rc("font", size=20, family="serif", serif="Computer Sans")
    rc("axes", titlesize=16, labelsize=20)
    rc("text", usetex=True) 
    plt.subplots_adjust(top=0.98, bottom=0.05, left=0.05, right=0.98, wspace=0.15, hspace=0.15)

    for i in range(4):
        for j in range(3):
            print("I am on file: " + plot_files[i][j])
            ax2 = fig.add_subplot(4,3,(i*3)+j+1)
            dfile = glob.glob(plot_files[i][j] + "*_data.dat")[0]
            data = np.loadtxt(dfile)
            ax2.plot(data[:,0]-data[0,0], (data[:,1]/0.0005)/1.e4, lw=2, color="black")
            samplefile = glob.glob(plot_files[i][j] + "*posterior_sample.txt")[0]
            sample = np.atleast_2d(np.loadtxt(samplefile))
            lcs = (sample[:,-data.shape[0]:]/0.0005)/1.e4
            sampleinds = np.random.randint(0,len(lcs), 10)
            for s in sampleinds:
                ax2.plot(data[:,0]-data[0,0],lcs[s, :], lw=1)
            bid = plot_files[i][j].split("_")[0]
            bst = plot_files[i][j].split("_")[1] 
            ax2.set_title("ObsID: %s, $t_\mathrm{start} = %.3f$s, $\mu_{nspikes} = %i$"%(bid, np.float(bst), plot_nbursts[i][j]),fontsize=16)
            ax2.set_xlim([0.001, data[-1,0]-data[0,0]])
            ax2.tick_params(axis='both', which='major', labelsize=16)

    ax.set_xlabel("Time since burst start [s]", fontsize=20, labelpad=15)
    ax.set_ylabel("Count rate [$10^{4}$ counts/s]", fontsize=20, labelpad=15)
    #plt.draw()
    #plt.tight_layout()
    plt.savefig("f4b.pdf", format="pdf")
    plt.close()



    ### bursts for plotting:
    ## 0,0: 090122044_+288.534 , nbursts = 1, fluence =482
    ## 0,1: 090122052, t_st = -024.534, nbursts = 4, fluence =746
    ## 0,2: 090122037a, t_st = +180.676, nbursts = 8, fluence =878
    ## 0,3: 090222540, t_st = -000.152, nbursts = 15, fluence =787
    ## 0,4: 090122052, t_st = +237.815, nbursts = 48, fluence =259
 
    ## 1,0: 090122037a, t_st = +146.710, nbursts = 2, fluence =1773
    ## 1,1: 090122037a, t_st = +143.096, nbursts = 5, fluence =1502
    ## 1,2: 090122044, t_st = +031.426, nbursts = 10, fluence =2233
    ## 1,3: 1090122194, t_st = +058.836, nbursts = 13, fluence =4212
    ## 1,4: 090122037a, t_st = +142.468, nbursts = 33, fluence =1210
     
    ## 2,0: 090122104, t_st = +164.550, nbursts = 3, fluence =5805
    ## 2,1: 090122283, t_st = +131.840, nbursts = 5, fluence =10506
    ## 2,2: 090122187, t_st = +064.998, nbursts = 8, fluence =5606
    ## 2,3: 090122173, t_st = +241.347, nbursts = 13, fluence =8587
    ## 2,4: 090122187, t_st = +292.952, nbursts = 13, fluence =7275


#        if fluence < 1000:
#            low_fluence.append([f, data, nbursts, lcs])
#        elif 1000. <= fluence < 10000.:
#            mid_fluence.append([f, data, nbursts, lcs])
#        else:
#            high_fluence.append([f, data, nbursts, lcs])


    return


def nspike_plot(par_unfiltered=None, par_filtered=None, datadir="./", nsims=100):
    """
    Make a histogram of the number of spikes per bin

    NOTE: requires files sgr1550_ttrig.dat and sgr1550_fluence.dat

    @param datadir: directory with posterior files and data files.
    @return:
    """

    ### parameters both filtered for low-amplitude bursts (with amp < bkg) and unfiltered
    if par_unfiltered is None and par_filtered is None:
        par_filtered, bids_filtered, hyper_filtered = \
            dnest_sample.extract_sample(datadir=datadir,nsims=nsims, filter_weak=True, trigfile="sgr1550_ttrig.dat")

        par_unfiltered, bids_unfiltered, hyper_filered = \
            dnest_sample.extract_sample(datadir=datadir, nsims=nsims, filter_weak=False, trigfile="sgr1550_ttrig.dat")


    nspikes_unfiltered, nspikes_filtered = [], []
    for i in xrange(nsims):

        sample_unfiltered = par_unfiltered[:,i]
        sample_filtered = par_filtered[:,i]

        nspikes_u = np.array([len(s.all) for s in sample_unfiltered])
        nspikes_f = np.array([len(s.all) for s in sample_filtered])

        nspikes_unfiltered.append(nspikes_u)
        nspikes_filtered.append(nspikes_f)


    fig = figure(figsize=(12,9))
    ax = fig.add_subplot(111)
    nu_all, nf_all = [], []
    for i,(u,f) in enumerate(zip(nspikes_unfiltered, nspikes_filtered)):

        nu, ubins = np.histogram(u, bins=50, range=[1,50], normed=False)
        nu_all.append(nu)

        nf, fbins = np.histogram(f, bins=50, range=[1,50], normed=False)
        nf_all.append(nf)

    #print("nu shape " + str(np.shape(nu_all)))
    nu_mean = np.mean(np.array(nu_all), axis=0)
    nf_mean = np.mean(np.array(nf_all), axis=0)

    y_bottom = np.zeros(len(nu_mean))

    #print("len ubins: %i"%(len(ubins)))
    #print("len y_bottom: %i"%(len(y_bottom)))
    #print("len nu_means: %i"%(len(nu_mean)))

    #ax.plot(ubins[:-1]+0.5, nu_mean, lw=2, color="navy", linestyle="steps-mid")
    #ax.fill(ubins[:-1]+0.5, 0.0, nu_mean, color="navy", alpha=0.7)
    #ax.plot(fbins[:-1]+0.5, nf_mean, lw=2, color="darkred", linestyle="steps-mid")
    #ax.fill_between(fbins[:-1]+0.5, y_bottom, nf_mean, color="darkred", alpha=0.7, drawstyle='steps-mid')

    #ax.plot(ubins[:-1]+0.5, nu_mean, color='navy',
    #        linewidth=2, label=None, linestyle="steps-mid")

    sns.distplot(u, bins=50, ax=ax,label="unfiltered sample", kde=False,
                 hist_kws={"histtype": "stepfilled", "range":[1,50], "alpha":0.6})
    #ax.bar(ubins[:-1]+0.5, nu_mean, ubins[1]-ubins[0]+0.005, color='navy',
    #       alpha=0.6, linewidth=0, align="center", label="unfiltered sample")
    #ax.plot(fbins[:-1]+0.5, nf_mean, color='darkred',
    #        linewidth=2, label=None, linestyle="steps-mid")

    sns.distplot(f, ax=ax,label="filtered sample", bins=50, kde=False,
                 hist_kws={"histtype": "stepfilled", "range":[1,50], "alpha":0.6})

    #ax.bar(fbins[:-1]+0.5, nf_mean, fbins[1]-fbins[0]+0.005, color='darkred',
    #       alpha=0.6, linewidth=0, align="center", label="filtered sample")

    axis([1,30, 0.0, 70])

    xlabel("Number of components", fontsize=24)
    ylabel("Number of bursts", fontsize=24)
    title("distribution of the number of components per burst")
    legend(loc="upper right", prop={"size":24})
    #savefig("sgr1550_nspikes.png", format="png")
    #close()
    draw()
    plt.tight_layout()
    savefig("f4.pdf", format="pdf")
    close()

    return



def waitingtime_plot(sample=None, bids=None, datadir="./", nsims=100, mean=True, filtered=True):

    if sample is None and bids is None:
        sample,bids,hyper = dnest_sample.extract_sample(datadir=datadir, nsims=nsims,
                                                          filter_weak=filtered, trigfile="sgr1550_ttrig.dat")



    waitingtimes = dnest_sample.waiting_times(sample, bids, datadir=datadir, nsims=nsims, mean=mean,
                                              trigfile="sgr1550_ttrig.dat", froot="sgr1550")

    return


def differential_plots(sample=None, datadir="./", nsims=100, mean=True, filtered=True, froot="sgr1550"):

    if sample is None:
        sample,bids,hyper = dnest_sample.extract_sample(datadir=datadir, nsims=nsims,
                                                          filter_weak=filtered, trigfile="sgr1550_ttrig.dat")

    dnest_sample.differential_distributions(sample, nsims=100, mean=True, froot=froot)

    return


def priors_differentials(par_exp=None, par_logn=None,  datadir="./", nsims=100, filtered=False):

    #if par_exp is None:
    #    par_exp,bids_exp = dnest_sample.extract_sample(datadir=datadir+"expprior/", nsims=nsims,
    #                                                      filter_weak=filtered, trigfile="sgr1550_ttrig.dat")

    if par_logn is None:
        par_logn, bids_logn, hyper_logn = dnest_sample.extract_sample(datadir=datadir+"lognormalprior/", nsims=nsims,
                                                          filter_weak=filtered, trigfile="sgr1550_ttrig.dat")

    #if par_gauss is None:
    #    par_gauss, bids_gauss = dnest_sample.extract_sample(datadir=datadir+"gaussprior/", nsims=nsims,
    #                                                      filter_weak=filtered, trigfile="sgr1550_ttrig.dat")



    db_exp, nd_exp, ab_exp, na_exp, fb_exp, nf_exp = \
        dnest_sample.differential_distributions(sample=par_exp,  nsims=nsims, makeplot=False,
                                                dt=0.0005, mean=True, normed=False)

    db_logn, nd_logn, ab_logn, na_logn, fb_logn, nf_logn = \
        dnest_sample.differential_distributions(sample=par_logn,  nsims=nsims, makeplot=False,
                                                dt=0.0005, mean=True, normed=False)

    #db_gauss, nd_gauss, ab_gauss, na_gauss, fb_gauss, nf_gauss = \
    #    dnest_sample.differential_distributions(sample=par_gauss,  nsims=nsims, makeplot=False,
    #                                            dt=0.0005, mean=True, normed=True)
    current_palette = sns.color_palette()
    fig = figure(figsize=(24,8))
    subplots_adjust(top=0.9, bottom=0.1, left=0.03, right=0.97, wspace=0.15, hspace=0.2)

    ### first subplot: differential duration distribution
    ax = fig.add_subplot(131)

    ax.bar(db_exp[:-1]+0.5, nd_exp, db_exp[1]-db_exp[0], color = current_palette[0], 
               alpha=0.7, linewidth=0, align="center", label="exponential prior")

    ax.bar(db_logn[:-1]+0.5, nd_logn, db_logn[1]-db_logn[0], color = current_palette[1],
               alpha=0.7, linewidth=0, align="center", label="log-normal prior")

    #ax.bar(db_gauss[:-1]+0.5, nd_gauss, db_gauss[1]-db_gauss[0], color='limegreen',
    #           alpha=0.7, linewidth=0, align="center", label="normal prior")

    axis([-4.0, np.log10(30.0), 0.0, np.max([np.max(nd_exp), np.max(nd_logn)])+10.0])
    legend(loc="upper left", prop={"size":20})

    ax.set_xlabel(r"$\log_{10}{(\mathrm{Duration})}$ [s]", fontsize=24)
    ax.set_ylabel("N($\log_{10}{\mathrm{Duration}}$)", fontsize=24)
    #ax.set_title("Differential Duration Distribution", fontsize=24)

    ax1 = fig.add_subplot(132)


    min_a, max_a = [], []

    ax1.bar(ab_exp[:-1]+0.5, na_exp, ab_exp[1]-ab_exp[0], color = current_palette[0],
               alpha=0.6, linewidth=0, align="center", label="exponential prior")

    ax1.bar(ab_logn[:-1]+0.5, na_logn, ab_logn[1]-ab_logn[0], color = current_palette[1],
               alpha=0.6, linewidth=0, align="center", label="log-normal prior")

    #ax1.bar(ab_gauss[:-1]+0.5, na_gauss, ab_gauss[1]-ab_gauss[0], color='limegreen',
    #           alpha=0.7, linewidth=0, align="center", label="normal prior")

    axis([np.log10(0.001), 3.0, 0.0, np.max([np.max(na_exp), np.max(na_logn)])+10.0])

    ax1.set_xlabel(r"$\log_{10}{(\mathrm{Amplitude})}$ [counts/s]", fontsize=24)
    ax1.set_ylabel("N($\log_{10}{\mathrm{Amplitude}}$)", fontsize=24)
    #ax1.set_title("Differential Amplitude Distribution", fontsize=24)

    ax2 = fig.add_subplot(133)
    ax2.bar(fb_exp[:-1]+0.5, nf_exp, fb_exp[1]-fb_exp[0], color= current_palette[0],
               alpha=0.6, linewidth=0, align="center", label="exponential prior")

    ax2.bar(fb_logn[:-1]+0.5, nf_logn, fb_logn[1]-fb_logn[0], color = current_palette[1],
               alpha=0.6, linewidth=0, align="center", label="log-normal prior")

    #ax2.bar(fb_gauss[:-1]+0.5, nf_gauss, fb_gauss[1]-fb_gauss[0], color='limegreen',
    #           alpha=0.7, linewidth=0, align="center", label="normal prior")


    axis([-13.9, -4.9, 0.0, np.max([np.max(nf_exp), np.max(nf_logn)])+10.0])
    ax2.set_xticks(np.arange(-13.0, -4.0, 2.0))
    ax2.set_xlabel(r"$\log_{10}{(\mathrm{Fluence})}$ [erg/cm$^2$]", fontsize=24)
    ax2.set_ylabel("N($\log_{10}{\mathrm{Fluence}}$)", fontsize=24)
    #ax2.set_title("Differential Fluence Distribution", fontsize=24)

    draw()
    plt.tight_layout()

    savefig("f10.pdf", format="pdf")
    close()


    return



def correlation_plots(sample=None, datadir="./", nsims=100, filtered=True, froot="sgr1550"):

    """
    Make plots with correlations for duration versus fluence and rise time versus fluence.


    @param sample:
    @param datadir:
    @param nsims:
    @return:
    """


    if sample is None:
        parameters_red,bids, hyper = dnest_sample.extract_sample(datadir=datadir, nsims=nsims,
                                                          filter_weak=filtered, trigfile="sgr1550_ttrig.dat")
    else:
        parameters_red = sample

    fluence_sample, duration_sample, risetime_sample = [], [], []

    for i in xrange(nsims):

        sample = parameters_red[:,i]
        fluence_all = np.array([np.array([a.fluence for a in s.all if a.duration > 0.0]) for s in sample])

        #fluence_all = fluence_all.flatten()
        duration_all = np.array([np.array([a.duration for a in s.all if a.duration > 0.0]) for s in sample])
        #risetime_all = risetime_all.flatten()

        risetime_all = np.array([np.array([a.scale for a in s.all if a.duration > 0.0]) for s in sample])

        #print("len fluence: " + str(len(fluence_all)))
        #print("len risetime: " + str(len(risetime_all)))

        fluence, duration, risetime = [], [], []
        for a,e,d in zip(risetime_all, fluence_all, duration_all):
            fluence.extend(e)
            duration.extend(d)
            risetime.extend(a)

        #print("len fluence: " + str(len(fluence)))
        #print("len risetime: " + str(len(risetime)))


        fluence_sample.append(fluence)
        duration_sample.append(duration)
        risetime_sample.append(risetime)

    ### samples for scatter plot
    fsamp = np.log10(fluence_sample[0])
    dsamp = np.log10(duration_sample[0])
    rsamp = np.log10(risetime_sample[0])


    fluence_flat, duration_flat, risetime_flat = [], [], []
    spdf, sprf = [], []

    for f,d,r in zip(fluence_sample, duration_sample, risetime_sample):
        fluence_flat.extend(np.log10(f))
        duration_flat.extend(np.log10(d))
        risetime_flat.extend(np.log10(r))

        spdf.append(scipy.stats.spearmanr(d,f))
        sprf.append(scipy.stats.spearmanr(r,f))


    fluence_flat = np.array(fluence_flat)
    risetime_flat = np.array(risetime_flat)
    duration_flat = np.array(duration_flat)

    spdf_mean = np.mean(spdf, axis=0)
    spdf_std = np.std(spdf, axis=0)
    sprf_mean = np.mean(sprf, axis=0)
    sprf_std = np.std(spdf, axis=0)


    fig = figure(figsize=(18,9))
    #subplots_adjust(top=0.9, bottom=0.1, left=0.03, right=0.97, wspace=0.15, hspace=0.2)

    ### first plot: duration versus fluence
    ax1 = fig.add_subplot(121)

    levels = [0.01, 0.05, 0.1, 0.2, 0.3]


    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

    #xmin, xmax = duration_flat.min(), duration_flat.max()
    #ymin, ymax = fluence_flat.min(), fluence_flat.max()
    xmin = -3.1
    xmax = 1.0
    ymin = -13.0
    ymax = -7.5
    ### Perform Kernel density estimate on data
    try:
        #X,Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        #positions = np.vstack([X.ravel(), Y.ravel()])
        #values = np.vstack([duration_flat, fluence_flat])
        #kernel = scipy.stats.gaussian_kde(values)
        #Z = np.reshape(kernel(positions).T, X.shape)


        sns.kdeplot(duration_flat, fluence_flat, cmap=cmap, shade=True, cut=5, ax=ax1)
        #im = ax1.imshow(np.transpose(Z), interpolation='nearest', origin='lower',
        #        cmap=cm.PuBuGn, extent=(-4.0,2.0,ymin,ymax))
        #im.set_clim(0.0,0.5)
        #plt.colorbar(im)

        #znew = scipy.ndimage.gaussian_filter(Z, sigma=4.0, order=0)
        #cs = ax1.contour(X,Y,znew,levels,
        #                 linewidths=2, colors="black", origin="lower")
        #manual_locations = [(0.0, -10.5), (-0.9, -11.5), (-0.7,-10.7), (-1.7,11.1), (-1.1,-10.4)]
        #plt.clabel(cs, fontsize=24, inline=1, manual=manual_locations)

    except ValueError:
        print("Not making contours.")


    ax1.scatter(dsamp, fsamp, color="black", alpha=0.7)

    ax1.text(0.5,0.05, r"Spearman Rank Coefficient $R = %.2f \pm %.2f$"%(spdf_mean[0], spdf_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax1.transAxes,
                fontsize=22)
    ax1.axis([-3., 0.0, -13.0, -7.])


    ax1.set_xlabel(r"$\log_{10}{(\mathrm{Duration} \; \mathrm{[s]})}$ ", fontsize=24)
    ax1.set_ylabel(r"$\log_{10}{(\mathrm{Fluence})}$ [$\mathrm{erg} \, \mathrm{cm}^{-1}$]", fontsize=24)


    ### first plot: duration versus fluence
    ax2 = fig.add_subplot(122, sharey=ax1)

    #xmin, xmax = risetime_flat.min(), risetime_flat.max()
    #ymin, ymax = fluence_flat.min(), fluence_flat.max()

    xmin = -4.5
    xmax = 0.5
    ymin = -13.0
    ymax = -7.5

    ### Perform Kernel density estimate on data
    try:
        #X,Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        #positions = np.vstack([X.ravel(), Y.ravel()])
        #values = np.vstack([risetime_flat, fluence_flat])
        #kernel = scipy.stats.gaussian_kde(values)
        #Z = np.reshape(kernel(positions).T, X.shape)

        #im = ax2.imshow(np.transpose(Z), interpolation='nearest', origin='lower',
        #        cmap=cm.PuBuGn, extent=(xmin,xmax,ymin,ymax))

        #im.set_clim(0.0,0.5)
        #plt.colorbar(im)
        sns.kdeplot(risetime_flat, fluence_flat, cmap=cmap, shade=True, cut=5, ax=ax2)

        #znew = scipy.ndimage.gaussian_filter(Z, sigma=4.0, order=0)
        #cs = ax2.contour(X,Y,znew,levels,
        #                 linewidths=2, colors="black", origin="lower")
        #manual_locations = [(-2.0, -12.0),(-1.75,-11.2),(-2.75,-11.5),(-2.25,-11.0),(-2.3,-10.2)]
        #clabel(cs, fontsize=24, inline=1, manual=manual_locations)

    except ValueError:
        print("Not making contours.")


    ax2.scatter(rsamp, fsamp, color="black", alpha=0.7)
    ax2.axis([-4.,0., -13.0, -7.])
    ax2.text(0.5,0.1, r"Spearman Rank Coefficient $R = %.2f \pm %.2f$"%(sprf_mean[0], sprf_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax2.transAxes,
                fontsize=22)
    ax2.set_xlabel(r"$\log_{10}{(\mathrm{Rise\, timescale} \; \mathrm{[s]})}$ ", fontsize=24)
    setp(ax2.get_yticklabels(), visible=False)
    draw()
    plt.tight_layout()

    savefig('f7.pdf', format="pdf")
    close()

    return


def correlation_plots_skewness(sample=None, datadir="./", nsims=100, filtered=True, froot="sgr1550"):

    """
    Make plots with correlations for duration versus flux and rise time versus flux.


    @param sample:
    @param datadir:
    @param nsims:
    @return:
    """


    if sample is None:
        parameters_red,bids,hyper = dnest_sample.extract_sample(datadir=datadir, nsims=nsims,
                                                          filter_weak=filtered, trigfile="sgr1550_ttrig.dat")
    else:
        parameters_red = sample

    skewness_sample, flux_sample, duration_sample, amplitude_sample = [], [], [], []

    for i in xrange(nsims):

        sample = parameters_red[:,i]
        #amplitude_all = np.array([np.array([a.amp for a in s.all if a.duration > 0.0]) for s in sample])
        #risetime_all = np.array([np.array([a.scale for a in s.all if a.duration > 0.0]) for s in sample])
        #flux_all = [a/r for a,r in zip(amplitude_all, risetime_all)]
#        duration_all = np.array([np.array([a.duration for a in s.all if a.duration > 0.0]) for s in sample])

#        skewness_all = np.array([np.array([a.skew for a in s.all if a.duration > 0.0]) for s in sample])


        amplitude_all = np.array([np.array([a.amp for a in s.all]) for s in sample])
        risetime_all = np.array([np.array([a.scale for a in s.all]) for s in sample])
        flux_all = [a/r for a,r in zip(amplitude_all, risetime_all)]
        skewness_all = np.array([np.array([a.skew for a in s.all]) for s in sample])
        duration_all = np.array([np.array([a.duration for a in s.all]) for s in sample])



        #print("len flux: " + str(len(amplitude_all)))
        #print("len skewness: " + str(len(risetime_all)))

        flux, duration, skewness, amplitude = [], [], [], []
        for s,a,f,d in zip(skewness_all, amplitude_all, flux_all, duration_all):
            flux.extend(np.log10(f))
            duration.extend(np.log10(d))
            skewness.extend(np.log10(s))
            amplitude.extend(np.log10(a))

        #print("len flux: " + str(len(flux)))
        #print("len skewness: " + str(len(skewness)))


        flux_sample.append(flux)
        duration_sample.append(duration)
        skewness_sample.append(skewness)
        amplitude_sample.append(amplitude)


    ### samples for scatter plot
    fsamp = flux_sample[0]
    dsamp = duration_sample[0]
    ssamp = skewness_sample[0]
    asamp = amplitude_sample[0]

    flux_flat, duration_flat, skewness_flat, amplitude_flat = [], [], [], []
    spda, spsf = [], []

    for f,d,s,a in zip(flux_sample, duration_sample, skewness_sample, amplitude_sample):
        flux_flat.extend(f)
        duration_flat.extend(d)
        skewness_flat.extend(s)
        amplitude_flat.extend(a)

        spda.append(scipy.stats.spearmanr(d,a))
        spsf.append(scipy.stats.spearmanr(f,s))


    flux_flat = np.array(flux_flat)
    skewness_flat = np.array(skewness_flat)
    duration_flat = np.array(duration_flat)
    amplitude_flat = np.array(amplitude_flat)

    #spda_mean = np.mean(spda, axis=0)
    #spda_std = np.std(spda, axis=0)
    spsf_mean = np.mean(spsf, axis=0)
    spsf_std = np.std(spsf, axis=0)


    fig = figure(figsize=(12,9))
    #subplots_adjust(top=0.9, bottom=0.1, left=0.03, right=0.97, wspace=0.15, hspace=0.2)

    ### first plot: duration versus flux
    #ax1 = fig.add_subplot(121)
    ax1 = fig.add_subplot(111)
    levels = [0.01, 0.05, 0.1, 0.2, 0.3]


    #xmin, xmax = duration_flat.min(), duration_flat.max()
    #ymin, ymax = fluence_flat.min(), fluence_flat.max()
    xmin = 0.0 #np.min(flux_flat)
    xmax = 6.0 #np.max(flux_flat)
    ymin = -2.5 #np.min(skewness_flat)
    ymax = 3.0 #np.max(skewness_flat)
    ### Perform Kernel density estimate on data
    cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

    try:
        #X,Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        #positions = np.vstack([X.ravel(), Y.ravel()])
        #values = np.vstack([flux_flat, skewness_flat])
        #kernel = scipy.stats.gaussian_kde(values)
        #Z = np.reshape(kernel(positions).T, X.shape)


        #im = ax1.imshow(np.transpose(Z), interpolation='bicubic', origin='lower',
        #        cmap=cm.PuBuGn, extent=(xmin,xmax,ymin,ymax))
        #im.set_clim(0.0,0.5)
        #plt.colorbar(im)

        #znew = scipy.ndimage.gaussian_filter(Z, sigma=4.0, order=0)
        #cs = ax1.contour(X,Y,znew,levels, linewidths=2, colors="black", origin="lower")
        #manual_locations = [(0.0, -10.5), (-0.9, -11.5), (-0.7,-10.7), (-1.7,11.1), (-1.1,-10.4)]
        #plt.clabel(cs, fontsize=24, inline=1, manual=manual_locations)

        #divider = make_axes_locatable(ax1)
        #cax = divider.append_axes("right", size="5%", pad=0.5)

        #im.set_clim(0.0,0.5)
        #plt.colorbar(im, cax=cax)
        sns.kdeplot(flux_flat, skewness_flat, cmap=cmap, shade=True, cut=5, ax=ax1)


    except ValueError:
        print("Not making contours.")


    ax1.scatter(fsamp, ssamp, color="black")

    #ax1.text(0.5,0.05, r"Spearman Rank Coefficient $R = %.2f \pm %.2f$"%(spsf_mean[0], spsf_std[0]),
    #            verticalalignment='center', horizontalalignment='center', color='black', transform=ax1.transAxes,
    #            fontsize=18)
    ax1.axis([xmin,xmax,ymin,ymax])


    ax1.set_xlabel(r"$\log_{10}{(\mathrm{Flux}} \; A/\tau)$ [counts/s] ", fontsize=24)
    ax1.set_ylabel(r"$\log_{10}{(\mathrm{Skewness})}$", fontsize=24)

    ax1.text(0.5,0.05, r"Spearman Rank Coefficient $R = %.2f \pm %.2f$"%(spsf_mean[0], spsf_std[0]),
                verticalalignment='center', horizontalalignment='center', color='black', transform=ax1.transAxes,
                fontsize=22)

    ### first plot: duration versus flux
#    ax2 = fig.add_subplot(122)
#
#    #xmin, xmax = risetime_flat.min(), risetime_flat.max()
#    #ymin, ymax = fluence_flat.min(), fluence_flat.max()
#
#    xmin = -3.0 #np.min(duration_flat)
#    xmax = 2.0 #np.max(duration_flat)
#    ymin = np.min(amplitude_flat)
#    ymax = np.max(amplitude_flat)

    ### Perform Kernel density estimate on data
#    try:
#        X,Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
#        positions = np.vstack([X.ravel(), Y.ravel()])
#        values = np.vstack([duration_flat, amplitude_flat])
#        kernel = scipy.stats.gaussian_kde(values)
#        Z = np.reshape(kernel(positions).T, X.shape)

#        im = ax2.imshow(np.transpose(Z), interpolation='bicubic', origin='lower',
#                cmap=cm.PuBuGn, extent=(xmin,xmax,ymin,ymax))

#        im.set_clim(0.0,0.5)
#        plt.colorbar(im)

#        cs = ax2.contour(X,Y,Z,levels, linewidths=2, colors="black", origin="lower")
        #manual_locations = [(-2.0, -12.0),(-1.75,-11.2),(-2.75,-11.5),(-2.25,-11.0),(-2.3,-10.2)]
        #clabel(cs, fontsize=24, inline=1, manual=manual_locations)

#    except ValueError:
#        print("Not making contours.")


#    ax2.scatter(dsamp, asamp, color="black")
#    ax2.axis([xmin,xmax,ymin,ymax])
    #ax2.text(0.5,0.05, r"Spearman Rank Coefficient $R = %.2f \pm %.2f$"%(spda_mean[0], spda_std[0]),
    #            verticalalignment='center', horizontalalignment='center', color='black', transform=ax2.transAxes,
    #            fontsize=18)
#    ax2.set_xlabel(r"$\log_{10}{(\mathrm{Rise\, timescale} \; \mathrm{[s]})}$ ", fontsize=24)
    #setp(ax2.get_yticklabels(), visible=False)
#    ax2.set_ylabel(r"$\log_{10}{(\mathrm{Amplitude} \; \mathrm{[counts/bin]})}$ ", fontsize=24)
    draw()
    plt.tight_layout()

    savefig('f8.pdf', format="pdf")
    close()

    return


def all_dnest_plots(datadir="./", nsims=100, froot="sgr1550"):


    par_filtered, bids_filtered, hyper_filtered = \
        dnest_sample.extract_sample(datadir=datadir, nsims=nsims, filter_weak=True, trigfile="%s_ttrig.dat"%froot)

    par_unfiltered, bids_unfiltered, hyper_unfiltered = \
        dnest_sample.extract_sample(datadir=datadir, nsims=nsims, filter_weak=False, trigfile="%s_ttrig.dat"%froot)



    par_exp, bids_exp, hyper_exp = dnest_sample.extract_sample(datadir=datadir+"expprior/", nsims=nsims,
                                                      filter_weak=False, trigfile="%s_ttrig.dat"%froot)
    par_logn, bids_logn, hyper_logn = dnest_sample.extract_sample(datadir=datadir+"lognormalprior/", nsims=nsims,
                                                      filter_weak=False, trigfile="%s_ttrig.dat"%froot)



    nspike_plot(par_unfiltered, par_filtered)
    correlation_plots(par_filtered)
    correlation_plots_skewness(par_filtered)
    waitingtime_plot(par_unfiltered, bids_unfiltered)
    differential_plots(par_filtered)
    priors_nspikes(par_exp, par_logn)
    priors_differentials(par_exp, par_logn)

    return

def main():

    plot_example_bursts()

    return


if __name__ == "__main__":

    main()
