import numpy as np
import word
import burstmodel
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.cm as cm
import generaltools as gt

rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)



def example_model():

    times = np.arange(1000)/1000.0
    w = word.TwoExp(times)

    t0 = 0.5
    scale = 0.05
    amp = 1.0
    skew = 5.0

    word_counts = w.model(t0, scale, amp, skew)

    fig = plt.figure(figsize=(18,6))

    plt.subplot(1,3,1)
    plt.plot(times, word_counts, lw=2, color='black')
    plt.xlabel(r"$\xi$")
    plt.ylabel('Counts per bin in arbitrary units')
    plt.title(r'single word, $t=0.5$, $\tau=0.05$, $A=1$, $s=5$', fontsize=20)


    counts = np.ones(len(times))
    b = burstmodel.BurstDict(times, counts, [word.TwoExp for w in range(3)])

    wordparams = [0.1, np.log(0.05), np.log(60.0), np.log(5.0), 0.4, np.log(0.01),
                  np.log(100.0), np.log(1.0), 0.7, np.log(0.04), np.log(50), -2, np.log(10.0)]


    model_counts = b.model_means(wordparams)

    plt.subplot(1,3,2)

    plt.plot(times, model_counts, lw=2, color='black')
    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel('Counts per bin in arbitrary units')
    plt.title(r'three words, $t=0.1,0.4,0.7$, $\tau=0.05, 0.01, 0.04$, $A=60, 100, 50$, $s=5,1,0.1$', fontsize=13)

    poisson_counts = np.array([np.random.poisson(c) for c in model_counts])

    plt.subplot(1,3,3)
    plt.plot(times, poisson_counts, lw=2, color='black')
    plt.xlabel(r"Time $t$ [s]")
    plt.ylabel('Counts per bin in arbitrary units')
    plt.title(r'three words, /w Poisson, $t=0.1,0.4,0.7$, $\tau=0.05, 0.01, 0.04$, $A=60, 100, 50$, $s=5,1,0.1$', fontsize=13)

    plt.savefig("example_words.png", format='png')
    plt.close()

    return



def parameter_distributions(filename, namestr="allbursts"):

    """
    Make a bunch of distributions from the posteriors of a number of bursts.

    This requires a python pickle file with a dictionary of the posterior results of various
    quantities, right now with keywords:
    scale_max: posterior maximum of log(scale)
    scale_cl: 0.05 lower quantile of log(scale)
    scale_cu: 0.95 upper quantile of log(scale)
    skew_max: posterior maximum of log(skew)
    skew_cl: 0.05 lower quantile of log(skew)
    skew_cuL 0.95 upper quantile of log(skew)

    At the moment, scale and skew are fixed for all words within a burst, but this could easily be changed.
    Similarly, it should be just as easy to extend this to other parameters (e.g. position + amplitude).

    The kind of file needed here is easily created by manipulating function plot_all_bursts in plot_parameters.py

    """


    allparas = gt.getpickle(filename)
    scale_max = np.array(allparas["scale_max"])
    scale_cl = np.array(allparas["scale_cl"])
    scale_cu = np.array(allparas["scale_cu"])
    skew_max = np.array(allparas["skew_max"])
    skew_cl = np.array(allparas["skew_cl"])
    skew_cu = np.array(allparas["skew_cu"])

    ### plot maximum scale for 1,5 and 10 words,
    ### plot in log_10 instead of log_e for ease of interpreting timescales
    hist(np.log10(np.exp(scale_max[:,0])), bins=15, color='navy', histtype='stepfilled', alpha=0.7, label="1 word")
    hist(np.log10(np.exp(scale_max[:,4])), bins=15, color='darkred', histtype='stepfilled', alpha=0.7, label="5 words")
    hist(np.log10(np.exp(scale_max[:,9])), bins=15, color='green', histtype='stepfilled', alpha=0.7, label="10 words")
    plt.legend()
    plt.xlabel(r"$\log_{10}{(\mathrm{scale})}$ [s]", fontsize=18)
    plt.ylabel(r"$N(\mathrm{bursts})$", fontsize=18)
    plt.title("Distribution of rise times for the 85 brightest bursts")
    plt.savefig(namestr + "_scale.eps", format="eps")
    plt.close()

    ### DO same plot with skew parameters
    hist(np.log10(np.exp(skew_max[:,0])), bins=15, color='navy', histtype='stepfilled', alpha=0.7, label="1 word")
    hist(np.log10(np.exp(skew_max[:,4])), bins=15, color='darkred', histtype='stepfilled', alpha=0.7, label="5 words")
    hist(np.log10(np.exp(skew_max[:,9])), bins=15, color='green', histtype='stepfilled', alpha=0.7, label="10 words")
    plt.xlabel(r"$\log_{10}{(\mathrm{skew})}$ [s]", fontsize=18)
    plt.ylabel(r"$N(\mathrm{bursts})$", fontsize=18)
    plt.legend()
    plt.title("Distribution of skew parameters for the 85 brightest bursts")
    plt.savefig(namestr + "_skew.eps", format="eps")
    plt.close()

    ### Plot all scales as a function of number of spikes in one plot, just to be able to look at them
    ### all at once:
    figure()
    for l,m,u in zip(scale_cl, scale_max, scale_cu):
        plt.errorbar(np.arange(10)+1, np.log10(np.exp(m)), yerr=[np.log10(np.exp(m-l)),
                                                                np.log10(np.exp(u-m))], fmt="--o", lw=2)
    plt.xlabel("Number of spikes in model", fontsize=18)
    plt.ylabel(r"$\log_{10}{(\mathrm{scale})}$ [s]", fontsize=18)
    plt.title("Change of rise time with number of spikes in model, for 85 brightest bursts")
    plt.savefig(namestr + "_scale_per_nspikes.eps", format='eps')
    plt.close()


    ### Same for skew
    figure()
    for l,m,u in zip(skew_cl, skew_max, skew_cu):
        plt.errorbar(np.arange(10)+1, np.log10(np.exp(m)), yerr=[np.log10(np.exp(m-l)),
                                                                np.log10(np.exp(u-m))], fmt="--o", lw=2)
    plt.xlabel("Number of spikes in model", fontsize=18)
    plt.ylabel(r"$\log_{10}{(\mathrm{skew})}$ [s]", fontsize=18)
    plt.title("Change of skew with number of spikes in model, for 85 brightest bursts")
    plt.savefig(namestr + "_skew_per_nspikes.eps", format='eps')
    plt.close()



    ### number of bins for the histogram
    nbins = 15

    ### plot an image of the scale histograms:
    scale_hist_all = []
    scale_max_log10 = np.log10(np.exp(scale_max))
    smin = min(scale_max_log10.flatten())
    smax = np.max(scale_max_log10.flatten())

    for s in np.transpose(scale_max_log10):
        h,bins = np.histogram(s, bins=nbins, range=[smin, smax])
        scale_hist_all.append(h)

    fig, ax = subplots()
    imshow(scale_hist_all, cmap=cm.hot)
    plt.axis([-0.5, -0.5+nbins, -0.5, np.min(np.shape(scale_max_log10))-1])

    ### get limits of x-axis
    start, end = ax.get_xlim()
    ### define tick positions
    tickpos = np.arange(start+0.5, end, 2)
    ### define tick labels
    ticklabels = np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/float(len(tickpos)))
    ### stupid way of defining significant digits
    ticklabels = [str(l)[:5] for l in ticklabels]
    xticks(tickpos, ticklabels)
    plt.xlabel(r"$\log_{10}{(\mathrm{scale})}$ [s]", fontsize=18)
    plt.ylabel("Number of spikes in model", fontsize=18)
    plt.savefig(namestr + "_scale_histograms.eps", format="eps")
    plt.close()

    skew_hist_all = []
    skew_max_log10 = np.exp(skew_max)
    smin = min(skew_max_log10.flatten())
    smax = np.max(skew_max_log10.flatten())

    for s in np.transpose(skew_max_log10):
        h,bins = np.histogram(s, bins=nbins, range=[smin, smax])
        skew_hist_all.append(h)

    fig, ax = subplots()
    imshow(skew_hist_all, cmap=cm.hot)
    plt.axis([-0.5, -0.5+nbins, -0.5, np.min(np.shape(skew_max_log10))-1])

    ### get limits of x-axis
    start, end = ax.get_xlim()
    ### define tick positions
    tickpos = np.arange(start+0.5, end, 2)
    ### define tick labels
    ticklabels = np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/float(len(tickpos)))
    ### stupid way of defining significant digits
    ticklabels = [str(l)[:5] for l in ticklabels]
    xticks(tickpos, ticklabels)
    plt.xlabel(r"$\log_{10}{(\mathrm{skew})}$ [s]", fontsize=18)
    plt.ylabel("Number of spikes in model", fontsize=18)
    plt.savefig(namestr + "_skew_histograms.eps", format="eps")
    plt.close()


    return

def main():

    example_model()

    return


if __name__ == "__main__":

    main()