import numpy as np
import word
import burstmodel

from pylab import *
rc("font", size=20, family="serif", serif="Computer Sans")
rc("text", usetex=True)

import matplotlib.cm as cm
import generaltools as gt




def example_model():

    times = np.arange(1000)/1000.0
    w = word.TwoExp(times)

    t0 = 0.5
    scale = 0.05
    amp = 1.0
    skew = 5.0

    word_counts = w.model(t0, scale, amp, skew)

    fig = plt.figure(figsize=(18,6))

    subplot(1,3,1)
    plot(times, word_counts, lw=2, color='black')
    xlabel(r"$\xi$")
    ylabel('Counts per bin in arbitrary units')
    title(r'single word, $t=0.5$, $\tau=0.05$, $A=1$, $s=5$', fontsize=20)


    counts = np.ones(len(times))
    b = burstmodel.BurstDict(times, counts, [word.TwoExp for w in range(3)])

    wordparams = [0.1, np.log(0.05), np.log(60.0), np.log(5.0), 0.4, np.log(0.01),
                  np.log(100.0), np.log(1.0), 0.7, np.log(0.04), np.log(50), -2, np.log(10.0)]


    model_counts = b.model_means(wordparams)

    subplot(1,3,2)

    plot(times, model_counts, lw=2, color='black')
    xlabel(r"Time $t$ [s]")
    ylabel('Counts per bin in arbitrary units')
    title(r'three words, $t=0.1,0.4,0.7$, $\tau=0.05, 0.01, 0.04$, $A=60, 100, 50$, $s=5,1,0.1$', fontsize=13)

    poisson_counts = np.array([np.random.poisson(c) for c in model_counts])

    subplot(1,3,3)
    plot(times, poisson_counts, lw=2, color='black')
    xlabel(r"Time $t$ [s]")
    ylabel('Counts per bin in arbitrary units')
    title(r'three words, /w Poisson, $t=0.1,0.4,0.7$, $\tau=0.05, 0.01, 0.04$, $A=60, 100, 50$, $s=5,1,0.1$', fontsize=13)

    savefig("example_words.png", format='png')
    close()

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
    legend()
    xlabel(r"$\log_{10}{(\mathrm{scale})}$ [s]", fontsize=18)
    ylabel(r"$N(\mathrm{bursts})$", fontsize=18)
    title("Distribution of rise times for the 85 brightest bursts")
    savefig(namestr + "_scale.eps", format="eps")
    close()

    ### DO same plot with skew parameters
    hist(np.log10(np.exp(skew_max[:,0])), bins=15, color='navy', histtype='stepfilled', alpha=0.7, label="1 word")
    hist(np.log10(np.exp(skew_max[:,4])), bins=15, color='darkred', histtype='stepfilled', alpha=0.7, label="5 words")
    hist(np.log10(np.exp(skew_max[:,9])), bins=15, color='green', histtype='stepfilled', alpha=0.7, label="10 words")
    xlabel(r"$\log_{10}{(\mathrm{skew})}$ [s]", fontsize=18)
    ylabel(r"$N(\mathrm{bursts})$", fontsize=18)
    legend()
    title("Distribution of skew parameters for the 85 brightest bursts")
    savefig(namestr + "_skew.eps", format="eps")
    close()

    ### Plot all scales as a function of number of spikes in one plot, just to be able to look at them
    ### all at once:
    figure()
    for l,m,u in zip(scale_cl, scale_max, scale_cu):
        errorbar(np.arange(10)+1, np.log10(np.exp(m)), yerr=[np.log10(np.exp(m-l)),
                                                                np.log10(np.exp(u-m))], fmt="--o", lw=2)
    xlabel("Number of spikes in model", fontsize=18)
    ylabel(r"$\log_{10}{(\mathrm{scale})}$ [s]", fontsize=18)
    title("Change of rise time with number of spikes in model, for 85 brightest bursts")
    savefig(namestr + "_scale_per_nspikes.eps", format='eps')
    close()


    ### Same for skew
    figure()
    for l,m,u in zip(skew_cl, skew_max, skew_cu):
        errorbar(np.arange(10)+1, np.log10(np.exp(m)), yerr=[np.log10(np.exp(m-l)),
                                                                np.log10(np.exp(u-m))], fmt="--o", lw=2)
    xlabel("Number of spikes in model", fontsize=18)
    ylabel(r"$\log_{10}{(\mathrm{skew})}$ [s]", fontsize=18)
    title("Change of skew with number of spikes in model, for 85 brightest bursts")
    savefig(namestr + "_skew_per_nspikes.eps", format='eps')
    close()



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
    axis([-0.5, -0.5+nbins, -0.5, np.min(np.shape(scale_max_log10))-1])

    ### get limits of x-axis
    start, end = ax.get_xlim()
    ### define tick positions
    tickpos = np.arange(start+0.5, end, 2)
    ### define tick labels
    ticklabels = np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/float(len(tickpos)))
    ### stupid way of defining significant digits
    ticklabels = [str(l)[:5] for l in ticklabels]
    xticks(tickpos, ticklabels)
    xlabel(r"$\log_{10}{(\mathrm{scale})}$ [s]", fontsize=18)
    ylabel("Number of spikes in model", fontsize=18)
    savefig(namestr + "_scale_histograms.eps", format="eps")
    close()

    skew_hist_all = []
    skew_max_log10 = np.log10(np.exp(skew_max))
    smin = min(skew_max_log10.flatten())
    smax = np.max(skew_max_log10.flatten())

    for s in np.transpose(skew_max_log10):
        h,bins = np.histogram(s, bins=nbins, range=[smin, smax])
        skew_hist_all.append(h)

    fig, ax = subplots()
    imshow(skew_hist_all, cmap=cm.hot)
    axis([-0.5, -0.5+nbins, -0.5, np.min(np.shape(skew_max_log10))-1])

    ### get limits of x-axis
    start, end = ax.get_xlim()
    ### define tick positions
    tickpos = np.arange(start+0.5, end, 2)
    ### define tick labels
    ticklabels = np.arange(bins[0], bins[-1], (bins[-1]-bins[0])/float(len(tickpos)))
    ### stupid way of defining significant digits
    ticklabels = [str(l)[:5] for l in ticklabels]
    xticks(tickpos, ticklabels)
    xlabel(r"$\log_{10}{(\mathrm{skew})}$ [s]", fontsize=18)
    ylabel("Number of spikes in model", fontsize=18)
    savefig(namestr + "_skew_histograms.eps", format="eps")
    close()


    return


def playing_around_with_amplitudes(filename, nwords = 10, namestr="allbursts"):

    allparas = gt.getpickle(filename)
    amp_max = np.array(allparas["amp_max"])
    t0_max = np.array(allparas["t0_max"])


    ### plot posterior maximum of amplitude parameter for the first spike versus posterior max of amplitude
    ### of all other spikes at least 10ms away
    amp0, amp_mean = [], []
    for i,(t,a) in enumerate(zip(t0_max, amp_max)):
        minind = np.argmin(t[nwords-1])
        a_temp = list(a[nwords-1])
        a0 = a_temp.pop(minind)
        t0_temp = list(t[nwords-1])
        t0 = t0_temp.pop(minind)
        a_otherspikes = [i for i,j in zip(a_temp, t0_temp) if j > t0+0.01]
        if not len(a_otherspikes) == 0 :
            amp_mean.append(np.mean(a_otherspikes))
            amp0.append(a0)
        else:
            continue

    fig = figure(figsize=(12, 9))
    scatter(amp0, amp_mean)
    xlabel("Amplitude of first spike in burst", fontsize=18)
    ylabel("Mean amplitude of other spikes in burst", fontsize=18)
    title("Amplitude of first spike versus mean amplitude of other spikes")
    savefig(namestr + "_firstspike_meanrest_k" + str(nwords) + ".eps", format='eps')
    close()

    ### Plot the amplitude
    amp_9 = [a[nwords-1] for a in amp_max]
    t0_9 = [a[nwords-1] for a in t0_max]
    pairs = [zip(t,a) for t,a in zip(t0_9, amp_9)]
    pairs_sorted = [sorted(p) for p in pairs]
    amp_max_sorted = [[a[1] for a in p] for p in pairs_sorted]


    ### Plot the amplitude for each spike in each burst for each position (as an integer number, not time)
    ### as an image
    ### This should help me see whether there are any overall trends in the amplitude
    fig = figure(figsize=(30,6))
    imshow(np.transpose(amp_max_sorted), cmap=cm.hot)
    colorbar()
    subplots_adjust(top=0.9, bottom=0.13, left=0.05, right=0.97, wspace=0.15, hspace=0.1)
    xlabel("Burst index", fontsize=20)
    ylabel("Spike position in burst", fontsize=20)
    title("Amplitude (colour) as a function of position in the burst (y-axis) for all bursts")
    savefig(namestr + "_amplitude_image_k" + str(nwords) + ".eps", format="eps")
    close()


    ### Compute mean and standard deviation for each spike position for all bursts
    amp_max_mean = np.mean(amp_max_sorted, axis=0)
    amp_max_std = np.std(amp_max_sorted, axis=0)

    ### plot mean amplitude versus spike position in the burst, with errors
    ### again, if there's an overall trend, I should see it here!
    fig = figure(figsize=(12,9))
    errorbar(np.arange(nwords)+1, amp_max_mean, yerr=amp_max_std, color='black', fmt="--o")
    xlabel("Location of spike relative to the other spikes")
    ylabel("Average amplitude over 85 bursts")
    title("Average amplitude for 85 bursts as a function of spike position in a burst")
    savefig(namestr + "_spike_amplitude_vs_location_k" + str(nwords) + ".eps", format="eps")
    close()

    ### For fun, plot the overall distribution of amplitudes:
    fig = figure(figsize=(12,9))
    hist(np.array(np.log10(amp_max_sorted)).flatten(), bins=30, color='navy', histtype='stepfilled')
    xlabel(r"$\log_{10}{(\mathrm{amplitude})}$ [s]", fontsize=18)
    ylabel("N(spikes)", fontsize=18)
    title("Distribution of amplitudes for all spikes in all bursts")
    savefig(namestr + "_amplitude_distribution_k" + str(nwords) + ".eps", format="eps")
    close()

    ### Plot distribution of waiting times between spikes
    t0_max_sorted = [[a[0] for a in p] for p in pairs_sorted]
    dt_all = [[t[i] - t[i-1] for i in np.arange(len(t[1:]))+1] for t in t0_max_sorted]
    fig = figure(figsize=(12,9))
    hist(np.array(np.log10(dt_all)).flatten(), bins=30, color="navy", histtype="stepfilled")
    xlabel(r"$\log_10{(\mathrm{waiting time})}$ [s]", fontsize=18)
    ylabel("N(spikes)", fontsize=18)
    title("Distribution of waiting times between adjacent spikes in all bursts")
    savefig(namestr + "_waitingtime_distribution_k" + str(nwords) + ".eps", format="eps")
    close()

    return



def main():

    example_model()

    return


if __name__ == "__main__":

    main()