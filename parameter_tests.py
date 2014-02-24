

### Tests for new parameter class
import matplotlib.pyplot as plt
from pylab import *
# Foreman-Mackey's taste in figures
#rc("font", size=20, family="serif", serif="Computer Sans")
#rc("text", usetex=True)

import numpy as np
import argparse

import parameters
import word
import burstmodel

def two_exp_parameter_tests():
    """
    This function tests the new classes in parameters.py: TwoExpParameters and TwoExpCombined.
    """

    ### single word parameters
    t0 = 0.1
    log_scale = -4.0
    log_skew = 2.0
    log_amp = 5.0

    ### parameter object
    theta = parameters.TwoExpParameters(t0=t0, scale=log_scale, amp=log_amp, skew=log_skew, log=True)

    print("Does TwoExpParameters store parameters correctly?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.t0))
    print("log(scale) log_scale = -4 : \t ... \t " + str(theta.log_scale))
    print("log(amplitude) log_amp = 5 : \t ... \t " + str(theta.log_amp))
    print("log(skew) log_skew = 2 : \t ... \t " + str(theta.log_skew) + "\n")

    print("Does TwoExpParameters convert log-parameters correctly?")
    print("scale = 0.018 : \t ... \t " + str(theta.scale))
    print("amp = 148.4 : \t ... \t " + str(theta.amp))
    print("skew = 7.38 : \t ... \t " + str(theta.skew) + "\n")

    ### Combined parameter object with two words, scale and skew individually defined
    t0_2 = 0.5
    log_scale_2 = -6.0
    log_skew_2 = -1.0
    log_amp_2 = 3.0

    theta = parameters.TwoExpCombined([t0, log_scale, log_amp, log_skew, t0_2, log_scale_2, log_amp_2, log_skew_2], 2, log=True)

    print("Does TwoExpCombined store parameters correctly?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.all[0].t0))
    print("log(scale) log_scale = -4 : \t ... \t " + str(theta.all[0].log_scale))
    print("log(amp) log_amp = 5 : \t ... \t " + str(theta.all[0].log_amp))
    print("log(skew) log_skew = 2 : \t ... \t " + str(theta.all[0].log_skew) + "\n")
    print("peak time t0_2 = 0.5 : \t ... \t " + str(theta.all[1].t0))
    print("log(scale) log_scale_2 = -6 : \t ... \t " + str(theta.all[1].log_scale))
    print("log(amp) log_amp_2 = 3 : \t ... \t " + str(theta.all[1].log_amp))
    print("log(skew) log_skew_2 = -1 : \t ... \t " + str(theta.all[1].log_skew) + "\n")


    ### test with scale_locked only:
    theta = parameters.TwoExpCombined([t0, log_amp, log_skew, t0_2, log_amp_2, log_skew_2, log_scale], 2, scale_locked=True, log=True)
    print("Does TwoExpCombined store parameters correctly when scale_locked=True ?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.all[0].t0))
    print("log(scale) log_scale = -4 : \t ... \t " + str(theta.all[0].log_scale))
    print("log(amp) log_amp = 5 : \t ... \t " + str(theta.all[0].log_amp))
    print("log(skew) log_skew = 2 : \t ... \t " + str(theta.all[0].log_skew) + "\n")
    print("peak time t0_2 = 0.5 : \t ... \t " + str(theta.all[1].t0))
    print("log(scale) log_scale_2 = -4 : \t ... \t " + str(theta.all[1].log_scale))
    print("log(amp) log_amp_2 = 3 : \t ... \t " + str(theta.all[1].log_amp))
    print("log(skew) log_skew_2 = -1 : \t ... \t " + str(theta.all[1].log_skew) + "\n")

    ### test with skew_locked only:
    theta = parameters.TwoExpCombined([t0, log_scale, log_amp, t0_2, log_scale_2, log_amp_2, log_skew], 2, skew_locked=True, log=True)
    print("Does TwoExpCombined store parameters correctly when skew_locked=True ?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.all[0].t0))
    print("log(scale) log_scale = -4 : \t ... \t " + str(theta.all[0].log_scale))
    print("log(amp) log_amp = 5 : \t ... \t " + str(theta.all[0].log_amp))
    print("log(skew) log_skew = 2 : \t ... \t " + str(theta.all[0].log_skew) + "\n")
    print("peak time t0_2 = 0.5 : \t ... \t " + str(theta.all[1].t0))
    print("log(scale) log_scale_2 = -6 : \t ... \t " + str(theta.all[1].log_scale))
    print("log(amp) log_amp_2 = 3 : \t ... \t " + str(theta.all[1].log_amp))
    print("log(skew) log_skew_2 = 2 : \t ... \t " + str(theta.all[1].log_skew) + "\n")

    ### test with both scale_locked and skew_locked:
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_scale, log_skew], 2, scale_locked=True, skew_locked=True, log=True)
    print("Does TwoExpCombined store parameters correctly when scale_locked=True and skew_locked = True ?")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.all[0].t0))
    print("log(scale) log_scale = -4 : \t ... \t " + str(theta.all[0].log_scale))
    print("log(amp) log_amp = 5 : \t ... \t " + str(theta.all[0].log_amp))
    print("log(skew) log_skew = 2 : \t ... \t " + str(theta.all[0].log_skew) + "\n")
    print("peak time t0_2 = 0.5 : \t ... \t " + str(theta.all[1].t0))
    print("log(scale) log_scale_2 = -4 : \t ... \t " + str(theta.all[1].log_scale))
    print("log(amp) log_amp_2 = 3 : \t ... \t " + str(theta.all[1].log_amp))
    print("log(skew) log_skew_2 = 2 : \t ... \t " + str(theta.all[1].log_skew) + "\n")

    ### test with both scale_locked and skew_locked:
    log_bkg = 2.0
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_scale, log_skew, log_bkg], 2, scale_locked=True, skew_locked=True, log=True, bkg = True)
    print("Does TwoExpCombined store parameters correctly when including a background parameter??")
    print("peak time t0 = 0.1 : \t ... \t " + str(theta.all[0].t0))
    print("log(scale) log_scale = -4 : \t ... \t " + str(theta.all[0].log_scale))
    print("log(amp) log_amp = 5 : \t ... \t " + str(theta.all[0].log_amp))
    print("log(skew) log_skew = 2 : \t ... \t " + str(theta.all[0].log_skew) + "\n")
    print("peak time t0_2 = 0.5 : \t ... \t " + str(theta.all[1].t0))
    print("log(scale) log_scale_2 = -4 : \t ... \t " + str(theta.all[1].log_scale))
    print("log(amp) log_amp_2 = 3 : \t ... \t " + str(theta.all[1].log_amp))
    print("log(skew) log_skew_2 = 2 : \t ... \t " + str(theta.all[1].log_skew) + "\n")
    print("log(background parameter): log_bkg = 2: \t ... \t " + str(theta.log_bkg))
    print("background parameter: bkg = 7.38: \t ... \t " + str(theta.bkg) + "\n")

    return


def word_tests():
    """
     Tests for classes TwoExp and CombinedWords with the new parameter classes.
    """

    times = np.arange(1000)/1000.0

    ### single word parameters
    t0 = 0.1
    log_scale = -4.0
    log_skew = 2.0
    log_amp = 5.0

    ### parameter object
    theta = parameters.TwoExpParameters(t0=t0, scale=log_scale, amp=log_amp, skew=log_skew, log=True)

    ### word object
    w = word.TwoExp(times)
    model_counts = w(theta)

    plt.figure()
    plt.plot(times, model_counts)
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Counts [cts/bin]", fontsize=18)
    plt.title(r"Word test 1: $t_0 = 0.1$, $\log{(scale)} = -4$, $\log{(amp)} = 5$, $\log{(skew)} = 2$")
    plt.savefig("parclass_word_test1.png", format="png")
    plt.close()

    print("Testing whether the prior works ...")
    print("This should be 0, all parameters in range: " + str(w.logprior(theta)))

    theta = parameters.TwoExpParameters(t0=-0.1, scale=log_scale, amp=log_amp, skew=log_skew, log=True)
    print("This should be inf, t0 < times[0]: " + str(w.logprior(theta)))

    theta = parameters.TwoExpParameters(t0=2.0, scale=log_scale, amp=log_amp, skew=log_skew, log=True)
    print("This should be inf, t0 > times[-1]: " + str(w.logprior(theta)))

    theta = parameters.TwoExpParameters(t0=t0, scale=np.log(w.Delta/10.0), amp=log_amp, skew=log_skew, log=True)
    print("This should be inf, log_scale < log(Delta): " + str(w.logprior(theta)))

    theta = parameters.TwoExpParameters(t0=t0, scale=np.log(w.T*2.0), amp=log_amp, skew=log_skew, log=True)
    print("This should be inf, log_scale > log(T): " + str(w.logprior(theta)))

    theta = parameters.TwoExpParameters(t0=t0, scale=log_scale, amp=-11, skew=log_skew, log=True)
    print("This should be inf, log_amp < -10: " + str(w.logprior(theta)))

    theta = parameters.TwoExpParameters(t0=t0, scale=log_scale, amp=np.log(3.5e5*2.0), skew=log_skew, log=True)
    print("This should be inf, log_amp > log(saturation_countrate = 3.5e5 cts/s): " + str(w.logprior(theta)))

    theta = parameters.TwoExpParameters(t0=t0, scale=log_scale, amp=log_amp, skew=-2.0, log=True)
    print("This should be inf, log_skew < -1.5: " + str(w.logprior(theta)))

    theta = parameters.TwoExpParameters(t0=t0, scale=log_scale, amp=log_amp, skew=4.0, log=True)
    print("This should be inf, log_skew > log(3): " + str(w.logprior(theta)) + "\n")


    print("Now testing with more than one word ...")

    ### testing Combined Words
    w = word.CombinedWords(times, [word.TwoExp, word.TwoExp])

    t0_2 = 0.5
    log_scale_2 = -6.0
    log_skew_2 = -1.0
    log_amp_2 = 4.0


    ### Test 2: independent parameters for both words
    theta = parameters.TwoExpCombined([t0, log_scale, log_amp, log_skew, t0_2, log_scale_2, log_amp_2, log_skew_2], 2, log=True)

    model_counts = w(theta)

    plt.figure()
    plt.plot(times, model_counts)
    plt.axis([times[0], times[-1], 0, max(model_counts)+10])
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Counts [cts/bin]", fontsize=18)
    plt.title(r"Word test 2: $t_0 = 0.1,0.5$, $\log{(scale)} = -4,-6$, $\log{(amp)} = 5,4$, $\log{(skew)} = 2,-1$")
    plt.savefig("parclass_word_test2.png", format="png")
    plt.close()


    ### Test 3: scale the same for both words
    theta = parameters.TwoExpCombined([t0, log_amp, log_skew, t0_2, log_amp_2, log_skew_2, log_scale], 2, scale_locked=True, log=True)

    model_counts = w(theta)

    plt.figure()
    plt.plot(times, model_counts)
    plt.axis([times[0], times[-1], 0, max(model_counts)+10])

    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Counts [cts/bin]", fontsize=18)
    plt.title(r"Word test 3, same scale: $t_0 = 0.1,0.5$, $\log{(scale)} = -4$, $\log{(amp)} = 5,4$, "
              r"$\log{(skew)} = 2,-1$", fontsize=12)
    plt.savefig("parclass_word_test3.png", format="png")
    plt.close()

    ### Test 4: skew the same for both words
    theta = parameters.TwoExpCombined([t0, log_scale, log_amp, t0_2, log_scale_2, log_amp_2, log_skew], 2, skew_locked=True, log=True)

    model_counts = w(theta)

    plt.figure()
    plt.plot(times, model_counts)
    plt.axis([times[0], times[-1], 0, max(model_counts)+10])
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Counts [cts/bin]", fontsize=18)
    plt.title(r"Word test 3, same skew: $t_0 = 0.1,0.5$, $\log{(scale)} = -4,-6$, $\log{(amp)} = 5,4$, "
              r"$\log{(skew)} = 2$", fontsize=12)
    plt.savefig("parclass_word_test4.png", format="png")
    plt.close()

    ### Test 5: Both scale and skew the same for both words
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_scale, log_skew], 2, scale_locked=True, skew_locked=True, log=True)

    model_counts = w(theta)

    plt.figure()
    plt.plot(times, model_counts)
    plt.axis([times[0], times[-1], 0, max(model_counts)+10])
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Counts [cts/bin]", fontsize=18)
    plt.title(r"Word test 5, same scale + skew: $t_0 = 0.1,0.5$, $\log{(scale)} = -4$, $\log{(amp)} = 5,4$, "
              r"$\log{(skew)} = 2$", fontsize=12)
    plt.savefig("parclass_word_test5.png", format="png")
    plt.close()


    ### Test 6: Adding a background parameter
    log_bkg = 3.0
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_scale, log_skew, log_bkg], 2, scale_locked=True, skew_locked=True, log=True, bkg=True)

    model_counts = w(theta)

    plt.figure()
    plt.plot(times, model_counts)
    plt.axis([times[0], times[-1], 0, max(model_counts)+10])
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Counts [cts/bin]", fontsize=18)
    plt.title(r"Word test 5: $t_0 = 0.1,0.5$, $\log{(scale)} = -4$, $\log{(amp)} = 5,4$, "
              r"$\log{(skew)} = 2$, $\log{(bkg)} = 3$", fontsize=12)
    plt.savefig("parclass_word_test6.png", format="png")
    plt.close()

    return


def burstdict_tests():
    """
    Testing BurstDict class in burstmodel.py for the new parameter class
    """

    times = np.arange(1000)/1000.0
    counts = np.ones(len(times))*0.005

    ### single word parameters
    t0 = 0.1
    log_scale = -4.0
    log_skew = 2.0
    log_amp = 3.0

    log_bkg = 2.0

    print("Testing background only ...")
    bd = burstmodel.BurstDict(times, counts, [])

    ## theta needs to be a parameter object:
    theta = parameters.TwoExpCombined([log_bkg], 0, bkg=True, log=True)
    model_counts = bd.model_means(theta)

    plt.figure()
    plt.plot(times, bd.countrate, lw=2, color='black', label="Simulated light curve")
    plt.plot(times, model_counts, lw=2, color='red', label='Model light curve')
    plt.axis([times[0], times[-1], 0, np.max([np.max(bd.countrate), np.max(model_counts)])+10])
    plt.legend()
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Count rate [cts/s]", fontsize=18)
    plt.title(r"Background only, $\log{(bkg)} = 2$", fontsize=12)
    plt.savefig("parclass_burstdict_test1.png", format="png")
    plt.close()

    print("... saved in parclass_burstdict_test1.png. \n")

    print("One word model: ")
    bd = burstmodel.BurstDict(times, counts, word.TwoExp)

    ## theta needs to be a parameter object:
    theta = parameters.TwoExpParameters(t0=t0, scale=log_scale, skew=log_skew, amp=log_amp, log=True)
    model_counts = bd.model_means(theta)

    plt.figure()
    plt.plot(times, bd.countrate, lw=2, color='black', label="Simulated light curve")
    plt.plot(times, model_counts, lw=2, color='red', label='Model light curve')
    plt.axis([times[0], times[-1], 0, np.max([np.max(bd.countrate), np.max(model_counts)])+10])
    plt.legend()
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Count rate [cts/s]", fontsize=18)
    plt.title(r"One word: $t_0 = 0.1$, $\log{(scale)} = -4$, $\log{(amp)} = 3$, $\log{(skew)} = 2$", fontsize=12)
    plt.savefig("parclass_burstdict_test2.png", format="png")
    plt.close()

    print("... saved in parclass_burstdict_test2.png \n")

    print("Two words next:")
    bd = burstmodel.BurstDict(times, counts, [word.TwoExp, word.TwoExp])

    t0_2 = 0.5
    log_amp_2 = 4.0
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_scale, log_skew, log_bkg], 2, log=True,
                                      scale_locked=True, skew_locked=True, bkg=True)

    model_counts = bd.model_means(theta)

    plt.figure()
    plt.plot(times, bd.countrate, lw=2, color='black', label="Simulated light curve")
    plt.plot(times, model_counts, lw=2, color='red', label='Model light curve')
    plt.axis([times[0], times[-1], 0, np.max([np.max(bd.countrate), np.max(model_counts)])+10])
    plt.legend()
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Count rate [cts/s]", fontsize=18)
    plt.title(r"Two words + bkg: $t_0 = 0.1,0.5$, $\log{(scale)} = -4$, $\log{(amp)} = 3,4$, $\log{(skew)} = 2$,"
              r"$log{(bkg)} = 2.0$", fontsize=12)
    plt.savefig("parclass_burstdict_test3.png", format="png")
    plt.close()

    print("... saved in parclass_burstdict_test3.png. \n")

    print("Testing method plot_model in class BurstDict ...")
    bd.plot_model(theta, plotname="parclass_burstdict_test4")

    print("Adding dummy posterior maximum to the plot ...")

    postmax = parameters.TwoExpCombined([t0+0.03, np.log(bd.countrate[0])+1.8, t0_2+0.05, np.log(bd.countrate[0])+1.5, log_scale-1, log_skew-0.4, log_bkg], 2, log=True,
                                      scale_locked=True, skew_locked=True, bkg=True)

    bd.plot_model(theta, postmax, "parclass_burstdict_test5")

    print("Test poissonify function ...")

    bd = burstmodel.BurstDict(times, counts*1000.0, [word.TwoExp, word.TwoExp])
    theta = parameters.TwoExpCombined([t0, np.log(bd.countrate[0])+2, t0_2, np.log(bd.countrate[0])+1, log_scale, log_skew, log_bkg], 2, log=True,
                                      scale_locked=True, skew_locked=True, bkg=True)

    poisson_countrate = bd.poissonify(theta)

    plt.figure()
    plt.plot(times, bd.countrate, lw=2, color='black', label="Simulated light curve")
    plt.plot(times, poisson_countrate, lw=1, color='red', label='Model light curve, poissonified')
    plt.axis([times[0], times[-1], 0, np.max([np.max(bd.countrate), np.max(poisson_countrate)])+10])
    plt.legend()
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Count rate [cts/s]", fontsize=18)
    plt.title(r"Two words + bkg, poissonified!", fontsize=12)
    plt.savefig("parclass_burstdict_test6.png", format="png")
    plt.close()

    return


def word_posterior_tests():

    """
    This function tests whether WordPosterior in burstmodel.py works properly.
    """


    times = np.arange(1000)/1000.0

    counts = np.ones(len(times))*5.0

    ### First test: Flat light curve
    bd = burstmodel.BurstDict(times, counts, [])

    lpost = burstmodel.WordPosterior(times, counts, bd, scale_locked=False, skew_locked=False, log=True, bkg=True)

    log_bkg = 2.0

    print("Log-Prior probability: " + str(lpost.logprior([log_bkg])))
    print("Log-Likelihood: " + str(lpost.loglike([log_bkg])))
    print("Log-Posterior probability: " + str(lpost([log_bkg])) + "\n")

    print("Make parameter exactly equal count rate:")
    log_bkg = np.log(np.max(bd.countrate))
    print("Log-Prior probability: " + str(lpost.logprior([log_bkg])))
    print("Log-Likelihood: " + str(lpost.loglike([log_bkg])))
    print("Log-Posterior probability: " + str(lpost([log_bkg])) + "\n")


    bd = burstmodel.BurstDict(times, counts*1000.0, [word.TwoExp, word.TwoExp])

    lpost = burstmodel.WordPosterior(times, counts, bd, scale_locked=True, skew_locked=True, log=True, bkg=True)


    t0 = 0.1
    log_scale = -4.0
    log_skew = 2.0
    log_amp = 3.0

    log_bkg = 2.0


    theta_list = [t0, log_amp, t0+0.3, log_amp+1, log_scale, log_skew, log_bkg]


    print("Log-Prior probability: " + str(lpost.logprior(theta_list)))
    print("Log-Likelihood: " + str(lpost.loglike(theta_list)))
    print("Log-Posterior probability: " + str(lpost(theta_list)) + "\n")

    print("Including a parameter outside the ranges allowed in the prior:")

    theta_list[0] = -0.1
    print("Log-Prior probability: " + str(lpost.logprior(theta_list)))
    print("Log-Likelihood: " + str(lpost.loglike(theta_list)))
    print("Log-Posterior probability: " + str(lpost(theta_list)) + "\n")

    theta_list[0] = t0
    theta_list[1] = 11
    theta_list[3] = 9

    print("Testing on a Poisson light curve of two spikes ...")
    ### Make a Poissonified light curve of two spikes
    theta = parameters.TwoExpCombined(theta_list, 2, log=True, scale_locked=True, skew_locked=True, bkg=True)

    poisson_countrate = bd.poissonify(theta)
    delta = times[1] - times[0]

    bd = burstmodel.BurstDict(times, poisson_countrate*delta, [word.TwoExp, word.TwoExp])

    lpost = burstmodel.WordPosterior(times, poisson_countrate*delta, bd, scale_locked=True,
                                     skew_locked=True, log=True, bkg=True)

    ### Compute probabilities for a vastly different test parameter set:
    print("First test: test parameters vastly different from actual light curve parameters")
    theta_test = [0.2, 11, 0.8, 10, -1, -1, 5]
    print("Log-Prior probability: " + str(lpost.logprior(theta_test)))
    print("Log-Likelihood: " + str(lpost.loglike(theta_test)))
    print("Log-Posterior probability: " + str(lpost(theta_test)) + "\n")

    theta =  parameters.TwoExpCombined(theta_list, 2, log=True, scale_locked=True, skew_locked=True, bkg=True)
    postmax = parameters.TwoExpCombined(theta_test, 2, log=True, scale_locked=True, skew_locked=True, bkg=True)
    bd.plot_model(theta, postmax=postmax, plotname="parclass_lpost_test1")
    print("Saved corresponding light curve in parclass_lpost_test1_lc.png \n")

    print("Second test: Test parameters same as input parameters")
    print("Log-Prior probability: " + str(lpost.logprior(theta_list)))
    print("Log-Likelihood: " + str(lpost.loglike(theta_list)))
    print("Log-Posterior probability: " + str(lpost(theta_list)) + "\n")

    postmax = parameters.TwoExpCombined(theta_list, 2, log=True, scale_locked=True, skew_locked=True, bkg=True)
    bd.plot_model(theta, postmax=postmax, plotname="parclass_lpost_test2")
    print("Saved corresponding light curve in parclass_lpost_test2_lc.png \n")

    return

def burstmodel_tests(long_run=True):
    """
    This model tests whether class BurstModel in burstmodel.py works properly
    parameter long_run sets whether a long MCMC run is performed. Set False for quick and dirty tests.
    """

    ### Making some fake data ...
    times = np.arange(1000)/1000.0

    counts = np.ones(len(times))*5.0

    t0 = 0.1
    t0_2 = 0.4
    log_scale = -4.0
    log_skew = 2.0
    log_amp = 9.0
    log_amp_2 = 11.0

    log_bkg = 5.0


    theta_list = [t0, log_amp, t0_2, log_amp_2, log_scale, log_skew, log_bkg]

    ### Make a Poissonified light curve of two spikes
    bd = burstmodel.BurstDict(times, counts, [word.TwoExp, word.TwoExp])
    theta = parameters.TwoExpCombined(theta_list, 2, log=True, scale_locked=True, skew_locked=True, bkg=True)

    poisson_countrate = bd.poissonify(theta)
    delta = times[1] - times[0]
    poisson_counts = poisson_countrate*delta

    bm = burstmodel.BurstModel(times, poisson_counts)

    print("First test: Does emcee run at all?")
    bd = burstmodel.BurstDict(times, counts, [word.TwoExp, word.TwoExp])
    initial_theta = [0.05, 8, 0.6, 12, -5, 1.5, 4]
    sampler = bm.mcmc(bd, initial_theta, nwalker=50, niter=50, burnin=50, scale_locked=True, skew_locked=True,
                      log=True, bkg=True, plot=True, plotname="parclass_burstmodel_test1")


    ### Do one without skew being the same:
    initial_theta = [0.05, 8, 2, 0.6, 12, 1, -5, 4]
    sampler = bm.mcmc(bd, initial_theta, nwalker=50, niter=50, burnin=50, scale_locked=True, skew_locked=False,
                      log=True, bkg=True, plot=True, plotname="parclass_burstmodel_test2")


    ### Do one without skew or scale being the same:
    initial_theta = [0.05, -6, 8, 2, 0.6, -4, 12, 1, 4]
    sampler = bm.mcmc(bd, initial_theta, nwalker=50, niter=50, burnin=50, scale_locked=False, skew_locked=False,
                      log=True, bkg=True, plot=True, plotname="parclass_burstmodel_test3")


    ### Test posterior maximum finding methods:
    quants, postmax = bm.find_postmax(sampler)

    for l,p,u in zip(quants["lower ci"], postmax, quants["upper ci"]):
        print("lower quantile: \t" + str(l) + " \t, postmax: \t" + str(p) + "\t, upper quantile: \t" + str(u))



    return





def main():

    if clargs.all:
        two_exp_parameter_tests()
        word_tests()
        burstdict_tests()
        word_posterior_tests()
        return

    if clargs.par_switch:
        two_exp_parameter_tests()
        return

    if clargs.word_switch:
        word_tests()
        return

    if clargs.bd_switch:
        burstdict_tests()
        return

    if clargs.post_switch:
        word_posterior_tests()
        return

    if clargs.model_switch:
        burstmodel_tests(clargs.long_run)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Various tests for the classes defined in parameters.py, word.py and burstmodel.py")

    parser.add_argument("-p", "--parameters", action="store_true", required=False, dest="par_switch",
                        help="Run parameter class tests")
    parser.add_argument("-w", "--word", action="store_true", required=False, dest="word_switch",
                        help="Run word class tests")
    parser.add_argument("-d", "--burstdict", action="store_true", required=False, dest="bd_switch",
                        help="Run burstdict class tests")
    parser.add_argument("-a", "--all", action="store_true", required=False, dest="all",
                        help="Run all tests at once!")
    parser.add_argument("--post", action="store_true", required=False, dest="post_switch",
                        help="Run tests on class WordPosterior with new parameter implementation")

    parser.add_argument("-m", "--model", action="store_true", required=False, dest="model_switch",
                        help="Run tests on class BurstModel with new parameter implementation")
    parser.add_argument("-l", "--longrun", action="store_true", required=False, dest="long_run",
                        help="When running BurstModel tests, do you want to perform a long MCMC run?")

    clargs = parser.parse_args()
    print("clargs.longrun: " + str(clargs.long_run))

    main()


