

### Tests for new parameter class
import matplotlib.pyplot as plt
from pylab import *
# Foreman-Mackey's taste in figures
#rc("font", size=20, family="serif", serif="Computer Sans")
#rc("text", usetex=True)

import numpy as np

import parameters
import word

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
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_skew, log_scale], 2, scale_locked=True, skew_locked=True, log=True)
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
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_skew, log_scale, log_bkg], 2, scale_locked=True, skew_locked=True, log=True, bkg = True)
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

    fig = plt.figure()
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

    fig = plt.figure()
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

    fig = plt.figure()
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

    fig = plt.figure()
    plt.plot(times, model_counts)
    plt.axis([times[0], times[-1], 0, max(model_counts)+10])
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Counts [cts/bin]", fontsize=18)
    plt.title(r"Word test 3, same skew: $t_0 = 0.1,0.5$, $\log{(scale)} = -4,-6$, $\log{(amp)} = 5,4$, "
              r"$\log{(skew)} = 2$", fontsize=12)
    plt.savefig("parclass_word_test4.png", format="png")
    plt.close()

    ### Test 5: Both scale and skew the same for both words
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_skew, log_scale], 2, scale_locked=True, skew_locked=True, log=True)

    model_counts = w(theta)

    fig = plt.figure()
    plt.plot(times, model_counts)
    plt.axis([times[0], times[-1], 0, max(model_counts)+10])
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Counts [cts/bin]", fontsize=18)
    plt.title(r"Word test 5, same scale + skew: $t_0 = 0.1,0.5$, $\log{(scale)} = -4$, $\log{(amp)} = 5,4$, "
              r"$\log{(skew)} = 2,-1$", fontsize=12)
    plt.savefig("parclass_word_test5.png", format="png")
    plt.close()


    ### Test 6: Adding a background parameter
    log_bkg = 3.0
    theta = parameters.TwoExpCombined([t0, log_amp, t0_2, log_amp_2, log_skew, log_scale, log_bkg], 2, scale_locked=True, skew_locked=True, log=True, bkg=True)

    model_counts = w(theta)

    fig = plt.figure()
    plt.plot(times, model_counts)
    plt.axis([times[0], times[-1], 0, max(model_counts)+10])
    plt.xlabel("Time [s]", fontsize=18)
    plt.ylabel("Counts [cts/bin]", fontsize=18)
    plt.title(r"Word test 5: $t_0 = 0.1,0.5$, $\log{(scale)} = -4$, $\log{(amp)} = 5,4$, "
              r"$\log{(skew)} = 2,-1$, $\log{(bkg)} = 3$", fontsize=12)
    plt.savefig("parclass_word_test6.png", format="png")
    plt.close()

    return


def main():

    two_exp_parameter_tests()
    word_tests()

    return

if __name__ == "__main__":

    main()

