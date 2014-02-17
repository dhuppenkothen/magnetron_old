import numpy as np
import word
import burstmodel
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
from pylab import *

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


def main():

    example_model()

    return


if __name__ == "__main__":

    main()