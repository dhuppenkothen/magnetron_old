


import numpy as np

import word
import burstmodel

def word_tests():

    ''' test suite for Word class'''

    print('First test: unscaled TwoExp word')
    times = np.arange(2000.0)/1000.0 - 1.0
    event_time = 0.00
    scale = 1.0
    skew = 1.0


    w = TwoExp(times)
    y = w(event_time)

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(r'Raw word, $t_\mathrm{start} = 0$, scale $\sigma = 1.0$, skew $\alpha = 1.0$')
    plt.savefig('word_test1.png', format='png')
    plt.close()

    print('Second test: scaled TwoExp word')
    event_time = 0.5
    scale = 0.5
    skew = 3.0
    amp = 10.0
    print('times: ' + str(times))
    w = TwoExp(times)
    y = w([event_time, scale,amp, skew])

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(r'Scaled word, $t_\mathrm{start} = 0.5$, scale $\sigma = 0.5$, amplitude $A = 10$, skew $\alpha = 3.0$')
    plt.savefig('word_test2.png', format='png')
    plt.close()


    print('Third test: two combined words')

    event_time1 = 0.0
    event_time2 = 0.5
    scale1 = 0.1
    scale2 = 0.1
    amp1 = 5.0
    amp2 = 10.0
    skew1 = 1.0
    skew2 = 5.0

    w = CombinedWords(times, [TwoExp, TwoExp])
    y = w([event_time1, scale1, amp1, skew1, event_time2, scale2, amp2, skew2])

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(r'2 Words, $t_{\mathrm{start}} = 0.0$ and $0.5$, scale $\sigma_{1,2} = 0.1$, $A_1 = 5$, $A_2 = 10$, skew $\alpha_1 = 1.0$, $\alpha_2 = 3$ ')
    plt.savefig('word_test3.png', format='png')
    plt.close()


    return



def burst_tests():

    ## 1-second time array
    times = np.arange(1000.0)/1000.0
    ## dummy variable: I don't currently have counts
    counts = np.ones(len(times))


    print('Test 1: Just one word')

    event_time = 0.5
    scale = 0.1
    skew = 3
    amp = 5.0
    bkg = 3.0

    ## initialise burst object
    b = burstmodel.BurstModel(times, counts, word.TwoExp)

    print('time array: ' + str(b.times))
    print('type of time array: ' + str(type(b.times)))
    print('counts array: ' + str(b.counts))
    print('type of counts array: ' + str(type(b.counts)))

    print('model function (going to test in detail below: ' + str(self.model))

    print('Delta (should be 0.001): ' + str(b.Delta))
    print('nbins_data (should be 1000): ' + str(b.nbins_data))

    print('Now testing whether model creation works ... ')
    theta = [event_time, scale, amp, skew, bkg]
    y = b.model(theta)

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(r'Model test, $t_\mathrm{start} = 0.5$, scale $\sigma = 0.1$, amplitude $A = 50$, skew $\alpha = 3$, $\mathrm{bkg} = 3')
    plt.savefig('burst_test1.png', format='png')
    plt.close()

    






