import matplotlib.pyplot as plt
from pylab import *

import numpy as np

import word
import burstmodel

### make a test data set: 3 spikes with varying positions,
### widths, skews and amplitudes
def test_data():
    times = np.arange(1000.0) / 1000.0
    counts = np.ones(len(times))

    event_time1 = 0.2
    scale1 = np.log(0.05)
    amp1 = np.log(50.0)
    skew1 = np.log(5)

    event_time2 = 0.4
    scale2 = np.log(0.01)
    amp2 = np.log(100.0)
    skew2 = np.log(1.0)

    event_time3 = 0.7
    scale3 = np.log(0.005)
    amp3 = np.log(20.0)
    skew3 = np.log(10)

    bkg = np.log(10)

    theta = [event_time1, scale1, amp1, skew1, event_time2, scale2, amp2, skew2, \
             event_time3, scale3, amp3, skew3, bkg]

    b = burstmodel.BurstDict(times, counts, [word.TwoExp, word.TwoExp, word.TwoExp])
    theta_packed = b.wordobject._pack(theta)
    theta_exp = b.wordobject._exp(theta_packed)
    y = b.wordmodel(times, theta_exp)

    counts = np.array([np.random.poisson(c) for c in y])
    b = burstmodel.BurstDict(times, counts, [word.TwoExp, word.TwoExp, word.TwoExp])

    return b, theta


def word_tests():
    ''' test suite for Word class'''

    print('First test: unscaled TwoExp word')
    times = np.arange(2000.0) / 1000.0 - 1.0
    event_time = 0.00
    scale = 1.0
    skew = 1.0

    w = word.TwoExp(times)
    y = w(event_time)

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(r'Raw word, $t_\mathrm{start} = 0$, scale $\sigma = 1.0$, skew $\alpha = 1.0$')
    plt.savefig('word_test1.png', format='png')
    plt.close()

    print(' ... saved in word_test1.png. \n')

    print('Second test: scaled TwoExp word')
    event_time = 0.5
    scale = 0.5
    skew = 3.0
    amp = 10.0
    print('times: ' + str(times))
    w = word.TwoExp(times)
    y = w([event_time, scale, amp, skew])

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(r'Scaled word, $t_\mathrm{start} = 0.5$, scale $\sigma = 0.5$, amplitude $A = 10$, skew $\alpha = 3.0$')
    plt.savefig('word_test2.png', format='png')
    plt.close()

    print(' ... saved in word_test2.png. \n')

    print('Testing logprior for single word: ')

    test_theta = [0.1, 0.1, 5.0, 4.0]
    print('test theta: ' + str(test_theta) + '; This should work!')
    print('logprior: ' + str(w.logprior(test_theta)) + '\n')

    test_theta = [1.1, 0.1, 5.0, 4.0]
    print('test theta: ' + str(test_theta) + '; This shouldnt work, event_time out of bounds!')
    print('logprior: ' + str(w.logprior(test_theta)) + '\n')

    test_theta = [0.5, 2.1, 1.0, 4.0]
    print('w.T: ' + str(w.T))
    print('test theta: ' + str(test_theta) + '; This shouldnt work, scale out of bounds!')
    print('logprior: ' + str(w.logprior(test_theta)) + '\n')

    test_theta = [0.5, 0.1, 1.0e-11, 4.0]
    print('test theta: ' + str(test_theta) + '; This shouldnt work, amplitude out of bounds!')
    print('logprior: ' + str(w.logprior(test_theta)) + '\n')

    test_theta = [0.5, 0.1, 1.0, np.exp(4.0)]
    print('test theta: ' + str(test_theta) + '; This shouldnt work, skew out of bounds!')
    print('logprior: ' + str(w.logprior(test_theta)) + '\n')

    print('Third test: two combined words')

    event_time1 = 0.0
    event_time2 = 0.5
    scale1 = 0.1
    scale2 = 0.1
    amp1 = 5.0
    amp2 = 10.0
    skew1 = 1.0
    skew2 = 5.0

    w = word.CombinedWords(times, [word.TwoExp, word.TwoExp])
    theta = [event_time1, scale1, amp1, skew1, event_time2, scale2, amp2, skew2]
    theta_packed = w._pack(theta)
    y = w(theta_packed)

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(
        r'2 Words, $t_{\mathrm{start}} = 0.0$ and $0.5$, scale $\sigma_{1,2} = 0.1$, $A_1 = 5$, $A_2 = 10$, skew $\alpha_1 = 1.0$, $\alpha_2 = 3$ ')
    plt.savefig('word_test3.png', format='png')
    plt.close()

    print(' ... saved in word_test3.png. \n')

    return


def burst_tests():
    ## 1-second time array
    times = np.arange(1000.0) / 1000.0
    ## dummy variable: I don't currently have counts
    counts = np.ones(len(times))

    print('Test 1: Just one word')

    event_time = 0.5
    scale = 0.1
    skew = 3
    amp = 5.0
    bkg = 3.0

    ## initialise burst object
    b = burstmodel.BurstDict(times, counts, word.TwoExp)

    print('time array: ' + str(b.times))
    print('type of time array: ' + str(type(b.times)))
    print('counts array: ' + str(b.counts))
    print('type of counts array: ' + str(type(b.counts)))

    print('model function (going to test in detail below: ' + str(b.wordmodel))

    print('Delta (should be 0.001): ' + str(b.Delta))
    print('nbins_data (should be 1000): ' + str(b.nbins_data))

    print('Now testing whether model creation works ... ')
    theta = [event_time, scale, amp, skew, bkg]
    y = b.wordmodel(times, theta)

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(
        r'Model test, $t_\mathrm{start} = 0.5$, scale $\sigma = 0.1$, amplitude $A = 5$, skew $\alpha = 3$, $\mathrm{bkg} = 3$')
    plt.savefig('burst_test1.png', format='png')
    plt.close()

    print('... saved in burst_test1.png. \n')

    print('Test 2: Three spikes')

    event_time1 = 0.2
    scale1 = np.log(0.05)
    amp1 = np.log(5)
    skew1 = np.log(5)

    event_time2 = 0.4
    scale2 = np.log(0.01)
    amp2 = np.log(10)
    skew2 = np.log(1.0)

    event_time3 = 0.7
    scale3 = np.log(0.005)
    amp3 = np.log(2)
    skew3 = np.log(10)

    bkg = np.log(3)

    theta = [event_time1, scale1, amp1, skew1, event_time2, scale2, amp2, skew2, \
             event_time3, scale3, amp3, skew3, bkg]

    b = burstmodel.BurstDict(times, counts, [word.TwoExp, word.TwoExp, word.TwoExp])
    theta_packed = b.wordobject._pack(theta)
    theta_exp = b.wordobject._exp(theta_packed)
    y = b.wordmodel(times, theta_exp)

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(r'Model test, $t_\mathrm{start} = 0.2, 0.4, 0.7$,' \
              r' scale $\sigma = 0.05, 0.01, 0.1$, amplitude $A = 5, 10, 2$,' \
              r' skew $\alpha = 5, 1, 10$, $\mathrm{bkg} = 3$', fontsize=10)
    plt.savefig('burst_test2.png', format='png')
    plt.close()

    print('... saved in burst_test2.png. \n')

    print('Test 3: call method model_means to make same model as above')

    print(theta)
    y = b.model_means(theta, nbins=10)

    plt.figure()
    plt.plot(times, y, lw=2, color='black')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title(r'Model test, $t_\mathrm{start} = 0.2, 0.4, 0.7$,' \
              r' scale $\sigma = 0.05, 0.01, 0.1$, amplitude $A = 5, 10, 2$,' \
              r' skew $\alpha = 5, 1, 10$, $\mathrm{bkg} = 3$', fontsize=10)
    plt.savefig('burst_test3.png', format='png')
    plt.close()

    print('... saved in burst_test3.png \n')

    print('burst_test2.png and burst_test3.png should look the same!')

    return


def wordposterior_tests():
    ## 1-second time array
    times = np.arange(1000.0) / 1000.0
    ## dummy variable: I don't currently have counts
    counts = np.ones(len(times))

    ## define burst model
    b = burstmodel.BurstDict(times, counts, [word.TwoExp, word.TwoExp, word.TwoExp])
    ## define posterior for model
    wpost = burstmodel.WordPosterior(times, counts, b)

    ## define a bunch of parameters
    event_time1 = 0.2
    scale1 = np.log(0.05)
    amp1 = np.log(5)
    skew1 = np.log(5)

    event_time2 = 0.4
    scale2 = np.log(0.01)
    amp2 = np.log(10)
    skew2 = np.log(1.0)

    event_time3 = 0.7
    scale3 = np.log(0.005)
    amp3 = np.log(2)
    skew3 = np.log(10)

    bkg = np.log(3)

    theta = [event_time1, scale1, amp1, skew1, event_time2, scale2, amp2, skew2, \
             event_time3, scale3, amp3, skew3, bkg]

    print('Test 1: Testing prior with reasonable parameters:')
    print('This should be 0: ' + str(wpost.logprior(theta)) + "\n")

    print('Test 2: Testing prior with unreasonable parameter in burst model: ')
    theta[0] = 2.1
    print('This should be -inf, event time out of bounds: ' + str(wpost.logprior(theta)) + "\n")

    print('Test 3: Testing prior with unreasonable background parameter: ')
    theta[0] = 0.2
    theta[-1] = np.log(3.8e5)
    print(
        'This should be -inf, backgorund count rate is above saturation count rate: ' + str(
            wpost.logprior(theta)) + "\n")

    print('Test 4: Log-likelihood with amplitudes: ')
    theta[-1] = np.log(3.0)

    print('This likelihood should be small, parameters do not match data: ' + str(wpost.loglike(theta)) + "\n")

    print('Test 5: Log-likelihood for near-zero amplitudes: ')

    amp1 = amp2 = amp3 = -100.0
    theta[-1] = np.log(1.0)

    theta = [event_time1, scale1, amp1, skew1, event_time2, scale2, amp2, skew2, \
             event_time3, scale3, amp3, skew3, bkg]

    print('This likelihood should be okay, parameters match data: ' + str(wpost.loglike(theta)) + "\n")

    print('Test 6: Log-likelihood for data matching initial parameters: ')
    amp1 = np.log(10.0)
    amp2 = np.log(5.0)
    amp3 = np.log(2.0)

    theta = [event_time1, scale1, amp1, skew1, event_time2, scale2, amp2, skew2, \
             event_time3, scale3, amp3, skew3, bkg]

    counts = b.model_means(theta)
    wpost = burstmodel.WordPosterior(times, counts, b)

    print('This likelihood should be okay, likelihood for same parameter set that data is created from: ' + str(
        wpost.loglike(theta)) + "\n")

    print('Test 7: Testing the logposterior of the last model: ')
    print('This should be same as Test 6: ' + str(wpost.logposterior(theta)) + "\n")

    print('Test 8: Testing the logposterior with an unreasonable prior: ')
    theta[-1] = np.log(3.8e5)
    print('This should be -inf, bkg > saturation count rate: ' + str(wpost.logposterior(theta)) + "\n")

    return
        
    
    




