import numpy as np
import lightcurve

def word(time, width, skew = 2.0):

    t = np.array(time)/width
    #y = exp(((x>0)*2-1)*x)
    y = np.zeros_like(t)
    y[t<=0] = np.exp(t[t<=0])
    y[t>0] = np.exp(-t[t>0]/skew)

    return y
#    y1 = [np.exp(x) for x in xvar if x < 0]
#    y2 = [np.exp(-x/skew) for x in xvar if x > 0]
#    y1.extend(y2)
#    return y1


#### theta[0] is move parameter
#### theta[1] is scale parameter
#### theta[2] is amplitude

def event_rate(time, theta):

    #x = x - theta[0]

    move = theta[0]
    scale = theta[1]
    amp = theta[2]

    time = time - move
    counts = amp*word(time, scale)

    return counts
    
### note: theta_all is a numpy array of n by m, 
### where n is the number of peaks, and m is the number of parameters
### per peak
def time_map(Delta, T, theta_all, nbins=10):

    delta = Delta/nbins
    nsmall = int(T/delta)
    time_small = np.arange(nsmall)*delta
    counts_small = np.zeros(nsmall)

    for t in theta_all:
        counts_temp = event_rate(time_small, t)
        counts_small = counts_small + counts_temp
    
    return counts_small 





# Poisson log likelihood based on a set of rates
# log[ prod exp(-lamb)*lamb^x/x! ]
# exp(-lamb)

### lambdas: numpy array of Poisson rates: mean expected integrated in a bin
import scipy.special
def log_likelihood(lambdas, data):
    return -np.sum(lambdas) + np.sum(data*np.log(lambdas))\
		-np.sum(scipy.special.gammaln(data + 1))






