import numpy as np

def word(time, scale, skew = 2.0):

    t = np.array(time)/scale
    #y = exp(((x>0)*2-1)*x)
    y = np.zeros_like(t)
    y[t<=0] = np.exp(t[t<=0])
    y[t>0] = np.exp(-t[t>0]/skew)

    return y
#    y1 = [np.exp(x) for x in xvar if x < 0]
#    y2 = [np.exp(-x/skew) for x in xvar if x > 0]
#    y1.extend(y2)
#    return y1


def event_rate(time, event_time, scale, amp, skew):

    #x = x - theta[0]

    time = time - event_time
    counts = amp*word(time, scale, skew)

    return counts
    
### note: theta_all is a numpy array of n by m, 
### where n is the number of peaks, and m is the number of parameters
### per peak
def model_means(Delta, nbins_data, skew, bkg, scale, theta_evt, nbins=10):

    delta = Delta/nbins
    nsmall = nbins_data*nbins
    time_small = np.arange(nsmall)*delta
    rate_small = np.zeros(nsmall)


    for (event_time, amp) in theta_evt:
       
        rate_temp = event_rate(time_small, t[0], event_time, scale, amp, skew)
        rate_small = rate_small + rate_temp

    nrow = len(counts_small)/nbins
    rate_map = rate_small.reshape(nrow, nbins)
    rate_map_sum = np.sum(rate_map, axis=1)*delta

    rate_map_all = rate_map_sum + bkg*Delta

    return rate_map_all


## go from numpy array weird shape
def unpack(theta):
    '''
    unpacks the numpy array of parameters into a form that model_means can read.

    returns: skew, bkg, scale, theta_evt

    theta_evt is an array of npeaks by 2, where npeaks is the number of peaks
    each row is (event_time, amp)
    '''
    skew = np.exp(theta[0])
    bkg = np.exp(theta[1])
    scale = np.exp(theta[2])

    theta_evt = theta[3:].reshape((len(theta)-3)/2, 2)
   
    for i,t in enumerate(theta_evt):
        theta_evt[i][1] = np.exp(t[1])
 
    return skew, bkg, scale, theta_evt


## go from weird shape to numpy array
def pack(skew, bkg, scale, theta_evt):

    theta = np.zeros(len(theta_evt.flatten())+3)
    theta[0] = np.log(skew)
    theta[1] = np.log(bkg)
    theta[2] = np.log(scale)

    for i,t in enumerate(theta_evt):
        theta_evt[i][1] = np.log(t[1])

    theta[3:] = theta_evt.flatten()

    return theta


class DictPosterior(object):

    '''
    note: implicit assumption is that array times is equally spaced
          nbins_data: number of bins in data
          nbins: multiplicative factor for model light curve bins
    '''
    def __init__(self, times, counts, model, npar, nbins=10):
        self.times = times
        self.counts = counts
        self.model = model
        self.npar = npar

        self.Delta = times[1]-times[0]
        self.nbins_data = len(times)
        self.nbins = nbins



    def logprior(theta):

        # Our prior for a SINGLE WORD
        # Input: parameter vector, which gets unpacked into named things
        # feel free to change the order if that's how you defined it - BJB

        saturation_countrate = 3.5e5 ### in counts/s
        T = self.times[-1] - self.times[0]
    

        skew, bkg, scale, theta_evt = unpack(theta)
        if  scale < self.Delta or scale > T or skew < np.exp(-1.5) or skew > np.exp(3.0):
            return -np.Inf

        all_event_times = theta_evt[:,0]
        all_amp = theta_evt[:,1]
        if np.min(event_times) < self.times[0] or np.max(event_times) > self.times[1] or \
                np.min(all_amp) < 0.1/self.Delta or np.max(all_amp) > saturation_countrate:
            return -np.Inf

        return 0.


    ## theta = [scale, skew, move1, amp1, move2, amp2]    
    def loglike(theta):
        
        ### unpack theta:
        skew, bkg, scale, theta_evt = unpack(theta)
        
        lambdas = model_means(self.Delta, self.nbins_data, skew, bkg, scale, theta_evt, nbins=self.nbins)

        return log_likelihood(lambdas, self.counts)

    ## change parameters:
    

    def logposterior(theta):
        return logprior(theta) + loglike(theta)


    def __call__(theta):
       # return logposterior(theta   def logposterior(theta):
        return logprior(theta) + loglike(theta)


    def __call__(theta):
        return logposterior(theta)


# Poisson log likelihood based on a set of rates
# log[ prod exp(-lamb)*lamb^x/x! ]
# exp(-lamb)

### lambdas: numpy array of Poisson rates: mean expected integrated in a bin
import scipy.special
def log_likelihood(lambdas, data):
    return -np.sum(lambdas) + np.sum(data*np.log(lambdas))\
        -np.sum(scipy.special.gammaln(data + 1))



# Our prior for a SINGLE WORD
# Input: parameter vector, which gets unpacked into named things
# feel free to change the order if that's how you defined it - BJB
#def log_prior(params):
#    [amplitude, scale, skew] = unpack(params)
#    if amplitude < np.log(-1000.) or amplitude > np.log(1000.) or \
#        scale < np.log(-6.) or scale > np.log(12.) or skew < np.log(-1.5) or \
#        skew > np.log(1.5):
#        return -np.Inf
#    return 0.
#    # okay now do it proper...

