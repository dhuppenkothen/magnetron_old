#### MODELLING BURSTS WITH WORDS #######################
#
# This module defines classes to model magnetar bursts  
# as superposition of simple shapes ("words")
#
# Classes:
#   BurstDict:  produces a superposition of arbitrary words
#               to use as a model for a magnetar bursts
#       methods: 
#                _create_model: yields a function that takes
#                               a list of times and a bunch of
#                               parameters and returns a count rate
#
#                model_means:   takes a list of times and some parameters,
#                               then makes a finer time grid, computers the
#                               count rate via the output of _create_model and
#                               averages neighbouring bins back to the original
#                               resolution of the initial time grid
#
#  WordPosterior:   class that defines the posterior for a Poisson-distributed
#                   data set and a superposition of words as a model
#
#
#

### python module imports
from collections import defaultdict
import cPickle as pickle
import argparse
import copy

### third party modules
import matplotlib.pyplot as plt
from pylab import * 
import numpy as np
import scipy.special
import emcee
import triangle

### local scripts
import word

saturation_countrate = 3.5e5
#### DEPRECATED: USE WORD CLASS INSTEAD #####
#def word(time, scale, skew = 2.0):

#    t = np.array(time)/scale
#    y = np.zeros_like(t)
#    y[t<=0] = np.exp(t[t<=0])
#    y[t>0] = np.exp(-t[t>0]/skew)

#    return y
#######




class BurstDict(object):

    def __init__(self, times, counts, wordlist):

        self.times = np.array(times)
        self.counts = np.array(counts)
        self.wordlist = wordlist
        self.wordmodel, self.wordobject = self._create_model()
        self.Delta = self.times[1] - self.times[0]
        self.nbins_data = len(self.times)
        return

    def _create_model(self):

        if size(self.wordlist) > 1:
            wordmodel = word.CombinedWords(self.times, self.wordlist)
        #    y = wordmodel(theta_exp[:-1]) + bkg
        elif size(self.wordlist) == 1:
            wordmodel = self.wordlist(self.times)
        #    y = wordmodel(theta_exp[:-1]) + bkg
        else:
            wordmodel = None
        #    y = np.zeros(len(self.times)) + bkg
  

        ### create a model definition that includes the background!
        ### theta_all is flat and has non-log parameters
        def event_rate(model_times, theta_exp):
            
            ### last element must be background counts!
            bkg = theta_exp[-1]
            #print('Theta exp in event_rate: ' + str(theta_exp))
            if size(self.wordlist) > 1:
                wordmodel = word.CombinedWords(model_times, self.wordlist)
                y = wordmodel(theta_exp[:-1]) + bkg
            elif size(self.wordlist) == 1:
                wordmodel = self.wordlist(model_times)
                y = wordmodel(theta_exp[:-1]) + bkg
            else: 
                wordmodel = None
                y = np.zeros(len(model_times)) + bkg

            return y

        return event_rate, wordmodel

    ### theta_all is flat and takes log parameters to be consistent with
    ### likelihood functions
    def model_means(self, theta_all, nbins=10):

        ## small time bin size delta
        delta = self.Delta/nbins
        ## number of small time bins
        nsmall = self.nbins_data*nbins
        ## make a high-resolution time array 
        times_small = np.arange(nsmall)*delta

        if self.wordlist >= 1: 
            theta_all_packed= self.wordobject._pack(theta_all)
            theta_exp = self.wordobject._exp(theta_all_packed)
            #print('theta_exp in model_means: ' + str(theta_exp) + "\n")
        else:
            theta_exp = theta_all
 
        ## compute high-resolution count rate
        rate_small = self.wordmodel(times_small, theta_exp)

        ## add together nbins neighbouring counts
        rate_map = rate_small.reshape(self.nbins_data, nbins)
        rate_map_sum = np.sum(rate_map, axis=1)/float(nbins)

        return rate_map_sum



class WordPosterior(object):

    '''
    WordPosterior class for making a word model posterior 
    note: burstmodel is of type BurstDict

    '''
    def __init__(self, times, counts, burstmodel):
        self.times = times
        self.counts = counts
        self.burstmodel = burstmodel

    def logprior(self, theta):

        bkg = theta[-1]

        lprior = 0
        theta_packed = self.burstmodel.wordobject._pack(theta)
        theta_exp = self.burstmodel.wordobject._exp(theta_packed)
        lprior = lprior + self.burstmodel.wordobject.logprior(theta_exp[:-1])

        if bkg > np.log(saturation_countrate) or np.isinf(lprior):
            return -np.inf
        else: 
            return 0.0     


    ### lambdas: numpy array of Poisson rates: mean expected integrated in a bin
    ### Poisson likelihood for data and a given model
    def _log_likelihood(self, lambdas, data):

        return -np.sum(lambdas) + np.sum(data*np.log(lambdas))\
            -np.sum(scipy.special.gammaln(data + 1))



    ### theta is flat and in log-space
    def loglike(self, theta):
        lambdas = self.burstmodel.model_means(theta) 

        return self._log_likelihood(lambdas, self.counts)


    def logposterior(self, theta):
        return self.logprior(theta) + self.loglike(theta)

    ## compute Bayesian Information Criterion
    def bic(self, theta):
        print('This does not do anything right now!')
        return

    def __call__(self, theta):
        print(theta)
        return self.logposterior(theta)



class BurstModel(object):

    def __init__(self, times, counts):
        self.times = times
        self.counts = counts
        self.T = self.times[-1] - self.times[0]
        self.Delta = self.times[1] - self.times[0]
        #self.burstdict = BurstDict(times, counts, wordlist)
        #self.lpost = WordPosterior(times, counts, self.burstdict) 
        return


        ### note to self: need to implement triangle package and make
        ### shiny triangle plots!
    def mcmc(self, burstmodel, initial_theta, nwalker=500, niter=200, burnin=100, plot=True, plotname = 'test'):

            lpost = WordPosterior(self.times, self.counts, burstmodel)

            if nwalker < 2*len(initial_theta):
                print('Too few walkers! Resetting to 2*len(theta)')
                nwalker = 2*len(initial_theta)

            p0 = [initial_theta+np.random.rand(len(initial_theta))*1.0e-3 for t in range(nwalker)]

            sampler = emcee.EnsembleSampler(nwalker, len(initial_theta), lpost)
            pos, prob, state = sampler.run_mcmc(p0, 200)
            sampler.reset()
            sampler.run_mcmc(pos, 1000, rstate0 = state)

            if plot:
                self.plot_mcmc(sampler.flatchain, plotname = 'test')
            
            return sampler

    def plot_mcmc(self, data, plotname):

            figure = triangle.corner(data, labels= [], truths = np.zeros(10))
            return


    def find_spikes(self, model = word.TwoExp, nmax = 10):

            all_burstdict = []
            all_sampler = []
            all_means, all_std = []


            for n in range(nmax):
                
                ## define burst model   
                wordlist = [model for m in range(n)]
                burstmodel = BurstDict(self.times, self.counts, wordlist) 

                #find position where to put a spike
                if n == 0:
                    theta_init = [np.mean(self.counts)]
                    sampler = self.mcmc(burstmodel, theta_init) 

                    postmean = map(lambda y: np.mean(y), sampler.flatchain)
                    poststd = map(lambda y: np.std(y), sampler.flatchain)

                    all_sampler.append(sampler)
                    all_means.append(postmean)
                    all_std.append(poststd)
                    all_burstdict.append(burstdict)

                    bkg = postmean[0]


                ## extract posterior means from last model run
                old_postmeans = all_means[-1] 
                old_burstdict = all_burstdict[-1]
                model = old_burstdict.model_means(old_postmeans, nbins=10) 

                datamodel_ratio = self.counts/model
                max_diff = max(datamodel_ratio)
                max_ind = np.where(datamodel_ratio == max_diff)[0]
                max_loc = self.times[max_ind]

                if model == word.TwoExp:
                   new_event_time = max_loc
                   new_scale = 0.1*self.T
                   new_amp = max_diff
                   new_skew = 1.0
                   theta_new_init = [new_event_time, new_scale, new_amp, new_skew]

                 
                theta_init = np.zeros(len(old_postmeans)+model.npar)
                theta_init[:len(old_postmeans)-1] = old_postmeans
                theta_init[len(old_postmeans)-1:-1] = theta_new_init
                theta_init[-1] = old_postmeans[-1]

                ## wiggle around parameters a bit
                random_shift = (np.random.rand(len(theta_init))-0.5)/100.0
                theta_init = theta_init*(1.0+random_shift)

                sampler = self.mcmc(burstmodel, theta_init)

                postmean = map(lambda y: np.mean(y), sampler.flatchain)
                poststd = map(lambda y: np.std(y), sampler.flatchain)

                all_sampler.append(sampler)
                all_means.append(postmean)
                all_std.append(poststd)
                all_burstdict.append(burstdict)

            return all_sampler, all_means, all_std, all_burstdict

                ## now I need to: return count rate from previous model
                ## then find highest data/model outlier
                ## place new initial guess there, + small deviation in all paras?
                ## run mcmc
                ## append new posterior solution to old one 
    




#######################################################################
#### OLD IMPLEMENTATION! ##############################################
#######################################################################

class DictPosterior(object):

    '''
    note: implicit assumption is that array times is equally spaced
          nbins_data: number of bins in data
          nbins: multiplicative factor for model light curve bins
    '''
    def __init__(self, times, counts, wordmodel, nbins=10):
        self.times = times
        self.counts = counts
        self.model = model
        #self.npar = npar

        self.Delta = times[1]-times[0]
        self.nbins_data = len(times)
        self.nbins = nbins



    def logprior(self, theta):

        # Our prior for a SINGLE WORD
        # Input: parameter vector, which gets unpacked into named things
        # feel free to change the order if that's how you defined it - BJB

        saturation_countrate = 3.5e5 ### in counts/s
        T = self.times[-1] - self.times[0]
    
 
        skew, bkg, scale, theta_evt = unpack(theta)
#        print('theta_evt in logprior: ' + str(theta_evt)) 
#        print('theta in logprior: ' + str(theta))

        if  scale < self.Delta or scale > T or skew < np.exp(-1.5) or skew > np.exp(3.0) or \
                bkg < 0 or bkg > saturation_countrate:
            return -np.Inf

        all_event_times = theta_evt[:,0]
        all_amp = theta_evt[:,1]
        if np.min(all_event_times) < self.times[0] or np.max(all_event_times) > self.times[-1] or \
                np.min(all_amp) < 0.1/self.Delta or np.max(all_amp) > saturation_countrate:
            return -np.Inf

        return 0.


    ## theta = [scale, skew, move1, amp1, move2, amp2]    
    def loglike(self, theta):
  
#        print('theta in loglike: ' + str(theta))

        ### unpack theta:
        skew, bkg, scale, theta_evt = unpack(theta)
        #print('theta_evt in loglike: ' + str(theta_evt))
        #print('theta in loglike: ' + str(theta))



        lambdas = model_means(self.Delta, self.nbins_data, skew, bkg, scale, theta_evt, nbins=self.nbins)

        return log_likelihood(lambdas, self.counts)

    ## change parameters:
    

    def logposterior(self, theta):
        return self.logprior(theta) + self.loglike(theta)


    def __call__(self, theta):
        return self.logposterior(theta)




#### DEPRECATED: USE WORD CLASS INSTEAD #####
#def word(time, scale, skew = 2.0):

#    t = np.array(time)/scale
#    y = np.zeros_like(t)
#    y[t<=0] = np.exp(t[t<=0])
#    y[t>0] = np.exp(-t[t>0]/skew)

#    return y
#######

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

        rate_temp = event_rate(time_small, event_time, scale, amp, skew)
        rate_small = rate_small + rate_temp

    nrow = len(rate_small)/nbins
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

    #print('theta in unpack: ' + str(theta))

    theta = np.array(theta)
    theta_evt = copy.copy(theta[3:]).reshape((len(theta)-3)/2, 2)

    for i in range(len(theta_evt)):
    #    print(theta_evt[i][1])
        theta_evt[i][1] = np.exp(theta_evt[i][1])

    return skew, bkg, scale, theta_evt


## go from weird shape to numpy array
def pack(skew, bkg, scale, theta_evt):

    theta = np.zeros(len(theta_evt.flatten())+3)
    theta[0] = np.log(skew)
    theta[1] = np.log(bkg)
    theta[2] = np.log(scale)

    for i in range(len(theta_evt)):
        theta_evt[i][1] = np.log(theta_evt[i][1])

    theta[3:] = theta_evt.flatten()

    return theta



def model_burst():

    times = np.arange(1000.0)/10000.0
    counts = np.random.poisson(2000.0, size=len(times))

    skew = 3.0
    scale = 0.005
    bkg = 5.0

    nspikes = 2
    theta_evt = np.zeros((nspikes,2))

    for i in range(nspikes):
        theta_evt[i] = [np.random.rand()*times[-1], 10.0/(times[1]-times[0])]
        
    theta = pack(skew, bkg, scale, theta_evt)

    return theta

def initial_guess(times, counts, skew, bkg, scale, theta_evt):

    times = times - times[0]

    ### initialise guess for burst 090122218, tstart = 47.4096 
    #skew = 5.0
    #bkg = 2000.0
    #scale = 0.01
    #theta_evt = np.array([[0.82, 30000], [0.87, 60000], [0.95, 50000], [1.05, 20000]])


    Delta = times[1]-times[0]
    nbins_data = len(times)

    counts_model = model_means(Delta, nbins_data, skew, bkg, scale, theta_evt, nbins=10)

    figure()
    plt.plot(times, counts, 'k')
    plt.plot(times, counts_model, 'r')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title('Light curve with initial guess for model', fontsize=18)

    theta = pack(skew, bkg, scale, np.array(theta_evt))

    return theta, counts_model


### put in burst times array and counts array
def test_burst(times, counts, theta_guess, namestr = 'testburst', nwalker=32):

    skew, bkg, scale, theta_evt = unpack(theta_guess)

    #Delta = times[1]-times[0]
    #nbins_data = len(times)
 
    theta, counts_model = initial_guess(times, counts, skew, bkg, scale, theta_evt)

    figure()
    plt.plot(times, counts, 'k')
    plt.plot(times, counts_model, 'r')
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title('Light curve with initial guess for model', fontsize=18)
    plt.savefig(namestr + '_initialguess.png', format='png')
    plt.close()

    theta = pack(skew, bkg, scale, theta_evt)   

    lpost = DictPosterior(times, counts)
   
    if nwalker < 2*len(theta):
        nwalker = 2*len(theta)

    p0 = [theta+np.random.rand(len(theta))*1.0e-3 for t in range(nwalker)]

    sampler = emcee.EnsembleSampler(nwalker, len(theta), lpost)
    pos, prob, state = sampler.run_mcmc(p0, 200)
    sampler.reset()
    sampler.run_mcmc(pos, 1000, rstate0 = state)

    plot_test(times, counts, sampler.flatchain[-10:])
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Counts per bin', fontsize=18)
    plt.title('Light curve with draws from posterior sample', fontsize=18)
    plt.savefig(namestr + '_posteriorsample.png', format='png')
    plt.close()


    ### quick hack to save emcee sampler to disc
    f = open(namestr + '_sampler.dat', 'w')
    pickle.dump(sampler, f)
    f.close()

    return 


def plot_test(times, counts, theta):

    plt.plot(times, counts, 'k')


    Delta = times[1]-times[0]
    nbins_data = len(times)


    for t in np.atleast_2d(theta):

        skew, bkg, scale, theta_evt = unpack(t)
        counts_model = model_means(Delta, nbins_data, skew, bkg, scale, theta_evt, nbins=10)
        plt.plot(times, counts_model, 'r')

    return

# Poisson log likelihood based on a set of rates
# log[ prod exp(-lamb)*lamb^x/x! ]
# exp(-lamb)

### lambdas: numpy array of Poisson rates: mean expected integrated in a bin
import scipy.special
def log_likelihood(lambdas, data):

    return -np.sum(lambdas) + np.sum(data*np.log(lambdas))\
        -np.sum(scipy.special.gammaln(data + 1))


def conversion(filename):
    f=open(filename, 'r')
    output_lists=defaultdict(list)
    for line in f:
        if not line.startswith('#'):
             line=[value for value in line.split()]
             for col, data in enumerate(line):
                 output_lists[col].append(data)
    return output_lists

def main(): 

    data = conversion(filename)
    times = np.array([float(t) for t in data[0]])    
    counts = np.array([float(c) for c in data[1]])

    test_burst(times, counts, namestr = namestr, nwalker=nwalkers)
   
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to play around with Fermi/GBM magnetar data')
    parser.add_argument('-f', '--filename', action='store', dest ='filename', help='input filename')
    parser.add_argument('-n', '--namestr', action='store', dest='namestr', help='Output filename string')
    parser.add_argument('--nwalkers', action='store', default='32', help='Number of emcee walkers')
 

    clargs = parser.parse_args()
    
    nwalkers = int(clargs.nwalkers)
    filename = clargs.filename
    namestr = clargs.namestr

    main()
