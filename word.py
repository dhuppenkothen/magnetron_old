import matplotlib.pyplot as plt
from pylab import *

import numpy as np

import parameters
### the saturation count rate for Fermi/GBM
### is a global variable
saturation_countrate = 3.5e5

def depth(L):
    d = (isinstance(L, list) or isinstance(L, np.ndarray)) and max(map(depth, L))+1
    return d

class Word(object):
    """ General Word class: container for word shapes of various forms.
    !!! DO NOT INSTANTIATE THIS CLASS! USE SUBCLASSES INSTEAD!!!"""

    ### on initialisation, read in list of times and save as attribute
    def __init__(self, times):
        self.times = np.array(times)
        self.T = self.times[-1] - self.times[0]
        self.Delta = self.times[1] - self.times[0]

    def model(self, *parameters):
        print('Superclass. No model definition for superclass Word')
        return


    def __call__(self, theta):
        return self.model(theta)


class TwoExp(Word, object):
    """ This class contains the definitions for a simple word shape
    corresponding to a rising exponential, followed by a falling exponential,
    with a sharp peak in the middle."""
    npar = 4

    def __init__(self, times):
        self.parclass = parameters.TwoExpCombined
        Word.__init__(self, times)
        return


    def model(self, theta):
        """ The model method contains the actual function definition.
        Returns a numpy-array of size len(self.times)
        Parameters:
        event_time = start time of word relative to start time of time series
        scale = horizontal scale parameter to stretch/compress word
        skew = skewness parameter: how much faster is the rise than the decay?
        """

        t = (self.times - theta.t0) / theta.scale
        y = np.zeros_like(t)
        y[t <= 0] = np.exp(t[t <= 0])
        y[t > 0] = np.exp(-t[t > 0] / theta.skew)

        y = np.array(y)*theta.amp

        if hasattr(theta, "bkg"):
            y = y + theta.bkg

        return y

    def logprior(self, theta):
        #if depth(theta_packed) > 1:
        #    theta_flat = theta_packed[0]
        #else:
        #    theta_flat = theta_packed

        assert isinstance(theta, parameters.TwoExpParameters), "input parameters not an object of type TwoExpParameters"

        event_time = theta.t0
        scale = theta.log_scale
        amp = theta.log_amp
        skew = theta.log_skew

        if scale < np.log(self.Delta) or scale > np.log(self.T) or skew < -1.5 or skew > 3.0 or \
                event_time < self.times[0] or event_time > self.times[-1] or \
                amp < -10.0 or amp > np.log(saturation_countrate):
            lprior =  -np.Inf
        else:
            lprior = 0.0

        if hasattr(theta, "bkg"):
            if theta.bkg < 0 or theta.bkg > saturation_countrate:
                lprior = -np.inf

        return lprior

    def __call__(self, theta):
        assert isinstance(theta, parameters.TwoExpParameters), "input parameters not an object of type TwoExpParameters"
        return self.model(theta)


class CombinedWords(Word, object):
    """ This class combines several individual Word objects and
        combines them to a single time series model.
        Instantiate with an array of times and a list with Word object
        definitions
    """

    def __init__(self, times, wordlist):

        ### instantiate word objects
        self.wordlist = [w(times) for w in wordlist]
        self.npar_list = [w.npar for w in self.wordlist]
        self.npar = np.sum(self.npar_list)

        Word.__init__(self, times)
        return


    def model(self, theta):

        assert isinstance(theta, parameters.TwoExpCombined), "At the moment, I only have TwoExp as model and parameter"\
                                                             "class, and you've not picked it!"

        y = np.zeros(len(self.times))

        for t, w in zip(theta.all, self.wordlist):

            y = y + w(t) ## add word to output array

        if hasattr(theta, "bkg"):
            #print("I am in bkg")
            #print("theta.bkg: " + str(theta.bkg))
            y = y + np.ones(len(self.times))*theta.bkg

        return y

    def logprior(self, theta):

        lprior = 0.0
        for t, w in zip(theta.all, self.wordlist):
            lprior = lprior + w.logprior(t)

        if hasattr(theta, "bkg"):
            if theta.bkg < 0 or theta.bkg > saturation_countrate:
                lprior = -np.inf
        return lprior


    def __call__(self, theta):
        return self.model(theta)



