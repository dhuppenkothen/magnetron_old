import matplotlib.pyplot as plt
from pylab import *

import numpy as np

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

    def _pack(self, theta_flat, npars=None):

        """ General pack method:
            Requires
            npars = list of numbers of parameters for each element in new packed                    array
            theta_flat = simple list or numpy 1D array with parameters
        """
        theta_new = []

        if npars is None:
            npars = self.npar
            ## dummy variable to count numbers of parameter I have already iterated
        ## through
        par_counter = 0
        if size(npars) == 1:
            npars = [npars]
            ## loop over all parameters
        for n in npars:
            theta_new.append(theta_flat[par_counter:par_counter + n])
            par_counter = par_counter + n

        ### if there are more parameters than are in the words, append
        ### the rest at the end
        if np.sum(npars) < len(theta_flat):
            theta_new.extend(theta_flat[np.sum(npars):])
            ## theta_new will be a weird list with len(npars) elements and each
        ## element will have length n, for each n in npars
        return theta_new

    @staticmethod
    def _unpack(theta):
        """ General unpack function
        take a weirdly shaped list or numpy array in n dimensions
        and flatten array to 1d
        """
        theta_flat = []
        for t in theta:
            theta_flat.extend(t)
        return np.array(theta_flat)

    def __call__(self, *theta_all):
        return self.model(*theta_all)


class TwoExp(Word, object):
    """ This class contains the definitions for a simple word shape
    corresponding to a rising exponential, followed by a falling exponential,
    with a sharp peak in the middle."""

    def __init__(self, times):
        self.npar = 4
        Word.__init__(self, times)
        return

    @staticmethod
    def _exp(theta_packed):
#        print(theta_packed)
        d = depth(theta_packed)
        #print('depth: ' + str(d))
        if d > 1:
            theta_temp = theta_packed[0]
        else:
            theta_temp = theta_packed
        #print('theta_packed in _exp: ' + str(theta_temp))
        theta_exp = [theta_temp[0], np.exp(theta_temp[1]), np.exp(theta_temp[2]), np.exp(theta_temp[3])]
        if d > 1:
            theta_exp = [theta_exp]
            theta_exp.extend(np.exp(theta_packed[1:]))
        return theta_exp

    @staticmethod
    def _log(theta_packed):

        d = depth(theta_packed)
        if d > 1:
            theta_temp = theta_packed[0]
        else:
            theta_temp = theta_packed
        theta_log = [theta_temp[0], np.log(theta_temp[1]), np.log(theta_temp[2]), np.log(theta_temp[3])]
        if d > 1:
            theta_log = [theta_log]
            theta_log.extend(np.log(theta_packed[1:]))

        return theta_log

    def model(self, event_time, scale=1.0, amp=1.0, skew=1.0):
        """ The model method contains the actual function definition.
        Returns a numpy-array of size len(self.times)
        Parameters:
        event_time = start time of word relative to start time of time series
        scale = horizontal scale parameter to stretch/compress word
        skew = skewness parameter: how much faster is the rise than the decay?
        """
        t = (self.times - event_time) / scale
        y = np.zeros_like(t)
        y[t <= 0] = np.exp(t[t <= 0])
        y[t > 0] = np.exp(-t[t > 0] / skew)

        return np.array(amp * y)

    def logprior(self, theta_packed):
        print('theta_packed: ' + str(theta_packed))
        if depth(theta_packed) > 1:
            theta_flat = theta_packed[0]
        else:
            theta_flat = theta_packed


        event_time = theta_flat[0]
        scale = np.log(theta_flat[1])
        amp = np.log(theta_flat[2])
        skew = np.log(theta_flat[3])

        #print(np.log(self.T))

        if scale < np.log(self.Delta) or scale > np.log(self.T) or skew < -1.5 or skew > 3.0 or \
                event_time < self.times[0] or event_time > self.times[-1] or \
                amp < -10.0 or amp > np.log(saturation_countrate):
            return -np.Inf
        else:
            return 0.0

    def __call__(self, theta_packed):
        #print('theta_packed in __call__: ' + str(theta_packed))
#        if not type(theta_packed) == list or not type(theta_packed) == np.array:
#            theta_packed = [theta_packed]
        return self.model(*theta_packed)


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

    def _exp(self, theta_packed):
        theta_exp = [w._exp(t) for t, w in zip(theta_packed[:len(self.wordlist)], self.wordlist)]
        if len(theta_packed) > len(self.wordlist):
            theta_exp.extend(np.exp(theta_packed[len(self.wordlist):]))
        return theta_exp

    def _log(self, theta_packed):
        theta_log = [w._log(t) for t, w in zip(theta_packed[:len(self.wordlist)], self.wordlist)]
        if len(theta_packed) > len(self.wordlist):
            theta_log.extend(np.log(theta_packed[len(self.wordlist):]))
        return theta_log

    def _pack(self, theta_flat):
        return Word._pack(self, theta_flat, self.npar_list)


    ### theta_all is packed
    def model(self, theta_packed):

        y = np.zeros(len(self.times))

        error_theta_individual = 'Number of elements in theta does not match required number of parameters in word!'
        for t, w in zip(theta_packed[:len(self.wordlist)], self.wordlist):
            assert len(t) == w.npar, error_theta_individual
            y = y + w(t) ## add word to output array

        return y


    def logprior(self, theta_packed):
#        print('theta_packed: ' + str(theta_packed))

        lprior = 0.0
        for t, w in zip(theta_packed[:len(self.wordlist)], self.wordlist):
            lprior = lprior + w.logprior(t)

        return lprior


    def __call__(self, theta_packed):
        print('theta_packed: ' + str(theta_packed))
    #        theta_packed = self._pack([w.npar for w in self.wordlist], theta_flat)
        return self.model(theta_packed)



