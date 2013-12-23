import matplotlib.pyplot as plt
from pylab import *

import numpy as np

class Word(object):
    ''' General Word class: container for word shapes of various forms.
    !!! DO NOT INSTANTIATE THIS CLASS! USE SUBCLASSES INSTEAD!!!'''

    ### on initialisation, read in list of times and save as attribute
    def __init__(self, times):
        self.times = np.array(times)

    def _pack(self, npars, theta_flat):

        ''' General pack method:
            Requires 
            npars = list of numbers of parameters for each element in new packed                    array
            theta_flat = simple list or numpy 1D array with parameters
        '''
        theta_new = []

        ## dummy variable to count numbers of parameter I have already iterated 
        ## through
        par_counter = 0
        ## loop over all parameters
        for n in npars:
            theta_new.append(theta_flat[par_counter:par_counter+n])
            par_counter = par_counter + n

        ## theta_new will be a weird list with len(npars) elements and each
        ## element will have length n, for each n in npars
        return theta_new

    def _unpack(self, theta):
        ''' General unpack function
        take a weirdly shaped list or numpy array in n dimensions
        and flatten array to 1d
        '''
        theta_flat = []
        for t in theta:
            theta_flat.extend(t)
        return np.array(theta_flat)

    def __call__(self, *theta_all):
        return model(*theta_all)


class TwoExp(Word, object):
    ''' This class contains the definitions for a simple word shape
    corresponding to a rising exponential, followed by a falling exponential,
    with a sharp peak in the middle.'''

    def __init__(self, times):
        self.npar = 4
        Word.__init__(self, times)
        return


    def model(self, event_time, scale=1.0, amp=1.0, skew=1.0):
        ''' The model method contains the actual function definition.
        Returns a numpy-array of size len(self.times)
        Parameters:
        event_time = start time of word relative to start time of time series
        scale = horizontal scale parameter to stretch/compress word
        skew = skewness parameter: how much faster is the rise than the decay?
        '''
        t = (self.times-event_time)/scale
        y = np.zeros_like(t)
        y[t<=0] = np.exp(t[t<=0])
        y[t>0] = np.exp(-t[t>0]/skew)

        return np.array(amp*y)
  
    def __call__(self, theta_flat):
        
        if not type(theta_flat) == list and not type(theta_flat) == np.array:
            theta_flat = [theta_flat]
        return self.model(*theta_flat)





class CombinedWords(Word, object):
    ''' This class combines several individual Word objects and
        combines them to a single time series model.
        Instantiate with an array of times and a list with Word object
        definitions
    '''
    def __init__(self, times, wordlist):

        ### instantiate word objects
        self.wordlist = [w(times) for w in wordlist]
        self.npar = np.sum([w.npar for w in self.wordlist])
        Word.__init__(self,times)
        return

    ### theta_all is packed
    def model(self, theta_packed):

        y = np.zeros(len(self.times))
        error_theta_packed = 'Length of theta_all does not match length of word list'
        assert len(theta_packed) == len(self.wordlist), error_theta_packed

        error_theta_individual = 'Number of elements in theta does not match required number of parameters in word!'
        for t,w in zip(theta_packed, self.wordlist):
            assert len(t) == w.npar, error_theta_individual

            y = y + w(t) ## add word to output array

        return y
 

    def __call__(self, theta_flat):

        theta_packed = self._pack([w.npar for w in self.wordlist], theta_flat)
        return self.model(theta_packed)



