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
#               plot_model:     takes a parameter vector and plots
#                               the data together with the model corresponding to
#                               that parameter vector
#
#  WordPosterior:   class that defines the posterior for a Poisson-distributed
#                   data set and a superposition of words as a model
#
# TODO: add command line functionality
#

### python module imports
from collections import defaultdict
import cPickle as pickle
import argparse
import glob

### third party modules
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
#from pylab import *
import matplotlib.cm as cm

import numpy as np
import scipy.special
import emcee
import triangle
from scipy.stats.mstats import mquantiles as quantiles


### local scripts
import word
import parameters

saturation_countrate = 3.5e5


#### READ ASCII DATA FROM FILE #############
#
# This is a useful little function that reads
# data from file and stores it in a dictionary
# with as many lists as the file had columns.
# The dictionary has the following architecture
# (e.g. for a file with three columns):
#
# {'0':[1st column data], '1':[2nd column data], '2':[3rd column data]}
#
#
# NOTE: Each element of the lists is still a *STRING*, because
# the function doesn't make an assumption about what type of data
# you're trying to read! Numbers need to be converted before using them!
#
def conversion(filename):
    f=open(filename, 'r')
    output_lists = defaultdict(list)
    for line in f:
        if not line.startswith('#'):
             line=[value for value in line.split()]
             for col, data in enumerate(line):
                 output_lists[col].append(data)
    return output_lists

######################################################################

###### READ GBM LIGHT CURVES FROM FILE
#
# Light curves are stored in ASCII files
# with two columns, where
#
# column 1 : time stamps (middle of time bin)
# column 2 : the number of counts *per bin*
#
# Takes a filename as string and returns two numpy-arrays
# with time stamps and counts per bin
#
def read_gbm_lightcurves(filename):

    data = conversion(filename)
    times = np.array([float(t) for t in data[0]])
    counts = np.array([float(c) for c in data[1]])

    return times, counts

#######################################################################




class BurstDict(object):

    def __init__(self, times, counts, wordlist):

        self.times = np.array(times)
        self.counts = np.array(counts)
        self.wordlist = wordlist
        self.wordmodel, self.wordobject = self._create_model()
        # noinspection PyPep8Naming
        self.Delta = self.times[1] - self.times[0]
        self.nbins_data = len(self.times)
        self.countrate = np.array(counts)/self.Delta

        return

    def _create_model(self,):

        if np.size(self.wordlist) > 1 or type(self.wordlist) is list:
            wordobject = word.CombinedWords(self.times, self.wordlist)
        elif np.size(self.wordlist) == 1:
            if word.depth(self.wordlist) > 1:
                wordobject = self.wordlist[0](self.times)
            else:
                wordobject = self.wordlist(self.times)
        else:
            wordobject = None


        ### create a model definition that includes the background!
        def event_rate(model_times, theta, log=True, bkg=True):

            assert isinstance(theta, parameters.TwoExpParameters) or isinstance(theta, parameters.TwoExpCombined),\
                "input parameters not an object of type TwoExpParameters"

            if np.size(self.wordlist) > 1 or type(self.wordlist) is list:
                wordobject  = word.CombinedWords(model_times, self.wordlist)
                y = wordobject(theta)
            elif np.size(self.wordlist) == 1:
                if word.depth(self.wordlist) > 1:
                    wordobject = self.wordlist[0](model_times)
                else:
                    wordobject = self.wordlist(model_times)
                y = wordobject(theta)
            else:
                if bkg:
                    y = np.ones(len(model_times))*theta.bkg

            return y

        return event_rate, wordobject


    def model_means(self, theta, nbins=10):

        assert isinstance(theta, parameters.TwoExpParameters) or isinstance(theta, parameters.TwoExpCombined),\
            "input parameters not an object of type TwoExpParameters"

        ## small time bin size delta
        delta = self.Delta/nbins
        ## number of small time bins
        nsmall = self.nbins_data*nbins
        ## make a high-resolution time array 
        times_small = np.arange(nsmall)*delta

        rate_small = self.wordmodel(times_small, theta)

        ## add together nbins neighbouring counts
        rate_map = rate_small.reshape(self.nbins_data, nbins)
        rate_map_sum = np.sum(rate_map, axis=1)/float(nbins)

        return rate_map_sum


    def poissonify(self, theta):

        """
        Make a Poisson light curve out of a model with parameters theta.
        Takes the parameters, makes a light curve with model count rate,
        then calculates the counts per bin x_i, and picks from a Poisson
        distribution with the mean x_i in every bin.
        """

        model_countrate = self.model_means(theta)
        model_counts = model_countrate*self.Delta
        poisson = np.vectorize(lambda x: np.random.poisson(x))
        return poisson(model_counts)/self.Delta



    def plot_model(self, theta, postmax = None, plotname='test'):

        assert isinstance(theta, parameters.TwoExpParameters) or isinstance(theta, parameters.TwoExpCombined),\
            "input parameters not an object of type TwoExpParameters"

        model_counts = self.model_means(theta, nbins=10)
        if not postmax is None:

            assert isinstance(postmax, parameters.TwoExpParameters) or isinstance(theta, parameters.TwoExpCombined),\
                "input parameters not an object of type TwoExpParameters"
            model_counts_postmax = self.model_means(postmax, nbins=10)


        fig = plt.figure(figsize=(10,8))
        plt.plot(self.times, self.countrate, lw=1, color='black', label='input data')
        plt.plot(self.times, model_counts, lw=2, color='red', label='model light curve: posterior mean')
        if not postmax is None:
            plt.plot(self.times, model_counts_postmax, lw=2, color='blue', label='model light curve: posterior max')
        plt.legend()
        plt.xlabel('Time [s]', fontsize=18)
        plt.ylabel('Count Rate [counts/bin]', fontsize=18)
        plt.title('An awesome model light curve!')
        plt.savefig(plotname + '_lc.png', format='png')
        plt.close()
        return

################################################################
################################################################
################################################################

class WordPosterior(object):

    """
    WordPosterior class for making a word model posterior

    times: input time stamps of time series
    counts: corresponding counts per bin
    model: BurstDict instance with word model
    scale_locked: is the scale the same for all words?
    skew_locked: is the skew the same for all words?
    log: are parameters going to be input as logs?
    bkg: is there a background parameter?

    """
    def __init__(self, times, counts, model, scale_locked=False, skew_locked=False, log=True, bkg=True):
        self.times = times
        self.counts = counts
        self.model = model
        self.Delta = self.times[1] - self.times[0]
        self.countrate = self.counts/self.Delta

        self.log = log
        self.bkg = bkg
        self.scale_locked = scale_locked
        self.skew_locked = skew_locked

        if np.size(model.wordlist) > 1:
            self.ncomp = len(model.wordlist)
            self.wordmodel = model.wordlist[0]
        elif np.size(model.wordlist) == 1:

            if word.depth(model.wordlist) > 1:
                self.wordmodel = model.wordlist[0]
            else:
                self.wordmodel = model.wordlist
            self.ncomp = 1
        else:
            self.wordmodel = None
            self.ncomp = 0

        return


    def logprior(self, theta):

        if not isinstance(theta, parameters.Parameters):
            if self.wordmodel is word.TwoExp or self.ncomp == 0:
                theta = parameters.TwoExpCombined(theta, self.ncomp, parclass=parameters.TwoExpParameters,
                                                  scale_locked=self.scale_locked, skew_locked=self.skew_locked,
                                                  log=self.log, bkg=self.bkg)

            else:
                raise Exception("Word class not known! Needs to be implemented!")


        assert isinstance(theta, (parameters.TwoExpParameters, parameters.TwoExpCombined)),\
            "input parameters not an object of type TwoExpParameters"

        return self.model.wordobject.logprior(theta)



    ### lambdas: numpy array of Poisson rates: mean expected integrated in a bin
    ### Poisson likelihood for data and a given model

    def _log_likelihood(self, lambdas, data):


        return -np.sum(lambdas*self.Delta) + np.sum(data*np.log(lambdas*self.Delta))\
            -np.sum(scipy.special.gammaln(data + 1))



    ### theta is flat and in log-space
    def loglike(self, theta):

        if not isinstance(theta, parameters.Parameters):
            if self.wordmodel is word.TwoExp or self.ncomp == 0:
                theta = parameters.TwoExpCombined(theta, self.ncomp, parclass=parameters.TwoExpParameters,
                                                  scale_locked=self.scale_locked, skew_locked=self.skew_locked,
                                                  log=self.log, bkg=self.bkg)

            else:
                raise Exception("Word class not known! Needs to be implemented!")



        assert isinstance(theta, parameters.TwoExpParameters) or isinstance(theta, parameters.TwoExpCombined),\
            "input parameters not an object of type TwoExpParameters"

        lambdas = self.model.model_means(theta)

        return self._log_likelihood(lambdas, self.countrate)


    def logposterior(self, theta):

        if not isinstance(theta, parameters.Parameters):
            if self.wordmodel is word.TwoExp or self.ncomp == 0:
                theta = parameters.TwoExpCombined(theta, self.ncomp, parclass=parameters.TwoExpParameters,
                                                  scale_locked=self.scale_locked, skew_locked=self.skew_locked,
                                                  log=self.log, bkg=self.bkg)

            else:
                raise Exception("Word class not known! Needs to be implemented!")

        return self.logprior(theta) + self.loglike(theta)

    ## compute Bayesian Information Criterion
    @staticmethod
    def bic(theta):
        print('This does not do anything right now!')
        return

    def __call__(self, theta):
        return self.logposterior(theta)

###############################################################################
###############################################################################
###############################################################################



class BurstModel(object):

    def __init__(self, times, counts):
        self.times = times - times[0]
        self.tstart = times[0]
        self.counts = counts

        # noinspection PyPep8Naming
        self.T = self.times[-1] - self.times[0]
        self.Delta = self.times[1] - self.times[0]

        return


        ### note to self: need to implement triangle package and make
        ### shiny triangle plots!
    def mcmc(self, model, initial_theta, nwalker=500, niter=200, burnin=100, scale_locked=False,
             skew_locked=False, log=True, bkg=True, plot=True, plotname = 'test'):


        lpost = WordPosterior(self.times, self.counts, model, scale_locked=scale_locked, skew_locked=skew_locked,
                              log=log, bkg=bkg)

        if nwalker < 2*len(initial_theta):
            print('Too few walkers! Resetting to 2*len(theta)')
            nwalker = 2*len(initial_theta)


        print("Finding initial walkers:")
        p0 = []
        for t in range(nwalker):
            lpost_theta_init = -np.inf
            counter = 0
            while np.isinf(lpost_theta_init):
                if counter > 1000:
                    raise Exception("Can't find initial theta inside prior!")
                p0_temp = initial_theta+np.random.rand(len(initial_theta))*1.0e-3
                lpost_theta_init = lpost(p0_temp)
                counter =+ 1
            p0.append(p0_temp)

        print("Done. Starting the sampling.")
        ### test code: run burnin with a few, long chains, then the actual sampler with the results
        ### from the burnin phase

        #sampler_burnin = emcee.EnsembleSampler(nwalker_burnin, len(initial_theta), lpost)
        #pos, prob, state = sampler_burnin.run_mcmc(p0_burnin, burnin)

        #niter_burnin = int(np.round(np.mean(sampler_burnin.acor))*nwalker/nwalker_burnin)
        #sampler_burnin.reset()

        #sampler_burnin.run_mcmc(pos, niter_burnin , rstate0 = state)

        #p0 = np.random.choice(sampler_burnin.flatchain, size=nwalker, replace=False,
        #                      p=sampler_burnin.flatlnprobability)

        sampler = emcee.EnsembleSampler(nwalker, len(initial_theta), lpost)
        pos, prob, state = sampler.run_mcmc(p0, burnin)
        print("Burned in. Now doing real run ...")
        sampler.reset()
        sampler.run_mcmc(pos, niter, rstate0 = state)

        print("...sampling done. Am I plotting?")

        if plot:
            print("Yes, I'm plotting! Stay tuned for awesome plots")
            if np.size(model.wordlist) == 0:
                if log:
                    plotlabels = ["log(bkg)"]
                else:
                    plotlabels = ["bkg"]
            else:
                if lpost.wordmodel is word.TwoExp:
                    if log:
                        plotlabels = np.array([parameters.TwoExpParameters.parnames_log for n in range(lpost.ncomp)])
                        plotlabels = plotlabels.flatten()
                    else:
                        plotlabels = np.array([parameters.TwoExpParameters.parnames for n in range(lpost.ncomp)])
                        plotlabels = plotlabels.flatten()

                    par_scale = plotlabels[1]
                    par_skew = plotlabels[3]

                    if scale_locked:
                        plotlabels = np.delete(plotlabels, np.where(plotlabels == par_scale)[0])
                        plotlabels = np.append(plotlabels, par_scale)

                    if skew_locked:
                        plotlabels = np.delete(plotlabels, np.where(plotlabels == par_skew)[0])
                        plotlabels = np.append(plotlabels, par_skew)

                    if bkg:
                        if log:
                            plotlabels = np.append(plotlabels, "log(bkg)")
                        else:
                            plotlabels = np.append(plotlabels, "bkg")

            print("plotlabels: " + str(plotlabels))
            self.plot_mcmc(sampler.flatchain[-5000:], plotname=plotname, plotlabels=plotlabels)

        print('Sampler autocorrelation length: ' + str(sampler.acor))
        print('Sampler mean acceptance fraction: ' + str(np.mean(sampler.acceptance_fraction)))

        return sampler


    def find_postmax(self, sampler, ncomp, scale_locked=False, skew_locked=False, log=True, bkg=True):

        ### first attempt: get maxima from marginalised posteriors
        flatchain = sampler.flatchain[-10000:]

        if np.shape(flatchain)[0] > np.shape(flatchain)[1]:
            flatchain = np.transpose(flatchain)


        quants = self._quantiles(flatchain, ncomp, scale_locked=scale_locked, skew_locked=skew_locked, log=log, bkg=bkg)

        ### second attempt: find maximum posterior probability, return corresponding parameter vector
        postprob = sampler.flatlnprobability

        maxi = postprob.argmax()
        postmax = sampler.flatchain[maxi]


        if self.model is word.TwoExp:

            postmax = parameters.TwoExpCombined(postmax, ncomp, parclass=parameters.TwoExpParameters,
                                                        scale_locked=scale_locked, skew_locked=skew_locked,
                                                        log=True, bkg=True)

        else:
            raise Exception("Model not implemented! Please add appropriate subclasses in parameters.py and word.py,"
                                "then come back!")

        parlist = postmax._extract_params(log=False)

        for i,p in enumerate(parlist):
            print('posterior maximum for parameter ' + str(i) + ': ' + str(p))



        return quants, postmax


    @staticmethod
    def plot_mcmc(sample, plotname, plotlabels=None):
            if plotlabels == None:
                plotlabels = ['bla' for s in range(np.shape(sample)[1])]

            #try:
            #    assert np.shape(sample)[1] > np.shape(sample)[0]
            #except AssertionError:
            #    sample = np.transpose(sample)
            #    sample = np.transpose(sample)

            #print('shape sample: ' + str(np.shape(sample)))

            figure = triangle.corner(sample, labels= [p for p in plotlabels])
            figure.savefig(plotname + ".png")
            plt.close()
            return

    def _quantiles(self, sample, ncomp, interval=0.9, scale_locked=False, skew_locked=False, log=True, bkg=True):

            all_intervals = [0.5-interval/2.0, 0.5, 0.5+interval/2.0]

            ### empty lists for quantiles
            ci_lower, cmean, ci_upper = [], [], []

            try:
                assert np.shape(sample)[1] > np.shape(sample)[0]
            except AssertionError:
                sample = np.transpose(sample)
            except IndexError:
                #print("Single-dimension array")
                q = quantiles(sample, all_intervals)
                ci_lower = q[0]
                ci_median = q[1]
                ci_upper = q[2]

            ### loop over the parameters ###
            for i,k in enumerate(sample):

                #print("I am on parameter: " + str(i))

                q = quantiles(k, all_intervals)

                ci_lower.append(q[0])
                cmean.append(q[1])
                ci_upper.append(q[2])

            if self.model is word.TwoExp:

                ci_lower = parameters.TwoExpCombined(ci_lower, ncomp, parclass=parameters.TwoExpParameters,
                                                        scale_locked=scale_locked, skew_locked=skew_locked,
                                                        log=True, bkg=True)
                ci_median = parameters.TwoExpCombined(ci_median, ncomp, parclass=parameters.TwoExpParameters,
                                                        scale_locked=scale_locked, skew_locked=skew_locked,
                                                        log=True, bkg=True)

                ci_upper = parameters.TwoExpCombined(ci_upper, ncomp, parclass=parameters.TwoExpParameters,
                                                        scale_locked=scale_locked, skew_locked=skew_locked,
                                                        log=True, bkg=True)

            else:
                raise Exception("Model not implemented! Please add appropriate subclasses in parameters.py and word.py,"
                                "then come back!")
            quants = [ci_lower, ci_median, ci_upper]

            return quants




    def find_spikes(self, model=word.TwoExp, nmax = 10, nwalker=500, niter=100, burnin=200, namestr='test', \
                    scale_locked=False, skew_locked=False):


            all_burstdict = []
            all_sampler = []
            all_means, all_err = [], []
            all_theta_init = []
            all_quants, all_postmax= [], []

            theta_init = [np.log(np.mean(self.counts))]
            bm = BurstDict(self.times, self.counts, [])


            
            #print('k = 0, theta_init : ' + str(burstmodel.wordobject._exp(theta_init)))

            sampler = self.mcmc(bm, theta_init, niter=niter, nwalker=nwalker, burnin=burnin,
                                scale_locked=scale_locked, skew_locked=skew_locked, plot=True, log=True, bkg=True,
                                plotname=namestr + '_k0_posteriors')

            postmean = np.mean(sampler.flatchain, axis=0)
            posterr = np.std(sampler.flatchain, axis=0)
            quants, postmax = self.find_postmax(sampler)

            postmean = parameters.TwoExpCombined(postmean, 0, parclass=parameters.TwoExpParameters,
                                                    scale_locked=scale_locked, skew_locked=skew_locked,
                                                    log=True, bkg=True)

            bm.plot_model(postmean, postmax=postmax, plotname=namestr + '_k' + str(0))

            #all_sampler.append(sampler.flatchain[-50000:])
            all_means.append(postmean)
            all_err.append(posterr)
            all_quants.append(quants)
            all_postmax.append(postmax)
            all_burstdict.append(bm)
            all_theta_init.append(theta_init)
            print('posterior means, k = 0: ')
            print(' --- background parameter: ' + str(postmean[0]) + ' +/- ' +  str(np.exp(posterr[0])) + "\n")

            all_results = {'sampler': sampler, 'means': postmean, 'err': posterr, 'quants': quants, 'max': postmax,
                           'init': theta_init}

            sampler_file = open(namestr + '_k0_posterior.dat','w')
            pickle.dump(all_results, sampler_file)
            sampler_file.close()


            ### test change for pushing to bitbucket
            for n in np.arange(nmax)+1:
                ## define burst model
                old_postmeans = all_means[-1]
                old_burstdict = all_burstdict[-1]


                lpost = WordPosterior(self.times, self.counts, bm, scale_locked=scale_locked,
                                      skew_locked=skew_locked, log=log, bkg=bkg)
#                if scale_locked and not skew_locked and n>1:
#                    lpost = WordPosteriorSameScale(self.times, self.counts, burstmodel)
#                    old_means = lpost._insert_scale(old_postmeans)
#                elif scale_locked and skew_locked and n > 1:
#                    lpost = WordPosteriorSameScaleSameSkew(self.times, self.counts, burstmodel)
#                    old_means = lpost._insert_params(old_postmeans)
#                else:
#                    lpost = WordPosterior(self.times, self.counts, burstmodel)
#                    old_means = old_postmeans


                ## extract posterior means from last model run

                model_counts = old_burstdict.model_means(old_postmeans, nbins=10)

                datamodel_ratio = self.counts/model_counts
                max_diff = max(datamodel_ratio)
                max_ind = np.argmax(datamodel_ratio)
                max_loc = self.times[max_ind]
                print('max_loc:' + str(max_loc))

                wordlist = [model for m in range(n)]
                bm = BurstDict(self.times, self.counts, wordlist)


                if model == word.TwoExp:

                    new_event_time = max_loc
                    new_word = [new_event_time]
                    new_amp = np.log(max_diff)
                    if not hasattr(old_postmeans, "scale"):
                        new_scale = np.log(0.1*self.T)
                        new_word.append(new_scale)
                    new_word.append(new_amp)
                    if not hasattr(old_postmeans, "skew"):
                        new_skew = np.log(1.0)
                        new_word.append(new_skew)

                    old_postmeans._add_word(new_word)

                    theta_init = old_postmeans._extract_params(log=True)

                theta_init = np.array(theta_init)

                #print('n = ' + str(n) + ', theta_init = ' + str(theta_init))
                sampler = self.mcmc(bm, theta_init, niter=niter, nwalker=nwalker, burnin=burnin,
                                    scale_locked=scale_locked, skew_locked=skew_locked, plot=True, log=True, bkg=True,
                                    plotname=namestr + '_k' + str(n) + '_posteriors')

                postmean = np.mean(sampler.flatchain, axis=0)
                posterr = np.std(sampler.flatchain, axis=0)
                quants, postmax = self.find_postmax(sampler)


                print('Posterior means, k = ' + str(n) + ': ')
                for i,(p,e) in enumerate(zip(postmean, posterr)):
                    print('--- parameter ' + str(i) + ': ' + str(p) + ' +/- ' + str(e))
                    #if i == 0:
                    #    print(' --- parameter ' + str(i) + ': ' + str(p) + ' +/- ' +  str(e))
                    #if i == len(postmean)-1:
                    #    print(' --- parameter ' + str(i) + ': ' + str(np.exp(p)) + ' +/- ' +  str(np.exp(e)) + "\n")
                    #else:
                    #    print(' --- parameter ' + str(i) + ': ' + str(np.exp(p)) + ' +/- ' +  str(np.exp(e)))


#                if scale_locked and not skew_locked:# and n>1:
#                    lpost = WordPosteriorSameScale(self.times, self.counts, burstmodel)
#                    new_postmean = lpost._insert_scale(postmean)
#                    new_postmax = lpost._insert_scale(postmax)
#                elif scale_locked and skew_locked:# and n > 1:
#                    lpost = WordPosteriorSameScaleSameSkew(self.times, self.counts, burstmodel)
#                    new_postmean = lpost._insert_params(postmean)
#                    new_postmax = lpost._insert_params(postmax)
#                else:
#                    lpost = WordPosterior(self.times, self.counts, burstmodel)
#                    new_postmean = postmean
#                    new_postmax = postmax


                if model is word.TwoExp:
                    postmean = parameters.TwoExpCombined(postmean, n, parclass=parameters.TwoExpParameters,
                                                        scale_locked=scale_locked, skew_locked=skew_locked,
                                                        log=True, bkg=True)
                else:
                    raise Exception("Model not known! At the moment, only word.TwoExp is implemented!")

                bm.plot_model(postmean, postmax=postmax, plotname=namestr + '_k' + str(n))

                #all_sampler.append(sampler.flatchain[-50000:])
                all_means.append(postmean)
                all_err.append(posterr)
                all_quants.append(quants)
                all_postmax.append(postmax)
                all_burstdict.append(bm)
                all_theta_init.append(theta_init)

                all_results = {'sampler': sampler, 'means': postmean, 'err': posterr, 'quants': quants, 'max': postmax,
                            'init':theta_init}

                sampler_file= open(namestr + '_k' + str(n) + '_posterior.dat', 'w')
                pickle.dump(all_results, sampler_file)
                sampler_file.close()



            return all_means, all_err, all_postmax, all_quants, all_theta_init

                ## now I need to: return count rate from previous model
                ## then find highest data/model outlier
                ## place new initial guess there, + small deviation in all paras?
                ## run mcmc
                ## append new posterior solution to old one 
    
    @staticmethod
    def plot_quants(postmax, all_quants, model=word.TwoExp, namestr='test', log=True):

        if model is word.TwoExp:
            parnames = ["t0", "log_scale", "log_amp", "log_skew"]

        max_words = len(postmax[-1].all)

        for n in xrange(postmax[1].all[0].npar):

            par_all = []
            plt.figure()
            ymin, ymax = [], []
            for i,(p,ci,cu) in enumerate(zip(all_quants[:,0], all_quants[:,1], all_quants[:,2])):
                par_temp = [a.__dict__[parnames[n]] for a in p.all]
                lower_temp = [a.__dict__[parnames[0]] for a in ci.all]
                upper_temp = [a.__dict__[parnames[0]] for a in cu.all]
                plt.errorbar(np.arange(len(par_temp))+i, par_temp, yerr=[par_temp-lower_temp, upper_temp-par_temp],
                             fmt="--o", lw=2, label="spike " + str(i), color=cm.hsv(i*30))
                ymin.append(np.min(lower_temp))
                ymax.append(np.max(upper_temp))
            plt.axis([0, np.arange(max_words)+5, np.min(ymin), np.max(ymax)])
            plt.legend()
            plt.xlabel("Number of spikes in the model", fontsize=16)
            plt.ylabel(postmax[1].all[0].parnames[n])
            plt.savefig(namestr + "_" + str(parnames[n]) + ".png", format="png")
            plt.close()



        ### I AM HERE!

#        npar = model.npar
#        nspikes = len(postmax)-1
#        allmax = np.zeros((nspikes, nspikes*npar+1))
#        all_cl = np.zeros((nspikes, nspikes*npar+1))
#        all_cu = np.zeros((nspikes, nspikes*npar+1))

#        for i,(p,q) in enumerate(zip(postmax[1:], all_quants[1:])):
#            allmax[i,:len(p)-1] = p[:-1]
#            all_cl[i,:len(p)-1] = q['lower ci'][:-1]
#            all_cu[i,:len(p)-1] = q['upper ci'][:-1]

#        for n in xrange(npar):
#            fig = plt.figure()
#            ## I AM HERE
#            ymin, ymax = [], []
#            for s in xrange(nspikes):
#                #print(allmax[s:, n+s*npar])
#                ymin.append(np.min(all_cl[s:,n+s*npar]))
#                ymax.append(np.max(all_cu[s:,n+s*npar]))
#                plt.errorbar(np.arange(nspikes-s)+s+1.0+0.1*s, allmax[s:, n+s*npar],
#                             yerr=[allmax[s:, n+s*npar]- all_cl[s:,n+s*npar],all_cu[s:,n+s*npar]-allmax[s:, n+s*npar]],
#                             fmt='--o', lw=2, label="spike " + str(s), color=cm.hsv(s*30))
#            plt.axis([0.0, nspikes+5, min(ymin), max(ymax)])
#            plt.legend()
#            plt.xlabel("Number of spikes in the model", fontsize=16)
#            plt.ylabel(model.parnames[n], fontsize="16")
#            plt.savefig(namestr + '_par' + str(n) + '.png', format='png')
#            plt.close()

        return

    def _exp_all(self, all_means, model=word.TwoExp):
        all_means_exp = []
        for i,a in enumerate(all_means):
            if i == 0:
                means_exp = np.exp(a)
                all_means_exp.append(means_exp)
            else:
                wordlist = [model for m in range(i)]
                #print('wordlist: ' + str(wordlist))
                w = word.CombinedWords(self.times, wordlist)
                means_pack = w._pack(a)
                means_exp = w._exp(means_pack)
                all_means_exp.append(w._unpack(means_exp))

        return all_means_exp




def main():

    if len(filenames) == 0:
        raise Exception("No files in directory!")

    for f in filenames:
        filecomponents = f.split("/")
        fname = filecomponents[-1]
        froot = fname[:-9]


        if instrument == 'gbm':
            times, counts = read_gbm_lightcurves(f)
            ### check for saturation
            tres = times[1] - times[0]
            countrate = counts/np.float(tres)
            if np.max(countrate) >= saturation_countrate:
                print("Burst is saturated. Excluding ...")
                continue
        else:
            raise Exception("Instrument not known!")



        bm = BurstModel(times, counts)

        all_means, all_err, all_postmax, all_quants, all_theta_init = \
            bm.find_spikes(nmax=10, nwalker=500, niter=200, burnin=200, namestr=froot)


        bm.plot_quants(all_postmax, all_quants, namestr=froot)

        #posterior_dict = {'samples':all_sampler, 'means':all_means, 'err':all_err, 'quants':all_quants,
        #                 'theta_init':all_theta_init}

        #posterior_file = open(froot + '_posteriors.dat', 'w')
        #pickle.dump(posterior_dict, posterior_file)
        #posterior_file.close()

    return




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model magnetar bursts with spikes!')

    modechoice = parser.add_mutually_exclusive_group(required = True)
    modechoice.add_argument('-a', '--all', action='store_true', dest='all', help='run on all files in the directory')
    modechoice.add_argument('-s', '--single', action='store_true', dest='single', help='run on a single file')

    parser.add_argument('-w', '--nwalker', action='store', dest='nwalker', required=False,
                        type=int, default=500, help='Number of emcee walkers')
    parser.add_argument('-i', '--niter', action="store", dest='niter', required=False,
                        type=int, default=200, help='number of emcee iterations')
    parser.add_argument('--instrument', action='store', dest='instrument', default='gbm', required=False,
                        help = "Instrument data was taken with")


    singleparser = parser.add_argument_group('single file', 'options for running script on a single file')
    singleparser.add_argument('-f', '--filename', action='store', dest='filename', help='file with data')

    allparser = parser.add_argument_group('all bursts', 'options for running script on all bursts')
    allparser.add_argument("-d", "--dir", dest="dir", action="store", default='./', help='directory with data files')


    clargs = parser.parse_args()

    nwalker = int(clargs.nwalker)
    niter = int(clargs.niter)

    if clargs.single and not clargs.all:
        mode = 'single'
        filenames = [clargs.filename]

    elif clargs.all and not clargs.single:
        mode = 'all'
        if not clargs.dir[-1] == "/":
            clargs.dir = clargs.dir + "/"
        filenames = glob.glob(clargs.dir + '*_data.dat')


    if clargs.instrument.lower() in ["fermi", "fermigbm", "gbm"]:
        instrument="gbm"
        bid_index = 9
        bst_index = [10, 17]
    elif clargs.instrument.lower() in ["rxte", "rossi", "xte"]:
        instrument="rxte"

    else:
        print("Instrument not recognised! Using filename as root")
        instrument=None

    main()





#############################################################################################################
#############################################################################################################
#############################################################################################################


#######################################################################
#### OLD IMPLEMENTATION! ##############################################
#######################################################################

#class DictPosterior(object):

#    """
#    note: implicit assumption is that array times is equally spaced
#          nbins_data: number of bins in data
#          nbins: multiplicative factor for model light curve bins
#    """
#    def __init__(self, times, counts, wordmodel, nbins=10):
#        self.times = times
#        self.counts = counts
#        self.model = wordmodel
        #self.npar = npar

#        self.Delta = times[1]-times[0]
#        self.nbins_data = len(times)
#        self.nbins = nbins



#    def logprior(self, theta):

        # Our prior for a SINGLE WORD
        # Input: parameter vector, which gets unpacked into named things
        # feel free to change the order if that's how you defined it - BJB

#        saturation_countrate = 3.5e5 ### in counts/s
#        T = self.times[-1] - self.times[0]
    
 
#        skew, bkg, scale, theta_evt = unpack(theta)

#        if  scale < self.Delta or scale > T or skew < np.exp(-1.5) or skew > np.exp(3.0) or \
#                bkg < 0 or bkg > saturation_countrate:
#            return -np.Inf

#        all_event_times = theta_evt[:,0]
#        all_amp = theta_evt[:,1]
#        if np.min(all_event_times) < self.times[0] or np.max(all_event_times) > self.times[-1] or \
#                np.min(all_amp) < 0.1/self.Delta or np.max(all_amp) > saturation_countrate:
#            return -np.Inf

#        return 0.


    ## theta = [scale, skew, move1, amp1, move2, amp2]    
#    def loglike(self, theta):
  

        ### unpack theta:
#        skew, bkg, scale, theta_evt = unpack(theta)
#        lambdas = model_means(self.Delta, self.nbins_data, skew, bkg, scale, theta_evt, nbins=self.nbins)

#        return log_likelihood(lambdas, self.counts)

    ## change parameters:
    

#    def logposterior(self, theta):
#        return self.logprior(theta) + self.loglike(theta)


#    def __call__(self, theta):
#        return self.logposterior(theta)




#### DEPRECATED: USE WORD CLASS INSTEAD #####
#def word(time, scale, skew = 2.0):

#    t = np.array(time)/scale
#    y = np.zeros_like(t)
#    y[t<=0] = np.exp(t[t<=0])
#    y[t>0] = np.exp(-t[t>0]/skew)

#    return y
#######

#def event_rate(time, event_time, scale, amp, skew):

    #x = x - theta[0]

#    time = time - event_time
#    counts = amp*word(time, scale, skew)

#    return counts

### note: theta_all is a numpy array of n by m, 
### where n is the number of peaks, and m is the number of parameters
### per peak
#def model_means(Delta, nbins_data, skew, bkg, scale, theta_evt, nbins=10):

#    delta = Delta/nbins
#    nsmall = nbins_data*nbins
#    time_small = np.arange(nsmall)*delta
#    rate_small = np.zeros(nsmall)


#    for (event_time, amp) in theta_evt:

#        rate_temp = event_rate(time_small, event_time, scale, amp, skew)
#        rate_small = rate_small + rate_temp

#    nrow = len(rate_small)/nbins
#    rate_map = rate_small.reshape(nrow, nbins)
#    rate_map_sum = np.sum(rate_map, axis=1)*delta

#    rate_map_all = rate_map_sum + bkg*Delta

#    return rate_map_all

## go from numpy array weird shape
#def unpack(theta):
#    """
#    unpacks the numpy array of parameters into a form that model_means can read.

#    returns: skew, bkg, scale, theta_evt

#    theta_evt is an array of npeaks by 2, where npeaks is the number of peaks
#    each row is (event_time, amp)
#    """


#    skew = np.exp(theta[0])
#    bkg = np.exp(theta[1])
#    scale = np.exp(theta[2])


#    theta = np.array(theta)
#    theta_evt = copy.copy(theta[3:]).reshape((len(theta)-3)/2, 2)

#    for i in range(len(theta_evt)):
#        theta_evt[i][1] = np.exp(theta_evt[i][1])

#    return skew, bkg, scale, theta_evt


## go from weird shape to numpy array
#def pack(skew, bkg, scale, theta_evt):

#    theta = np.zeros(len(theta_evt.flatten())+3)
#    theta[0] = np.log(skew)
#    theta[1] = np.log(bkg)
#    theta[2] = np.log(scale)

#    for i in range(len(theta_evt)):
#        theta_evt[i][1] = np.log(theta_evt[i][1])

#    theta[3:] = theta_evt.flatten()

#    return theta


# noinspection PyNoneFunctionAssignment
#def model_burst():

#    times = np.arange(1000.0)/10000.0
#    counts = np.random.poisson(2000.0, size=len(times))

#    skew = 3.0
#    scale = 0.005
#    bkg = 5.0

#    nspikes = 2
#    theta_evt = np.zeros((nspikes,2))

#    for i in range(nspikes):
#        theta_evt[i] = [np.random.rand()*times[-1], 10.0/(times[1]-times[0])]

#    theta = pack(skew, bkg, scale, theta_evt)

#    return theta

#def initial_guess(times, counts, skew, bkg, scale, theta_evt):

#    times = times - times[0]

    ### initialise guess for burst 090122218, tstart = 47.4096 
    #skew = 5.0
    #bkg = 2000.0
    #scale = 0.01
    #theta_evt = np.array([[0.82, 30000], [0.87, 60000], [0.95, 50000], [1.05, 20000]])


#    Delta = times[1]-times[0]
#    nbins_data = len(times)

#    counts_model = model_means(Delta, nbins_data, skew, bkg, scale, theta_evt, nbins=10)

#    figure()
#    plt.plot(times, counts, 'k')
#    plt.plot(times, counts_model, 'r')
#    plt.xlabel('Time [s]', fontsize=18)
#    plt.ylabel('Counts per bin', fontsize=18)
#    plt.title('Light curve with initial guess for model', fontsize=18)

#    theta = pack(skew, bkg, scale, np.array(theta_evt))

#    return theta, counts_model


### put in burst times array and counts array
#def test_burst(times, counts, theta_guess, namestr = 'testburst', nwalker=32):

#    skew, bkg, scale, theta_evt = unpack(theta_guess)

    #Delta = times[1]-times[0]
    #nbins_data = len(times)
 
#    theta, counts_model = initial_guess(times, counts, skew, bkg, scale, theta_evt)

#    figure()
#    plt.plot(times, counts, 'k')
#    plt.plot(times, counts_model, 'r')
#    plt.xlabel('Time [s]', fontsize=18)
#    plt.ylabel('Counts per bin', fontsize=18)
#    plt.title('Light curve with initial guess for model', fontsize=18)
#    plt.savefig(namestr + '_initialguess.png', format='png')
#    plt.close()

#    theta = pack(skew, bkg, scale, theta_evt)

#    lpost = DictPosterior(times, counts)
   
#    if nwalker < 2*len(theta):
#        nwalker = 2*len(theta)

#    p0 = [theta+np.random.rand(len(theta))*1.0e-3 for t in range(nwalker)]

#    sampler = emcee.EnsembleSampler(nwalker, len(theta), lpost)
#    pos, prob, state = sampler.run_mcmc(p0, 200)
#    sampler.reset()
#    sampler.run_mcmc(pos, 1000, rstate0 = state)

#    plot_test(times, counts, sampler.flatchain[-10:])
#    plt.xlabel('Time [s]', fontsize=18)
#    plt.ylabel('Counts per bin', fontsize=18)
#    plt.title('Light curve with draws from posterior sample', fontsize=18)
#    plt.savefig(namestr + '_posteriorsample.png', format='png')
#    plt.close()


    ### quick hack to save emcee sampler to disc
#    f = open(namestr + '_sampler.dat', 'w')
#    pickle.dump(sampler, f)
#    f.close()

#    return


#def plot_test(times, counts, theta):

#    plt.plot(times, counts, 'k')


#    Delta = times[1]-times[0]
#    nbins_data = len(times)


#    for t in np.atleast_2d(theta):

#        skew, bkg, scale, theta_evt = unpack(t)
#        counts_model = model_means(Delta, nbins_data, skew, bkg, scale, theta_evt, nbins=10)
#        plt.plot(times, counts_model, 'r')

#    return

# Poisson log likelihood based on a set of rates
# log[ prod exp(-lamb)*lamb^x/x! ]
# exp(-lamb)

### lambdas: numpy array of Poisson rates: mean expected integrated in a bin
#import scipy.special
#def log_likelihood(lambdas, data):

#    return -np.sum(lambdas) + np.sum(data*np.log(lambdas))\
#        -np.sum(scipy.special.gammaln(data + 1))


#def conversion(filename):
#    f=open(filename, 'r')
#    output_lists=defaultdict(list)
#    for line in f:
#        if not line.startswith('#'):
#             line=[value for value in line.split()]
#             for col, data in enumerate(line):
#                 output_lists[col].append(data)
#    return output_lists

##def main():

#    data = conversion(filename)
#    times = np.array([float(t) for t in data[0]])
#    counts = np.array([float(c) for c in data[1]])

#    test_burst(times, counts, namestr = namestr, nwalker=nwalkers)
   
#    return


#if __name__ == '__main__':

#    parser = argparse.ArgumentParser(description='Script to play around with Fermi/GBM magnetar data')
#    parser.add_argument('-f', '--filename', action='store', dest ='filename', help='input filename')
#    parser.add_argument('-n', '--namestr', action='store', dest='namestr', help='Output filename string')
#    parser.add_argument('--nwalkers', action='store', default='32', help='Number of emcee walkers')
 

#    clargs = parser.parse_args()
    
#    nwalkers = int(clargs.nwalkers)
#    filename = clargs.filename
#    namestr = clargs.namestr

#    main()
