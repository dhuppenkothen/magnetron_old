import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
#from pylab import *
import matplotlib.cm as cm


import numpy as np
import glob
import argparse
import cPickle as pickle

import word
import burstmodel
import parameters

##### GET DATA FROM PICKLED PYTHON OBJECT (FROM PROCESSING PIPELINE)
#
# Pickling data is a really easy way to store data in binary format.
# This function reads back pickled data and stores it in memory.
#
#
def getpickle(picklefile):
    file = open(picklefile, 'r')
    procdata = pickle.load(file)
    return procdata
########################################################################






class WordPosteriorSample(object):

    def __init__(self, filename):

        self.file = filename
        self.bid, self.bst, self.k = self.split_filename()
        return

    def split_filename(self):
        fdata = self.file.split("_")
        bid = fdata[0]
        bst = np.float(fdata[1])
        if fdata[2][-2] == "k":
            k = np.float(fdata[2][-1])
        else:
            k = np.float(fdata[2][-2:])
        return bid, bst, k

    def read_from_file(self):
        data = getpickle(self.file)
        samples = data["sampler"]
        postmax = data["max"]
        return samples, postmax


class AllPosteriorSamples(object):

    def __init__(self, all_filenames, bid, bst):
        self.bid = bid
        self.bst = bst

        self.all_models = self.all_words(all_filenames)

        return

    def read_data(self, dir="./"):

        filename = str(dir) + self.bid + "_" + self.bst + "_data.dat"
        print('filename: ' + str(filename))

        self.times, self.counts = burstmodel.read_gbm_lightcurves(filename)

        return


    def all_words(self, filenames):

        ### worst list comprehension ever
        all_models = [WordPosteriorSample(f) for f in filenames if \
                     f.split("_")[0] == self.bid and f.split("_")[1] == self.bst]


        all_models_sorted = sorted(all_models, key=lambda models: models.k)
        #print("a: " + str([a.k for a in all_models_sorted]))

        return all_models_sorted

    def samples(self):

        all_data = [a.read_from_file() for a in self.all_models]
        all_samples = [a[0] for a in all_data]
        all_postmax = [a[1] for a in all_data]

        return all_samples, all_postmax

    def quants(self, samples, interval = 0.9, scale_locked=False, skew_locked=False,
               log = True, bkg = True):

        print(scale_locked)
        print(skew_locked)
        all_quants = []
        #print('samples: ' + str(samples))
        for i,s in enumerate(samples):

            if np.shape(s)[0] > np.shape(s)[1]:
                s = np.transpose(s)

            quants = self.bm._quantiles(s, i, scale_locked=scale_locked, skew_locked=skew_locked,
                                        log=log, bkg=bkg)
            all_quants.append(quants)

        return all_quants


#### DEPRECATED! USE bm.plot_results INSTEAD!
    def plot_model(self, samples, postmax = None, nsamples = 1000, scale_locked=False, skew_locked=False,
                   model = word.TwoExp):


        npar = model.npar
        npar_add = 1
        if scale_locked:
            npar -= 1
            npar_add += 1
        if skew_locked:
            npar -= 1
            npar_add += 1

        npar_samples = min(np.shape(samples))
        npar_words = npar_samples - npar_add
        nwords = npar_words/npar


        assert hasattr(self, "times"), "times not defined!"
        assert hasattr(self, "counts"), "counts not defined!"

        bm = burstmodel.BurstDict(self.times, self.counts, [model for m in xrange(nwords)])

        if scale_locked and not skew_locked:
            lpost = burstmodel.WordPosteriorSameScale(self.times, self.counts, bm)

        elif scale_locked and skew_locked:
            lpost = burstmodel.WordPosteriorSameScaleSameSkew(self.times, self.counts, bm)

        else:
            lpost = burstmodel.WordPosterior(self.times, self.counts, bm)


        #samples = np.transpose(samples)
        #print('samples shape: ' + str(np.shape(samples)))
        #print("samples[0]: " + str(samples[0]))


        len_samples = np.arange(np.max(np.shape(samples)))
        sample_ind = np.random.choice(len_samples, size=nsamples, replace=False)

        all_model_counts = []

        for j,i in enumerate(sample_ind):
            #print("i = " + str(j))
            theta_temp = samples[i]
            #print("theta_temp: " + str(theta_temp))
            if scale_locked and not skew_locked:
                #print("I am in scale_locked")
                theta_new = lpost._insert_scale(theta_temp)
                theta_new = np.array(theta_new).flatten()
            elif scale_locked and skew_locked:
                #print("I am in skew_locked")
                theta_new = lpost._insert_params(theta_temp)
                theta_new = np.array(theta_new).flatten()
            else:
                #print('I am nowhere useful!')
                theta_new = theta_temp
                theta_new = np.array(theta_new).flatten()

            #print("theta_new: " + str(theta_new))
            model_counts = bm.model_means(theta_new)

            all_model_counts.append(model_counts)


        if not postmax == None:
            if scale_locked and not skew_locked:
                postmax_new = lpost._insert_scale(postmax)
            elif scale_locked and skew_locked:
                postmax_new = lpost._insert_params(postmax)
            else:
                postmax_new = postmax
            #print("postmax_new: " + str(postmax_new))
            postmax_counts = bm.model_means(postmax_new)



        mean_counts = np.mean(all_model_counts, axis=0)
        model_counts_cl, model_counts_median, model_counts_cu = [], [], []

        all_model_counts = np.transpose(all_model_counts)
        for a in all_model_counts:
            quants = burstmodel.BurstModel._quantiles(a)
            model_counts_cl.append(quants['lower ci'])
            model_counts_median.append(quants['mean'])
            model_counts_cu.append(quants['upper ci'])



        fig = plt.figure(figsize=(10,8))
        #print("len times: " + str(len(self.times)))
        #print("len counts: " + str(len(self.counts)))
        #print("len mean counts: " + str(len(mean_counts)))
        #print("len model counts cl: " + str(len(model_counts_cl)))
        #print("len model counts cu: " + str(len(model_counts_cu)))
        plt.plot(self.times, self.counts, lw=1, color='black', label='input data')
        plt.plot(self.times, mean_counts, lw=2, color='darkred', label='model light curve: mean of posterior sample')
        plt.plot(self.times, model_counts_cl, lw=0.8, color='darkred')
        plt.plot(self.times, model_counts_cu, lw=0.8, color='darkred')
        plt.fill_between(self.times, model_counts_cl, model_counts_cu, color="red", alpha=0.3)
        if not postmax == None:
            plt.plot(self.times, postmax_counts, lw=2, color='blue', label='model light curve: posterior max')
        plt.legend()
        plt.xlabel('Time [s]', fontsize=18)
        plt.ylabel('Count Rate [counts/bin]', fontsize=18)
        plt.title('An awesome model light curve!')
        plt.savefig(self.bid + "_" + str(self.bst) + '_k' + str(nwords) + '_lc.png', format='png')
        plt.close()
        return


    def plot_quants(self, quants, model=word.TwoExp, scale_locked=False, skew_locked=False):

        postmedian = [q['mean'] for q in quants]

        npar = model.npar
        nspikes = len(postmedian)-1
        npar_add = 0
        if scale_locked:
            npar = npar - 1
            npar_add += 1
        if skew_locked:
            npar = npar - 1
            npar_add += 1

        #print("npar: " + str(npar))

        allmax = np.zeros((nspikes, nspikes*npar+npar_add))
        all_cl = np.zeros((nspikes, nspikes*npar+npar_add))
        all_cu = np.zeros((nspikes, nspikes*npar+npar_add))

        #print("nspikes*npar+npar_add:  " + str(nspikes*npar+npar_add))


        for i,(p,q) in enumerate(zip(postmedian[1:], quants[1:])):
            allmax[i,:len(p)-1] = p[:-1]
            all_cl[i,:len(p)-1] = q['lower ci'][:-1]
            all_cu[i,:len(p)-1] = q['upper ci'][:-1]


        #print("allmax: " + str(allmax))
        #print("all_cl: " + str(all_cl))

        allmax_scale, all_cl_scale, all_cu_scale = [], [], []
        allmax_skew, all_cl_skew, all_cu_skew = [], [], []
        for n in xrange(npar):
            #print('n: ' + str(n))
            fig = plt.figure()
            ## I AM HERE
            ymin, ymax = [], []
            if scale_locked and n == 1:
                nplot = n+1
            else:
                nplot = n

            for s in xrange(nspikes):
                #print(allmax[s:, n+s*npar])
                if nplot == 1:
                    allmax_scale.append(allmax[s:, n+s*npar])
                    all_cl_scale.append(all_cl[s:,n+s*npar])
                    all_cu_scale.append(all_cu[s:,n+s*npar])
                if nplot == 3:
                    allmax_skew.append(allmax[s:, n+s*npar])
                    all_cl_skew.append(all_cl[s:,n+s*npar])
                    all_cu_skew.append(all_cu[s:,n+s*npar])

                ymin.append(np.min(all_cl[s:,n+s*npar]))
                ymax.append(np.max(all_cu[s:,n+s*npar]))
                plt.errorbar(np.arange(nspikes-s)+s+1.0+0.1*s, allmax[s:, n+s*npar],
                             yerr=[allmax[s:, n+s*npar]- all_cl[s:,n+s*npar],all_cu[s:,n+s*npar]-allmax[s:, n+s*npar]],
                             fmt='--o', lw=2, label="spike " + str(s), color=cm.hsv(s*30))
            plt.axis([0.0, nspikes+5, min(ymin), max(ymax)])
            plt.legend()
            plt.xlabel("Number of spikes in the model", fontsize=16)
            plt.ylabel(model.parnames[nplot], fontsize="16")
            plt.savefig(self.bid + "_" + str(self.bst) + '_par' + str(nplot) + '.png', format='png')
            plt.close()

        if scale_locked:
            #print('I am in scale_locked in plot!')
            allmax_scale = np.array([allmax[i,i*npar+npar_add+1] for i in xrange(nspikes)])
            all_cl_scale = np.array([all_cl[i,i*npar+npar_add+1] for i in xrange(nspikes)])
            all_cu_scale = np.array([all_cu[i,i*npar+npar_add+1] for i in xrange(nspikes)])
            fig = plt.figure()
            #print("allmax in scale_locked: " + str(allmax_scale))
            plt.errorbar(np.arange(nspikes)+1.0, allmax_scale,
                         yerr = [allmax_scale-all_cl_scale, all_cu_scale-allmax_scale], fmt="--o", lw=2,
                         label = "all spikes")
            plt.axis([0.0, nspikes, min(all_cl_scale), max(all_cu_scale)])
            plt.legend()
            plt.xlabel('Number of spikes in model', fontsize=16)
            plt.ylabel(model.parnames[1], fontsize='16')
            plt.savefig(self.bid + "_" + str(self.bst) + "_par1.png")
            plt.close()

        if skew_locked:
            #print('I am in skew_locked in plot!')
            if scale_locked:
                par_position = 0
            else:
                par_position = 1

            allmax_skew = np.array([allmax[i,i*npar+npar_add+par_position] for i in xrange(nspikes)])
            all_cl_skew = np.array([all_cl[i,i*npar+npar_add+par_position] for i in xrange(nspikes)])
            all_cu_skew = np.array([all_cu[i,i*npar+npar_add+par_position] for i in xrange(nspikes)])

            fig = plt.figure()
            #print("allmax in skew_locked: " + str(allmax_skew))
            plt.errorbar(np.arange(nspikes)+1.0, allmax_skew,
                         yerr = [allmax_skew-all_cl_skew, all_cu_skew-allmax_skew],
                         fmt="--o", lw=2, label = "all spikes")
            plt.axis([0.0, nspikes, min(all_cl_skew), max(all_cu_skew)])
            plt.legend()
            plt.xlabel('Number of spikes in model', fontsize=16)
            plt.ylabel(model.parnames[3], fontsize='16')
            plt.savefig(self.bid + "_" + str(self.bst) + "_par3.png")
            plt.close()
        return allmax_scale, all_cl_scale, all_cu_scale, allmax_skew, all_cl_skew, all_cu_skew




def plot_all_bursts(scale_locked = False, skew_locked = False):

    print('I am in plot_all_bursts')
    filenames = glob.glob("*posterior.dat")

    #print(filenames)
    bids = [f.split("_")[0] for f in filenames]
    bsts = [f.split("_")[1] for f in filenames]


    print(bids)
    print(bsts)

    bids = set(bids)
    bsts = list(set(bsts))

    print('bid: ' + str(bid))
    #if not bid is None:
    #    print('I am in bid is none')
    #    bids = [bid]

    print(bids)
    print(bsts)

    #print(filenames)

    scale_postmax, scale_cl, scale_cu = [], [], []
    skew_postmax, skew_cl, skew_cu = [], [], []

    for i in bids:
        for j in bsts:
            print('bid: ' + str(i))
            print("bst: " + str(j))
            burst = AllPosteriorSamples(filenames, i,j)
            if len(burst.all_models) == 0:
                continue
            else:
                burst.times, burst.counts = burstmodel.read_gbm_lightcurves(i + "_" + j + "_data.dat")
                burst.bm = burstmodel.BurstModel(burst.times, burst.counts)
                print('Extracting samples ...')
                samples, postmax = burst.samples()
                print("... done. Making quantiles ...")
                all_quants = burst.quants(samples, interval=0.9, scale_locked=scale_locked, skew_locked=skew_locked)
                print("... done. Now plotting parameters.")
                #print("scale_locked: " + str(scale_locked))
                #print("skew_locked: " + str(skew_locked))
                #print('bid: ' + str(burst.bid))
                #print('bst: ' + str(burst.bst))
                #print("bid after making quantiles: " + str(bid))
                #allmax_scale, all_cl_scale, all_cu_scale, allmax_skew, all_cl_skew, all_cu_skew = \
                #        burst.plot_quants(all_quants, scale_locked = scale_locked, skew_locked = skew_locked)

                burst.bm.plot_quants(postmax, all_quants, namestr=burst.bid + "_" + burst.bst + "_")
                #scale_postmax.append(allmax_scale)
                #skew_postmax.append(allmax_skew)
                #scale_cl.append(all_cl_scale)
                #scale_cu.append(all_cu_scale)
                #skew_cl.append(all_cl_skew)
                #skew_cu.append(all_cu_skew)
                #print("bid after plotting quantiles: " + str(bid))
                #print("Plotting light curves:")
                #burst.read_data(dir=data_dir + "/")
                #print("shape(samples): " + str(np.shape(samples)))
                #print("bid after reading in data: " + str(bid))
                for (s,p) in zip(samples[1:], postmax[1:]):
                    burst.bm.plot_results(s, postmax =p, nsamples = nsamples, scale_locked=scale_locked,
                                     skew_locked=skew_locked, model = word.TwoExp, bkg=True, log=True,
                                     namestr=i + "_" + j + "_")
                print("And all done! Hoorah!")
                print("bid at the end: " + str(bid))
    #all_limits = {'scale_max': scale_postmax, 'skew_max':skew_postmax, 'scale_cl':scale_cl, 'scale_cu':scale_cu,
    #              'skew_cl':skew_cl, 'skew_cu':skew_cu}
    #f = open('allbursts_postparas.dat', 'w')
    #pickle.dump(all_limits, f)
    #f.close()

    return

def main():
    print('I am in main!')
    plot_all_bursts(scale, skew)

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Model magnetar bursts with spikes!')

    parser.add_argument('--scale-locked', action='store_true', dest='scale', required=False,
                        help='Scale the same for all words?')
    parser.add_argument('--skew-locked', action='store_true', dest='skew', required=False,
                        help='Skew the same for all words?')

    parser.add_argument('-d', "--dir", action="store", dest="data_dir", required=False, default=".",
                        help="Directory where the data files are located")
    parser.add_argument('-b', "--bid", action="store", dest="bid", required=False, default="None",
                        help="Pick specific burst ID to run on")

    parser.add_argument('-n', '--nsamples', action="store", dest="nsamples", required=False, default=1000,
                        type=int, help="Number of samples to be used in average light curve.")

    clargs = parser.parse_args()
    scale = clargs.scale
    skew = clargs.skew
    data_dir = clargs.data_dir
    bid = clargs.bid
    nsamples = clargs.nsamples

    main()