import glob
import numpy as np

import parameters
import word
import dnest_sample
import run_dnest
### Simulated light curves

def singlepeak(bkg=None, datadir="./", trigfile="sgr1550_ttrig.dat"):

    """
    Makes simulations of light curves with peaks from the model we consider (exponential rise/decay with skewness
    parameter).

    NOTE: if bkg = None, then an estimate of the background is read from the sample.
    THIS REQUIRES FILES OF TYPE "%s*posterior*"%datadir and a file with trigger data (e.g. trigfile="sgr1550_ttrig.dat")
    TO WORK! Also, you need to have set your directory with the posterior and trigger file correctly!


    :param bkg:
    :return:
    """


    ## length of light curve
    tseg = 0.2
    dt = 0.0005
    nsims=100

    ### time stamps of time bins for a light curve of length 0.4 seconds with bin size dt
    times = np.arange(0.0, tseg, dt)

    ### if bkg keyword is not given, extract from sample
    if bkg is None:
        ### extract estimate of Fermi/GBM background counts/bin for sample
        pars, bids =  dnest_sample.extract_sample(datadir=datadir, nsims=nsims, prior="exp", trigfile=trigfile)
        bkg_sample = []
        for i in xrange(nsims):
            sample = pars[:,i]
            bkg= np.array([s.bkg for s in sample])
            bkg_sample.extend(bkg)

        ### make histogram of background parameter estimates
        n, bins = np.histogram(np.log10(bkg_sample), bins=100, range=[-5.0, np.log10(5.0)])

        ### find index of maximum of histogram
        b_index = np.where(n == np.max(n))[0]

        ### compute actual value for the background at the maximum of the model distribution
        bmax = bins[b_index] + (bins[b_index+1] - bins[b_index])/2.0
        bkg = 10.0**bmax

    ###### Now make light curves with single peak #########

    amp_all = [1,5,10,50,100]
    t0 = 0.06
    tau_rise = 0.005
    skew = 5.0

    print("background: " + str(bkg))

    for a in amp_all:
        p = parameters.TwoExpParameters(t0, tau_rise, a, skew, bkg=bkg, log=False)
        model_counts = word.TwoExp(times).model(p)
        poisson_counts = np.array([np.random.poisson(c) for c in model_counts])

        filename = "%sonespike_a=%i_data.txt"%(datadir,a)

        np.savetxt(filename, np.transpose(np.array([times, poisson_counts])))

    return




def make_sims(datadir="./"):

    ###estimate for Fermi/GBM sample
    log_bkg = 0.18606270394577695
    bkg = 10.0**log_bkg

    ### simulations with single peak
    singlepeak(datadir=datadir, bkg=bkg)

    return

def run_sims(datadir="../data/", dnest_dir="./", key="onespike", nsims=500):

    files = glob.glob("%s*%s*_data.txt"%(datadir,key))
    print("files: " + str(files))
    for f in files:
        run_dnest.run_burst(f, dnest_dir=dnest_dir, nsims=nsims)

    return

def main():

    ## assume I'm running this code from the dnest directory in magnetron, and that the simulation files
    ## are in folder magnetron/data
    datadir = "../data/"
    dnest_dir = "./"

    ### make simulations
    make_sims(datadir=datadir)

    ### run simulations with one spike
    run_sims(datadir=datadir, dnest_dir=dnest_dir, key="onespike", nsims=250)

    return


if __name__ == "__main__":
    main()
