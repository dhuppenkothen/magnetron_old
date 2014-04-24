
import shutil
import subprocess
import time as tsys
import numpy as np
import copy

import postprocess

def rewrite_main(filename, dnest_dir = "./"):

    mfile = open(dnest_dir+"main.cpp", "r")
    mdata = mfile.readlines()
    mfile.close()

    ## replace filename in appropriate line:
    mdata[-6] = '\tData::get_instance().load("%s");\n'%filename

    mfile.close()

    mwrite_file = open(dnest_dir+"main.cpp.tmp", "w")

    for l in mdata:
        mwrite_file.write(l)

    mwrite_file.close()

    shutil.move(dnest_dir+"main.cpp.tmp", dnest_dir+"main.cpp")

    return


def rewrite_options(nlevels=200, dnest_dir="./"):

    mfile = open(dnest_dir+"OPTIONS", "r")
    mdata = mfile.readlines()
    mfile.close()

    mdata[-4] = '%i\t# maximum number of levels\n'%nlevels

    mwrite_file = open(dnest_dir+"OPTIONS.tmp", "w")

    for l in mdata:
        mwrite_file.write(l)

    mwrite_file.close()

    shutil.move(dnest_dir+"OPTIONS.tmp", dnest_dir+"OPTIONS")

    return


def remake_model():

    tstart = tsys.clock()
    subprocess.call(["make"])
    tsys.sleep(15)
    tend = tsys.clock()

    return


def extract_nlevels(filename):

    fsplit = filename.split("_")

    sdata = np.loadtxt("%s_%s_samples.txt"%(fsplit[0], fsplit[1]))

    nlevels = np.shape(sdata)[0]

    return nlevels



def run_burst(filename, dnest_dir = "./"):

    ### first run: set levels to 200
    rewrite_main(filename, dnest_dir)
    rewrite_options(filename, nlevels=200, dnest_dir=dnest_dir)
    remake_model()

    ## run DNest
    dnest_process = subprocess.Popen("./main")


    endflag = False
    while endflag is False:
        continue



    return


def postprocess_new(temperature=1., numResampleLogX=1, plot=False):

    cut = 0

    levels = np.atleast_2d(np.loadtxt("levels.txt"))
    sample_info = np.atleast_2d(np.loadtxt("sample_info.txt"))
    sample = np.atleast_2d(np.loadtxt("sample.txt"))

    sample = sample[int(cut*sample.shape[0]):, :]
    sample_info = sample_info[int(cut*sample_info.shape[0]):, :]

    if sample.shape[0] != sample_info.shape[0]:
        print('# Size mismatch. Truncating...')
        lowest = np.min([sample.shape[0], sample_info.shape[0]])
        sample = sample[0:lowest, :]
        sample_info = sample_info[0:lowest, :]

    # Convert to lists of tuples
    logl_levels = [(levels[i,1], levels[i, 2]) for i in xrange(0, levels.shape[0])] # logl, tiebreaker
    logl_samples = [(sample_info[i, 1], sample_info[i, 2], i) for i in xrange(0, sample.shape[0])] # logl, tiebreaker, id
    logx_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logp_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logP_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    P_samples = np.zeros((sample_info.shape[0], numResampleLogX))
    logz_estimates = np.zeros((numResampleLogX, 1))
    H_estimates = np.zeros((numResampleLogX, 1))

    # Find sandwiching level for each sample
    sandwich = sample_info[:,0].copy().astype('int')
    for i in xrange(0, sample.shape[0]):
        while sandwich[i] < levels.shape[0]-1 and logl_samples[i] > logl_levels[sandwich[i] + 1]:
            sandwich[i] += 1



    for z in xrange(0, numResampleLogX):
        # For each level
        for i in range(0, levels.shape[0]):
            # Find the samples sandwiched by this level
            which = np.nonzero(sandwich == i)[0]
            logl_samples_thisLevel = [] # (logl, tieBreaker, ID)
            for j in xrange(0, len(which)):
                logl_samples_thisLevel.append(copy.deepcopy(logl_samples[which[j]]))
            logl_samples_thisLevel = sorted(logl_samples_thisLevel)
            N = len(logl_samples_thisLevel)

            # Generate intermediate logx values
            logx_max = levels[i, 0]
            if i == levels.shape[0]-1:
                logx_min = -1E300
            else:
                logx_min = levels[i+1, 0]
            Umin = np.exp(logx_min - logx_max)

            if N == 0 or numResampleLogX > 1:
                U = Umin + (1. - Umin)*np.random.rand(len(which))
            else:
                U = Umin + (1. - Umin)*np.linspace(1./(N+1), 1. - 1./(N+1), N)
            logx_samples_thisLevel = np.sort(logx_max + np.log(U))[::-1]

            for j in xrange(0, which.size):
                logx_samples[logl_samples_thisLevel[j][2]][z] = logx_samples_thisLevel[j]

                if j != which.size - 1:
                    left = logx_samples_thisLevel[j+1]
                elif i == levels.shape[0]-1:
                    left = -1E300
                else:
                    left = levels[i+1][0]

                if j != 0:
                    right = logx_samples_thisLevel[j-1]
                else:
                    right = levels[i][0]

                logp_samples[logl_samples_thisLevel[j][2]][z] = np.log(0.5) + postprocess.logdiffexp(right, left)

        logl = sample_info[:,1]/temperature

        logp_samples[:,z] = logp_samples[:,z] - postprocess.logsumexp(logp_samples[:,z])
        logP_samples[:,z] = logp_samples[:,z] + logl
        logz_estimates[z] = postprocess.logsumexp(logP_samples[:,z])
        logP_samples[:,z] -= logz_estimates[z]
        P_samples[:,z] = np.exp(logP_samples[:,z])
        H_estimates[z] = -logz_estimates[z] + np.sum(P_samples[:,z]*logl)

    return logx_samples, P_samples


def find_weights(p_samples):

    p_ind = np.where(p_samples > 0)[0]
    if p_ind[0] > 10:
        return True
    else:
        return False

