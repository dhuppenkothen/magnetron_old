import burstmodel
import numpy as np


def runbursts(bid):

    filenames = glob.glob(str(bid) + '*data.dat')
    parafile = glob.glob(str(bid) + '_params.dat')
    pardata = burstmodel.conversion(parafile)

    bsts = np.array([float(t) for t in pardata[0]])
    params = np.array([np.array([float(t) for t in p.split(',')]) for p in pardata[1]])


    for i,f in enumerate(filenames):

        [bid, bstart, stuff] = f.split('_')

        data = burstmodel.conversion(f)
        times = np.array([float(t) for t in data[0]])
        counts = np.array([float(t) for t in data[1]])

        b_ind = np.where((bsts>float(bstart)-0.1) & (bsts<float(bstart)+0.1))[0][0]
        theta_guess = params[b_ind]

        burstmodel.test_burst(times, counts, theta_guess, namestr = str(bid) + '_' + str(bstart))
   
        return




