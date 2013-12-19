import burstmodel
import numpy as np
import argparse
import glob

### parfile needs to be an ascii file with the trigger ID 
### on the first column, the burst start time on the second and
### guesses for the parameters on the rest of the columns
def read_pardata(parfile):

    fpar = open(parfile, 'r')
    content = fpar.readlines()
 
    bids, bsts, theta_guesses =[], [], []

    for c in content:
        if not c[0] == '#':
            bids.append(c.split()[0])
            bsts.append(float(c.split()[1]))
            theta_guesses.append([float(t) for t in c.split()[2:]])

    return bids, bsts, theta_guesses

def runbursts(parfile):

    filenames = glob.glob('090122037a*data.dat')
    print(filenames)
    parafile = glob.glob(parfile)
    bids, bsts, theta_guesses = read_pardata(parfile) 

    bsts = np.array([float(b) for b in bsts])
    print(bsts)

    for i,f in enumerate(filenames):

        ## pick apart data file name to get identifiers
        [bid, bstart, stuff] = f.split('_')

        ## check whether there's an entry for this file name in the parameter
        ## list; if not, skip!
        print('bstart: ' + str(bstart))
        b_ind = np.where((bsts>float(bstart)-0.1) & (bsts<float(bstart)+0.1))[0]
        print(b_ind)
        if b_ind.size == 0:
            continue

        ## read in data
        data = burstmodel.conversion(f)
        times = np.array([float(t) for t in data[0]])
        counts = np.array([float(t) for t in data[1]])

        ### find right initial guess from list
        b_ind = np.where((bsts>float(bstart)-0.1) & (bsts<float(bstart)+0.1))[0][0]
        theta_guess = theta_guesses[b_ind]

        ## run preliminary analysis
        burstmodel.test_burst(times, counts, theta_guess, namestr = str(bid) + '_' + str(bstart))
   
    return


def main():

    runbursts(parfile)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Script to run the spike fitting via MCMC with a bunch of bursts')
    parser.add_argument('-f', '--filename', action='store', dest ='filename', help='input filename')

    clargs = parser.parse_args()
    parfile = clargs.filename

    main()
