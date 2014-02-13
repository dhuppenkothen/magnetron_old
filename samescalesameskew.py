
import glob
import argparse
import burstmodel



def main():

    if len(filenames) == 0:
        raise Exception("No files in directory!")

    for f in filenames:
        filecomponents = f.split("/")
        fname = filecomponents[-1]
        froot = fname[:-9]


        if instrument == 'gbm':
            times, counts = burstmodel.read_gbm_lightcurves(f)
        else:
            raise Exception("Instrument not known!")

        bm = burstmodel.BurstModel(times, counts)

        all_means, all_err, all_postmax, all_quants, all_theta_init = \
            bm.find_spikes(nmax=10, nwalker=500, niter=200, burnin=200, namestr=froot, scale_locked=scale,
                           skew_locked=skew)


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
    parser.add_argument('--scale', action='store', dest='scale', required=False, default='False', type=bool,
                        help="If true, scale will be the same for all words")
    parser.add_argument('--skew', action='store', dest='skew', required=False, default='False', type=bool,
                        help="If true, skew will be the same for all words")

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

    scale = clargs.scale
    skew = clargs.skew

    main()
