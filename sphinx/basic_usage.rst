

Basic Usage
============

Usage of the scripts in *Magnetron* depends strongly on the use case. Here, we focus on the
usage of the code at the highest level. For details on the individual classes consult :doc:`api`.


High-Level Scripts
===================

The primary objective is to decompose a time series into a number of simple shapes. This is done
by searching for a likely position of a peak, then defining a model with a single peak + a background
parameter, and running MCMC via `emcee <http://dan.iel.fm/emcee/current/>`_. Note that the branch *dnest*
also supports Diffusive Nested Sampling, if the relevant code is installed.
The results of the MCMC run are stored in a python pickle file as well as a number of diagnostic plot. 
Iteratively, a new model will be produced with another model component added at the most likely location
(the highest outlier of the data-previous model residuals). Again MCMC provides an approximation of the posterior
distribution of the parameters. This procedure is repeated up to the maximum number of model components defined
(10 by default). 

Running this procedure on a single or multiple time series proceeds via the script ``samescalesameskew.py``. 
This script can be invoked from the command line with a multitude of options::

    $ python samescalesameskew.py --help
    usage: samescalesameskew.py [-h] (-a | -s) [-w NWALKER] [-i NITER]
                                [--instrument INSTRUMENT] [--lock-scale]
                                [--lock-skew] [-f FILENAME] [-d DIR]
    Model magnetar bursts with spikes!
    optional arguments:
        -h, --help            show this help message and exit
        -a, --all             run on all files in the directory
        -s, --single          run on a single file
        -w NWALKER, --nwalker NWALKER
                                Number of emcee walkers
        -i NITER, --niter NITER
                                number of emcee iterations
         --instrument INSTRUMENT
                                Instrument data was taken with
        --lock-scale          If true, scale will be the same for all words
        --lock-skew           If true, skew will be the same for all words
        single file:
        options for running script on a single file
        -f FILENAME, --filename FILENAME
                                file with data
        all bursts:
        options for running script on all bursts
        -d DIR, --dir DIR     directory with data files

There's a main switch ``--single`` versus ``--all``, which tells the script whether to
run on a single data file (which then needs to be specified via the ``-f`` option), or
on all files in a directory (which needs then to be specified with the ``-d`` or ``--dir``
option). 

Data files **must** be in ASCII format and have at least two columns, where the first two will be read out. 
The **first** column must include the **time stamps** of the data points, the **second**
column the **counts per bin**. Unbinned data is currently not supported.

Despite the name of the script, whether the model considers one rise time and/or skewness
parameter per model component, or one rise time and/or skewness parameter for all model 
components simultaneously can be set with the keywords ``--lock-scale`` (for the rise time)
and ``--lock-skew`` for the skewness parameter. Note that these are True/False arguments:
inclusion of the argument on the command line will automatically set this True, absence of it
on the command line will set it False. 

Arguments ``--nwalker`` and ``--niter`` set the number of ensemble walkers and interations for the
MCMC run, respectively. At this point, one cannot change this between models considered (this would
need to be implemented separately). 

The ``--instrument`` argument currently does nothing; at the moment we only consider data recorded with
Fermi/GBM. If other data types are used, this could potentially be useful in the future to read in 
data in a consistent manner.

Below a few examples on how to run the script.

1. Run on a single time series data file, with no common parameters between model components; emcee will
use 500 ensemble walkers and evolve the Markov chains for 100 iterations (after a standard 200 iterations
of burning in)::

    $ python samescalesameskew.py -s -w 500 -i 100 -f "mydata.dat"

2. Run on all data files in directory ``./data/``, with the rise times linked between model components::

    $ python samescalesameskew.py -a -w 500 -i 100 --lock-scale -d "./data/"

3. Run on all data files in current directory, with rise times and skewness parameter linked between
model components::

    $ python samescalesameskew.py -a -w 500 -i 100 --lock-scale --lock-skew -d "./"




 


