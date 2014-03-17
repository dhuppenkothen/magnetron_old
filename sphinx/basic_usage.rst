

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


Tests
------

Simple tests are implemented in ``parameter_tests.py``. The functions in this script test the basic functionality
of the classes defined in  ``parameters.py``, ``word.py`` and ``burstdict.py``. 
These tests can be run all together or individually from within python or from the command line like this::

    $ python parameter_tests.py --help
        usage: parameter_tests.py [-h] [-p] [-w] [-d] [-a] [--post] [-m] [-l]
        
        Various tests for the classes defined in parameters.py, word.py and
        burstmodel.py
        
        optional arguments:
            -h, --help        show this help message and exit
            -p, --parameters  Run parameter class tests
            -w, --word        Run word class tests
            -d, --burstdict   Run burstdict class tests
            -a, --all         Run all tests at once!
            --post            Run tests on class WordPosterior with new parameter
                              implementation
            -m, --model       Run tests on class BurstModel with new parameter
                              implementation
            -l, --longrun     When running BurstModel tests, do you want to perform a
                              long MCMC run?


The tests for class ``BurstModel`` support a ``--longrun`` option; for many quick checks on
whether the code breaks or basic functionality is there, a full MCMC run would take too much time,
thus by default the number of ensemble walkers and iterations used is low. When ``-l`` or
``--longrun`` is set, a longer MCMC run will be performed.



Modeling Data with Model Shapes
---------------------------------

Running the whole procedure on a single or multiple time series proceeds via the script ``samescalesameskew.py``. 
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


Outputs are saved in a number of files, all of which have a common root. As we currently only look
at Fermi/GBM data, the root for the output filenames are taken from the input filename, minus ``_data.dat`` 
at the end. 

For each model, the script saves a python pickle file under ``fileroot_k[n]_posterior.dat``
(where [n] is the number of components in the model) with a dictionary with the following keywords:

* **means**: posterior means of the parameters, in a ``parameters.TwoExpCombined`` object
* **max**: posterior maximum of the parameters, in a ``parameters.TwoExpCombined`` object
* **sampler**: list of parameter sets, as given in ``s.flatchain``, where ``s`` is an object
  of type ``emcee.EnsembleSampler``.
* **lnprob**: log posterior probability of the parameter sets stored in **sampler**
* **err**: standard deviation for each parameter as computed from the samples in **sampler**
* **quants**: list with 0.05, 0.5 and 0.95 quantiles for each parameter.
* **init**: initial parameter set used as a starting point for the MCMC run
* (**niter**: number of iterations in MCMC run; this is a recent addition and not yet present
  in every data file)

Three types of plots are saved:

1. A triangle plot of the posterior parameter distributions, under ``fileroot_k[n]_posterior.png``
2. the original time series with the model of the posterior maximum overplotted in blue, and models for the
   0.05, 0.5 and 0.95 quantiles derived from 1000 randomly chosen parameter sets overplotted in red (bands),
   in ``fileroot_k[n]_lc.png``
3. time series of the actual Markov chains for each parameter in ``fileroot_k[n]_p[j]_chains.png``. ``j`` is 
   the jth parameter; I could put the actual parameter names, but I'm currently too lazy to do this for purely
   diagnostic plots (also, with a bit of knowledge of the code, it's easy to read off which is which)
4. for all models considered, a plot of the posterior quantiles of each parameter versus the number of 
   components in the model, grouped by parameter type. Produces four plots for the ``word.TwoExp`` model
   currently used in all analyses: ``fileroot_t0.png`` for the peak positions of each component,
   ``fileroot_log_scale.png`` for the logarithm of the rise times, ``fileroot_log_amp.png`` for the logarithm
   of the component amplitudes, and ``fileroot_log_skew.png`` for the logarithm of the skewness parameter. 


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



Extracting Information from Many Bursts
----------------------------------------

Making inferences over many bursts can be difficult. By default, the code run by ``samescalesameskew.py`` 
produces some output in the form of the MCMC samples for the parameters, as well as diagnostic plots.
It is possible to re-make these plots fromm the saved posterior distributions, change details of these
plots, and gather quantities like the posterior maxima and quantiles into one data file for analysis
across a whole ensemble of time series.

The easiest way to do this is by fiddling with ``plot_parameters.py``. This re-makes most of the plots 
returned by ``samescalesameskew.py``, but plotting can be commented out if only a file with the combined
results of the MCMC runs for many bursts is required.

Note that this scripts is currently set up to deal exclusively with Fermi/GBM data, which comes out of my
pipeline in files with ``BurstID_BurstStartTime_data.dat``-format.  

``plot_parameters.py`` can be called from the command line like this::

    $ python plot_parameters.py --help
    usage: plot_parameters.py [-h] [--scale-locked] [--skew-locked] [-d DATA_DIR]
                              [-b BID] [-n NSAMPLES] [-i NITER]

    Model magnetar bursts with spikes!

    optional arguments:
        -h, --help            show this help message and exit
        --scale-locked        Scale the same for all words?
        --skew-locked         Skew the same for all words?
         -d DATA_DIR, --dir DATA_DIR
                        Directory where the data files are located
        -b BID, --bid BID     Pick specific burst ID to run on
        -n NSAMPLES, --nsamples NSAMPLES
                            Number of samples to be used in average light curve.
        -i NITER, --niter NITER
                            Number of iterations in MCMC run

Again, one must specify whether rise time and skewness parameter are the same for each model
component. This requires knowledge of whatever arguments were used when running the analysis itself.
By default, the script takes the entire contents of directory specified with ``-d`` or ``--dir`` 
(default is ``./``), byt it is possible to specify a Fermi/GBM BurstID with ``-b`` or ``--bid`` to
run on.
For the quantiles overplotted on the output time series plots of the data and models, one may specify
how many samples to use in the computation of the quantiles via ``-n``or ``--nsamples``; a larger 
number translates into longer compute times. If the number specified with this argument is greater 
than the number of samples in the files storing the MCMC samples, it is automatically re-set to that
number.

The argument ``-i``, ``--niter`` is a recent addition. Previously, I did not save the number of iterations
per MCMC run anywhere, which makes computing the MCMC time series for diagnostics *a posterori* quite difficult.
For those files without **niter** keyword in ``fileroot_posterior.dat``, ``--niter`` must be set explicitly, or 
the code throws an exception.

This script returns some of the same plots as ``samescalesameskew.py``:

* ``fileroot_k[n]_lc.png``
* ``fileroot_k[n]_p[j]_chains.png``
* ``fileroot_t0.png``, ``fileroot_log_scale.png``, ``fileroot_log_amp.png``, ``fileroot_log_skew.png``, 

as well as a python pickle file with a dictionary storing quantities for all models and time series files 
in the directory considered:

* **t0_max**, **t0_cl**, **t0_m**, **t0_cu**: posterior maximum, 0.05, 0.5 and 0.95 quantiles for the peak time
* **scale_max**, **scale_cl**, **scale_m**, **scale_cu**: posterior maximum, 0.05, 0.5 and 0.95 quantiles for 
  the log rise time
* **amp_max**, **amp_cl**, **amp_m**, **amp_cu**: posterior maximum, 0.05, 0.5 and 0.95 quantiles for the 
  log amplitude
* **skew_max**, **skew_cl**, **skew_m**, **skew_cu**: posterior maximum, 0.05, 0.5 and 0.95 quantiles for the 
  log skewness parameter  

The latter can be used for further ensemble analysis.
