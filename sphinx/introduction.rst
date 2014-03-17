

Introduction
============

This documentation gives details of the code in project *TransientDict*, informally called *Magnetron*.
*Magnetron* is work in progress; it supplies code to decompose complex time series into a number of 
simple shapes (currently exponentials). Use at your own peril, but do let me know if it breaks, or make
a pull request if it breaks and you've fixed it (even better!). 

Two caveats:

1. Most of this documentation concerns the branch *parameterclass*, which is not *master*, but what I've
spent most of my development efforts on, and which seems to be fairly stable at this point.

2. There was a second caveat, but I've forgotten it now. It will come back to me.



To Do
======

This I need to do, with no guarantee of completeness. Feel free to add to it.

* create log files for running ``BurstModel.find_spikes``. Needs to save at the
very least autocorrelation times, and overall parameters like the number of 
ensemble walkers and iterations for the MCMC run, and whether the rise time and
skewness parameter are fixed between model runs.
* make new implementation in branch *parameterclass* compatible with the code
the others wrote in *master*. 


