"""
This is just quick-and-dirty stuff for eyeballing the data.
marcus.
"""

import numpy as np
import pylab as pl
import glob, sys, math, optparse


def whichWay(t,y):
    # Just for kicks: find the max, and test whether the nearest
    # NEXT max is before (show as red) or after (show as blue). This is
    # a really really really crude test of ANY kind of skew,
    # either in spike time, or skew-per-spike. 
    maxpoint = np.argmax(y)
    nextmax,prevmax = maxpoint,maxpoint

    tmpval = y[maxpoint]
    while (y[nextmax] <= tmpval):
        tmpval = y[nextmax]
	nextmax = nextmax+1
    gap_to_next = nextmax - maxpoint

    tmpval = y[maxpoint]
    while (y[prevmax] <= tmpval):
        tmpval = y[prevmax]
	prevmax = prevmax-1
    gap_to_prev = maxpoint - prevmax
    if (gap_to_next < gap_to_prev): return 'right'
    elif (gap_to_next > gap_to_prev): return 'left'
    return 'tied'

def near_the_max(t,y, spread=2):
    # Find the max, and test whether the mean of the nearest 'spread'
    # points to the right (ie immediately after the max) is higher
    # than the same to the left (before). If higher after than before
    # show as blue, else as red (or black if a tie or 'spread' takes
    # you beyond the data).

    maxpoint = np.argmax(y)
    if (maxpoint<spread) or (maxpoint>len(y)-spread): return 1.0,1.0
	
    lhs_mean = np.mean(y[maxpoint-spread:maxpoint])
    rhs_mean = np.mean(y[maxpoint+1:maxpoint+spread+1])
    return lhs_mean, rhs_mean


if __name__ == "__main__":

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-n",type = "int",dest = "numplots", default=99999,
                      help="some number of data files to plot (default is all)")
    parser.add_option("-i",type = "int",dest = "index",
                      help="index (just in the file listing) of a specific file to plot")
    parser.add_option("-s",type = "int",dest = "spread",default=5,
                      help="the spread of data around the max that we are interested in")
    opts, args = parser.parse_args()
    if (opts.index is None) and (opts.numplots is None):
        parser.print_help()
        sys.exit(-1)


    datafiles = {}
    names = []
    for filename in glob.glob("*.dat"):
        names.append(filename)
        datafiles[filename] = np.loadtxt(filename)
    if opts.index is not None:
        numplots = 1
        name = names[opts.index]
        names = [name]
    else:
        numplots = min(opts.numplots, len(names))
        names = names[:numplots]

    fig = pl.figure(figsize=(12,8))
    numrows = int(math.ceil(np.sqrt(numplots)))
    numcols = int(math.ceil(1.0*numplots/numrows))
    R,L = 0,0
    for i,name in enumerate(names):
        t = datafiles[name][:,0]
        y = datafiles[name][:,1]
        t = t - np.min(t)
        
        pl.subplot(numrows,numcols,i+1)
        #pl.text(np.min(t),np.max(y),name, color='black', fontsize=6,alpha=0.5)
        #if len(names)>1: pl.text(np.min(t),np.max(y)/2,str(i), color='k')

        # nearestMaxDirection = whichWay(t,y)

        L_mean, R_mean = near_the_max(t,y,opts.spread)
	ratio = R_mean/(R_mean + L_mean)
	eps = 0.001
	colour = 'black'
        if ratio > 0.5+eps: 
            colour='blue'
	    R = R+1
	elif ratio < 0.5-eps: 
            colour='red'
	    L = L+1

	base_alpha=0.1
        pl.plot(t,y,'-',color=colour, alpha = base_alpha + (1-base_alpha)*min(np.abs(ratio-0.5)/0.2,1.))
        pl.axis("off")
        pl.text(np.min(t),np.max(y)*2./3,'%.2f'%(ratio), color=colour, fontsize=9)


    outfile = 'transients'
    pl.savefig(outfile,dpi=300)
    print('Wrote %s.png'%(outfile))
    print 'spread: %3d, \t L: %3d, \t R: %3d' %(opts.spread,L,R)
