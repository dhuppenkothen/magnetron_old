"""
This is just for eyeballing the data,
marcus.
"""

import numpy as np
import pylab as pl
import glob, sys, math, optparse

def whichWay(t,y):
        # is the nearest other maximum to the left or the right of the "global" maximum?
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


if __name__ == "__main__":

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-n",type = "int",dest = "numplots", default=99999,
                      help="some number of data files to plot (default is all)")
    parser.add_option("-i",type = "int",dest = "index",
                      help="index (just in the file listing) of a specific file to plot")
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
    for i,name in enumerate(names):
        pl.subplot(numrows,numcols,i+1)
        m = datafiles[name]
        t = m[:,0]
        y = m[:,1]
        first = np.min(t)
        t = t - first
        
        pl.text(np.min(t),np.max(y),name, color='red', fontsize=8,alpha=0.3)
        if len(names)>1: pl.text(np.min(t),np.max(y)/2,str(i), color='k')

        # Just for kicks: find the max, and test whether the nearest
        # NEXT max is before (show as red) or after (show as blue). This is
        # a really really really crude test of ANY kind of skew,
        # either in spike time, or skew-per-spike. Result: I don't see any.
        nearestMaxDirection = whichWay(t,y)
        colour = 'black'
        if nearestMaxDirection == 'right': colour='blue'
        if nearestMaxDirection == 'left': colour='red'
        pl.plot(t,y,'-',color=colour)
        pl.axis("off")


    outfile = 'transients'
    pl.savefig(outfile,dpi=150)
    print('Wrote %s.png'%(outfile))
