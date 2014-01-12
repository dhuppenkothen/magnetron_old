"""
This is just for eyeballing the data,
marcus.
"""

import numpy as np
import pylab as pl
import glob, sys, math, optparse


if __name__ == "__main__":

    parser = optparse.OptionParser(usage="usage %prog [options]")
    parser.add_option("-n",type = "int",dest = "numplots", default=9,
                      help="some number of data files to plot (default is 9)")
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
        first = np.min(m[:,0])
        m[:,0] = m[:,0] - first
        pl.text(np.min(m[:,0]),np.max(m[:,1]),name, color='red', fontsize=8,alpha=0.3)
        if len(names)>1: pl.text(np.min(m[:,0]),np.max(m[:,1])/2,str(i), color='blue')
        pl.plot(m[:,0],m[:,1],'-k')
        pl.axis("off")

    outfile = 'transients'
    pl.savefig(outfile,dpi=150)
    print('Wrote %s.png'%(outfile))
