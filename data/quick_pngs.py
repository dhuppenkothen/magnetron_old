import numpy as np
import pylab as pl
import glob

# Another quick hack to eye-ball the data - Iain.


filenames = glob.glob("*.dat")
#filenames = [filenames[0]]

all_data = []
max_cc = 0
for ff in filenames:
    data = np.loadtxt(ff)
    tt = data[:,0]
    # Make all plots start at time zero
    tt = tt - tt.min()
    cc = data[:,1]
    err = np.sqrt(cc)
    c_mx = cc + 2*err
    c_mn = cc - 2*err
    # hacks to stop log plots going crazy
    cc = np.maximum(1, cc)
    c_mx = np.maximum(1, c_mx)
    c_mn = np.maximum(1, c_mn)
    # Track max count, so can have same y-axis for all plots
    max_cc = np.maximum(max_cc, cc.max())
    all_data.append((tt, cc, c_mn, c_mx))

for (ff, (tt, cc, c_mn, c_mx)) in zip(filenames, all_data):
    pl.clf()
    b_rate = np.median(cc)
    b_mn = b_rate - 3*np.sqrt(b_rate)
    b_mx = b_rate + 3*np.sqrt(b_rate)
    pl.fill_between(tt, np.tile(b_mn, tt.shape), np.tile(b_mx, tt.shape), color='r', alpha=0.2)
    b_mn = b_rate - 2*np.sqrt(b_rate)
    b_mx = b_rate + 2*np.sqrt(b_rate)
    pl.fill_between(tt, np.tile(b_mn, tt.shape), np.tile(b_mx, tt.shape), color='r', alpha=0.2)
    pl.fill_between(tt, c_mn, c_mx, alpha=0.5)
    pl.plot(tt, cc, 'k.-', linewidth=0.5)
    pl.plot(tt, np.tile(b_rate, tt.shape), 'r')
    pl.axis((0, tt.max(), 1, max_cc))
    pl.gca().set_yscale('log')
    pl.savefig(ff + ".png", bbox_inches="tight", pad_inches=0)
