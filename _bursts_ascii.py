
def save_bursts_ascii(alldict):

    for key, lis in alldict.iteritems():
        for b in lis:
            f = open(str(b.bid) + '_' + str(b.bst)[:7] + '_data.dat', 'w')
            maxind = np.array(b.time).searchsorted(b.time[0]+2.0)
            lc = lightcurve.Lightcurve(b.time[:maxind], timestep=0.005)
            for t,c in zip(lc.time, lc.counts):
                f.write(str(t) + "\t" + str(c) + "\n")
            f.close()
    return
