from pylab import *

data = np.loadtxt('/Users/daniela/data/magnetars/sgr1900/P30410/bursts_corrected/burstfiles_analysed/significant/30410-01-05-00_147267405.361_data.dat')
posterior_sample = atleast_2d(loadtxt('posterior_sample.txt'))

ion()
for i in xrange(0, posterior_sample.shape[0]):
  hold(False)
  plot(data[:,0], data[:,1], "-", color='black', lw=1)
  hold(True)
  plot(data[:,0], posterior_sample[i, -data.shape[0]:], 'r-')
  ylim([0, 1.1*data[:,1].max()])
  draw()

ioff()
show()

hist(posterior_sample[:,7], 20)
xlabel('Number of Bursts')
show()

pos = posterior_sample[:, 8:108]
pos = pos[pos != 0.]
hist(pos, 1000)
xlabel('Time')
title('Positions of Bursts')
show()

