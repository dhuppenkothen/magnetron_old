from pylab import *

data = loadtxt('data.txt')
posterior_sample = loadtxt('posterior_sample.txt')

ion()
for i in xrange(0, posterior_sample.shape[0]):
  hold(False)
  plot(data[:,0], data[:,1], 'bo')
  hold(True)
  plot(data[:,0], posterior_sample[i, -data.shape[0]:], 'r-')
  ylim([0, 1.1*data[:,1].max()])
  draw()

ioff()
show()


