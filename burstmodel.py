import numpy as np

def word(time, width, skew = 2.0):

    t = np.array(time)/width
    #y = exp(((x>0)*2-1)*x)
    y = np.zeros_like(t)
    y[t<=0] = np.exp(skew*t[t<=0])
    y[t>0] = np.exp(-t[t>0])

    return y
#    y1 = [np.exp(x) for x in xvar if x < 0]
#    y2 = [np.exp(-x/skew) for x in xvar if x > 0]
#    y1.extend(y2)
#    return y1


#### theta[0] is move parameter
#### theta[1] is scale parameter
#### theta[2] is amplitude

def double_exponential(time, theta):

    #x = x - theta[0]

    move = theta[0]
    scale = theta[1]
    amp = theta[2]

    time = time - move
    counts = amp*word(time, scale)

    return counts
    
# Poisson log likelihood based on a set of rates
# log[ prod exp(-lamb)*lamb^x/x! ]
# exp(-lamb)
# I (BJB) have tested this against R's dpois( , , log=TRUE)
import scipy.special
def log_likelihood(lambdas, data):
    return -np.sum(lambdas) + np.sum(data*np.log(lambdas))\
		-np.sum(scipy.special.gammaln(data + 1))

