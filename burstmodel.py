

def word(xvar, width, skew = 2.0):

    x = np.array(xvar)/width
    #y = exp(((x>0)*2-1)*x)
    y = zeros_like(x)
    y[x<0] = exp(x[x<0])
    y[x>0] = exp(-skew*x[x>0])

    return y
#    y1 = [np.exp(x) for x in xvar if x < 0]
#    y2 = [np.exp(-x/skew) for x in xvar if x > 0]
#    y1.extend(y2)
#    return y1


#### theta[0] is move parameter
#### theta[1] is scale parameter
#### theta[2] is amplitude

def double_exponential(x, theta):

    #x = x - theta[0]

    move = theta[0]
    scale = theta[1]
    amp = theta[2]

    x = x - move
    y = amp*word(x, scale)

    return y
    
# Poisson log likelihood based on a set of rates
# log[ prod exp(-lamb)*lamb^x/x! ]
# exp(-lamb)
import scipy.special
def log_likelihood(lambdas, data):
    return -np.sum(lambdas) - np.sum(data*np.log(lambdas))\
		-np.sum(scipy.special.gammaln(data))

