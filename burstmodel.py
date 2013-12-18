

def word(xvar, skew = 2.0):

    y1 = [np.exp(x) for x in xvar if x < 0]
    y2 = [np.exp(-x/skew) for x in xvar if x > 0]
    y1.extend(y2)
    return y1


def double_exponential(x, theta):

    x = x - theta[0]

    y = word(x)




    

