



class Word(object).

    def __init__(self, time):
        self.time = np.array(time)





class TwoExp(Word, object):

    def __init__(self, time):
        Word.__init__(time)
        return


    def model(scale, skew):

        t = self.time/scale
        y = np.zeros_like(t)
        y[t<=0] = np.exp(t[t<=0])
        y[t>0] = np.exp(-t[t>0]/skew)

        return y

    def __call__(scale, skew=2.0):

        return model(scale, skew)




