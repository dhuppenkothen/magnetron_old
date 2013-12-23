



class Word(object).

    def __init__(self, times):
        self.times = np.array(times)

    def _pack(self, npars, theta_flat):
        theta_new = []
        par_counter = 0
        for n in npars:
            theta_new.append(theta_flat[par_counter:par_counter+n]
            par_counter = par_counter + n

        return theta_new

    def _unpack(self, theta):
        theta_flat = []
        for t in theta:
            theta_flat.extend(t)
        return np.array(theta_flat)

class TwoExp(Word, object):

    def __init__(self, time):
        self.npar = 3
        Word.__init__(time)
        return


    def model(self, event_time, scale, skew):

        t = (self.times-event_time)/scale
        y = np.zeros_like(t)
        y[t<=0] = np.exp(t[t<=0])
        y[t>0] = np.exp(-t[t>0]/skew)

        return y

    def __call__(self, event_time, scale, skew=2.0):

        return model(event_time, scale, skew)





class CombinedWord(Word, object):

    def __init__(self, times,  wordlist):

        ### instantiate word objects
        self.wordlist = [w(times, counts) for w in wordlist]

        Word.__init__(times).
        return

    def model(self, *theta_all):

        y = np.zeros(len(self.times))
        error_theta_all = 'Length of theta_all does not match length of word list'
        assert len(theta_all) == len(self.wordlist), error_theta_all

        error_theta_individual = 'Number of elements in theta does not match required number of parameters in word!'
        for t,w in zip(self.wordlist):
            assert len(t) == w.npar, error_theta_individual

                y = y + w(t) ## add word to output array

        return y


    def __call__(self, *theta_all):
        return model(theta_all)


        
