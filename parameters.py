import numpy as np


class Parameters(object):

    def __init__(self):
        return



class TwoExpParameters(Parameters, object):


    npar = 4
    parnames_log = ['t_0', 'log(scale)', 'log(amp)', 'log(skew)']
    parnames = ['t_0', 'scale', 'amp', 'skew']
    def __init__(self, t0 = None, scale = None, amp = None, skew = None, log=True, bkg=None):

        if log:
            self.parnames = TwoExpParameters.parnames_log
        else:
            self.parnames = TwoExpParameters.parnames

        self.log = log
        self.t0 = t0
        if log:
            self.log_scale = scale
            self.log_amp = amp
            self.log_skew = skew
            self._exp()

        else:
            self.scale = scale
            self.skew = skew
            self.amp = amp
            self._log()

        if not bkg is None:
            self.bkg = bkg

        return


    def _exp(self):

        if not self.log_scale is None:
            self.scale = np.exp(self.log_scale)
        if not self.log_amp is None:
            self.amp = np.exp(self.log_amp)
        if not self.log_skew is None:
            self.skew = np.exp(self.log_skew)
        return

    def _log(self):

        if not self.scale is None:
            self.log_scale = np.log(self.scale)
        if not self.amp is None:
            self.log_amp = np.log(self.amp)
        if not self.skew is None:
            self.log_skew = np.log(self.skew)

        return

    def _extract_params(self, scale_locked=False, skew_locked=False, log=True):
        parlist = [self.t0]
        #print("type parlist: " + str(type(parlist)))

        if not scale_locked:
            if log:
                parlist.append(self.log_scale)
            else:
                parlist.append(self.scale)

        if log:
            parlist.append(self.log_amp)
        else:
            parlist.append(self.amp)

        if not skew_locked:
            if log:
                parlist.append(self.log_skew)
            else:
                parlist.append(self.skew)
        return parlist


class TwoExpCombined(Parameters, object):
    '''
    par: list of parameters
    ncomp: number of components


    '''

    def __init__(self, par, ncomp, parclass=TwoExpParameters, scale_locked=False, skew_locked=False, log=True, bkg=False):


        self.all = []
        self.log = log

        self.scale_locked = scale_locked
        self.skew_locked = skew_locked
        self.parclass = parclass

        npar = parclass.npar
        self.npar_all = np.sum([npar for p in xrange(ncomp)])
        n_ind = 0
        if bkg:
            #print("I am in bkg")
            if log:
                self.log_bkg = par[-1]
                self.bkg = np.exp(self.log_bkg)
            else:
                self.bkg = par[-1]
                self.log_bkg = np.log(self.bkg)
            n_ind -= 1

        if ncomp >=1:

            if skew_locked:
                #print("I am in skew_locked")
                npar -= 1
                #print('n_ind: ' + str(n_ind))
                #print("par[-1+n_ind]: " + str(par[-1+n_ind]))
                self.skew = par[-1+n_ind]
                n_ind -= 1
            if scale_locked:
                #print("I am in scale_locked")
                npar -= 1
                #print("scale index: " + str(-1+n_ind))
                self.scale = par[-1+n_ind]


            #print("npar: " + str(npar))
            for n in xrange(ncomp):
                par_temp = par[n*npar:(n*npar)+npar]
                t0 = par_temp[0]
                if scale_locked:
                    amp = par_temp[1]
                    scale = self.scale
                else:
                    amp = par_temp[2]
                    scale = par_temp[1]
                if skew_locked:
                    skew = self.skew
                else:
                    skew = par_temp[-1]

                p = parclass(t0=t0, scale=scale, amp=amp, skew=skew, log=log, bkg=None)
                self.all.append(p)

        elif ncomp == 0:
            self.all = []
            #if log:
            #   self.bkg = par[0]

        else:
            raise Exception("Something went horribly, horribly wrong. Try again!")

        return

    def _add_word(self, par):

        t0 = par[0]
        if self.scale_locked and hasattr(self, "scale"):
            scale = self.scale
            amp = par[1]
        else:
            scale = par[1]
            self.scale = scale
            amp = par[2]
        if self.skew_locked and hasattr(self, "skew"):
            skew = self.skew
        else:
            skew = par[-1]
            self.skew = skew

        new_word = self.parclass(t0=t0, scale=scale, amp=amp, skew=skew, log=self.log)
        self.all.append(new_word)

        return

    def _extract_params(self, log=True):

        parlist = []

        if np.size(self.all) >= 1:
            for a in self.all:
                parlist.append(a._extract_params(log=log, scale_locked=self.scale_locked, skew_locked=self.skew_locked))

            parlist = np.array(parlist).flatten()

            if hasattr(self, "scale"):
                parlist = np.append(parlist, self.scale)

            if hasattr(self, "skew"):
                parlist = np.append(parlist, self.skew)

        if self.bkg:
            if log:
                parlist = np.append(parlist, self.log_bkg)
            else:
                parlist = np.append(parlist, self.bkg)

        return parlist