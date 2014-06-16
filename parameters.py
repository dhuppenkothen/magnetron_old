### DEFINITION FOR PARAMETER OBJECTS #########
#
# This is so that I can define meaningful parameter objects
# and make passing around and extracting individual parameters
# easier and more transparent than passing them around in
# meaningless lists.
#
#
#

import numpy as np


class Parameters(object):

    """ Superclass Parameters.
    Nothing to see here.
    Define subclasses for individual types of models instead.
    """

    def __init__(self):
        return



class TwoExpParameters(Parameters, object):

    """
    class TwoExpParameters

    This class defines parameters for the model describing two exponentials,
    one rising and one falling, defined in class TwoExp in module word.

    This model has five parameters, to be passed into __init__:
    t0: peak time (=max count rate)
    scale: the width of the exponentials
    amp: the amplitude
    skew: a skewness parameter setting the skewness of one exponential to another

    additional parameters:
    log: if True, scale, amp and skew (and bkg)are given as logarithmic quantities
    bkg: if not None, then there's a constant background level to be added

    """

    npar = 4
    parnames_log = ['t_0', 'log(scale)', 'log(amp)', 'log(skew)']
    parnames = ['t_0', 'scale', 'amp', 'skew']
    def __init__(self, t0 = None, scale = None, amp = None, skew = None, log=True, bkg=None):

        ### set names of parameters, for plotting
        if log:
            self.parnames = TwoExpParameters.parnames_log
        else:
            self.parnames = TwoExpParameters.parnames

        ### store parameters as attributes
        self.log = log
        self.t0 = t0

        ### if parameters are logarithmic, save them in appropriate attributes
        ### and exponentiate them
        if log:
            self.log_scale = scale
            self.log_amp = amp
            self.log_skew = skew
            self._exp()

        ### else do reverse: save in attributes and take the logarithm
        else:
            self.scale = scale
            self.skew = skew
            self.amp = amp
            self._log()

        ### if the background parameter is set, then store that, too
        if not bkg is None:
            self.bkg = bkg

        return


    def _exp(self):

        """
        This method takes no arguments, but calls the log attributes of
        the instance of class TwoExpParameters and exponentiates them.
        The results are stored in the appropriate non-log attributes of the
        relevant parameters.
        """

        if not self.log_scale is None:
            self.scale = np.exp(self.log_scale)
        if not self.log_amp is None:
            self.amp = np.exp(self.log_amp)
        if not self.log_skew is None:
            self.skew = np.exp(self.log_skew)
        return

    def _log(self):

        """
        This method takes the logarithm of non-log parameters
        and stores them in the appropriate attributes.

        """

        if not self.scale is None:
            self.log_scale = np.log(self.scale)
        if not self.amp is None:
            self.log_amp = np.log(self.amp)
        if not self.skew is None:
            self.log_skew = np.log(self.skew)

        return

    def _extract_params(self, scale_locked=False, skew_locked=False, log=True):

        """
        Extract parameters from a parameter object of type TwoExpParameters and
        store them in a list.

        @param scale_locked: if True, then don't extract scale parameter; it will be take from elsewhere.
        @param skew_locked: if True, don't extract skew parameter; it will be taken from elsewhere
        @param log: if True, extract log versions of parameters, otherwise don't
        @return: returns a list with the relevant parameters.
        """

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

    def compute_energy(self, bkg=0):
        amp = self.amp - bkg
        norm = amp*self.scale
        first_term = 1.0
        second_term = self.skew

        self.energy = norm*(first_term + second_term)

        return self.energy

    def compute_duration(self, bkg=None):

        amp = self.amp
        print("bkg: " + str(bkg))
        print("amp: " + str(amp))

        if bkg is None:
            bkg = 0.01*amp

        #print("delta amp: %.4f"%(amp-bkg))

        skew = self.skew
        scale = self.scale

        log_fall = np.log(amp/bkg)
        log_rise = np.log(bkg/amp)


        t_start = self.t0 + scale*np.log(bkg/amp)
        t_end = self.t0 + skew*scale*np.log(amp/bkg)

        #fall_term = skew*scale*log_fall
        #rise_term = scale*log_rise

        #self.duration = fall_term - rise_term

        self.duration = t_end - t_start

        return self.duration




class TwoExpCombined(Parameters, object):

    def __init__(self, par, ncomp, parclass=TwoExpParameters, scale_locked=False, skew_locked=False, log=True, bkg=False):

        """
        This object stores parameters for combinations of instances of the TwoExp model defined in word.


        @param par: List with parameters. each component needs to be of type [position, scale, amplitude, skewness].
                    If scale_locked and/or skew_locked are True, define scale, skew and bkg at the end.
        @param ncomp: number of model components
        @param parclass: which class are the parameter objects of the individual components? Should be TwoExpParameters,
                        otherwise it'll break.
        @param scale_locked: if True, scale will be the same for all components; define at the end of par
                            (before skew and bkg)
        @param skew_locked: if True, skew will be the same for all components; define at the end of par
                            (between scale and bkg)
        @param log: if True, use log versions of parameters.
        @param bkg: background level; define at the end of par (after scale and skew)
        """

        ### save some input quantities
        self.all = []
        self.log = log

        self.scale_locked = scale_locked
        self.skew_locked = skew_locked
        self.parclass = parclass

        ### number of parameters per component
        npar = parclass.npar
        ### sum of all parameters
        self.npar_all = np.sum([npar for p in xrange(ncomp)])

        ### the following code defines an index for scale and skew
        ### such that they are extracted from the list in a correct way
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


        ### if there are more than one model component, and either skew or scale or both are locked,
        ### then extract and save scale and skew in the right way
        if ncomp >=1:
            if skew_locked:
                npar -= 1
                self.skew = par[-1+n_ind]
                n_ind -= 1
            if scale_locked:
                npar -= 1
                self.scale = par[-1+n_ind]

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

                ### make an individual parameter object for each component
                ### and distribute scale and skew in the right way
                p = parclass(t0=t0, scale=scale, amp=amp, skew=skew, log=log, bkg=None)
                self.all.append(p)

        ### if there is no model component, there will only be a background
        ### and the list of components will be empty.
        elif ncomp == 0:
            self.all = []
            #if log:
            #   self.bkg = par[0]

        else:
            raise Exception("Something went horribly, horribly wrong. Try again!")

        return


    def _add_word(self, par):

        """
        Add a word to the parameter set.

        @param par: The model parameters for the additional component.

        """

        ### first parameter is definitely peak time
        t0 = par[0]
        ### if the scale is the same for all model components, extract from right attribute
        if self.scale_locked and hasattr(self, "scale"):
            scale = self.scale
            amp = par[1]
        ### otherwise the scale had better be the second item of the input list
        else:
            scale = par[1]
            self.scale = scale
            amp = par[2]

        ### same procedure for skew as for scale
        if self.skew_locked and hasattr(self, "skew"):
            skew = self.skew
        else:
            skew = par[-1]
            self.skew = skew

        new_word = self.parclass(t0=t0, scale=scale, amp=amp, skew=skew, log=self.log)
        self.all.append(new_word)

        return

    def _extract_params(self, log=True):


        """

        Extract parameters from TwoExpCombined object, store in a list.

        @param log: if True, extract log-versions of parameters.
        @return: list of parameters for the combined model. If scale_locked and skew_locked
        """


        parlist = []

        if np.size(self.all) >= 1:
            for a in self.all:
                parlist.append(a._extract_params(log=log, scale_locked=self.scale_locked, skew_locked=self.skew_locked))

            parlist = np.array(parlist).flatten()

            if self.scale_locked:
                parlist = np.append(parlist, self.scale)

            if self.skew_locked:
                parlist = np.append(parlist, self.skew)

        if self.bkg:
            if log:
                parlist = np.append(parlist, self.log_bkg)
            else:
                parlist = np.append(parlist, self.bkg)

        return parlist

    def compute_energy(self):

        e_all = []

        if "bkg" in self.__dict__.keys():
            #bkg = self.bkg
            bkg= 0.0
        else:
            bkg = 0.0
        #print("bkg: %f"%bkg)
        for a in self.all:
            e = a.compute_energy(bkg=bkg)
            e_all.append(e)

        return e_all

    def compute_duration(self):

        e_all = []

        if "bkg" in self.__dict__.keys():
            bkg = self.bkg
        else:
            bkg = 0.0
        #print("bkg: %f"%bkg)
        for a in self.all:
            e = a.compute_duration(bkg=bkg)
            e_all.append(e)

        return e_all
