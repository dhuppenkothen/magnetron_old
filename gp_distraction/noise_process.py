# A Gauss Markov Random field with tridiagonal precision.
# Iain Murray, Jan 2014

# This class implements an OU process model. Given observations this object can
# report their probability: set_params returns a log-likelihood score of the
# noise-level, lengthscale, and amplitude parameters. One can also sample from
# the posterior process. The observation noise is expressed as a prec(ision), or
# 1/variance. There is also a prior_sample method for looking at samples
# unconstrained by observations.

import numpy as np
import scipy as sp
import scipy.linalg as spl

class NoiseModel(object):
    # Future: Could consider wrapping tri-diagonal routines in lapack instead of
    # using scipy banded routines.
    def __init__(self, times, obs,
            obs_prec=1.0, lengthscale=1.0, amplitude=1.0):
        times_are_sorted = (np.diff(times) > 0).all()
        assert(times_are_sorted)
        self.times = times
        self.obs = obs
        self.set_params(obs_prec, lengthscale, amplitude)
    def set_params(self, obs_prec, lengthscale, amplitude):
        # See Iain notes 2013-12-31 (1) for derivation of precision of
        # underlying process and (2) for marginal likelihood.
        # Inverse covariance matrices, known as precision matrices are named
        # "prec_...", and their Cholesky decompositions "cprec_...". They're
        # stored in (lower) banded matrix format, as described at:
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.cholesky_banded.html
        self.lower = False # Final row is diagonal (second row here)
        tau = 1.0/amplitude**2 # marginal precision
        rhos = np.exp(-np.diff(self.times) / lengthscale)
        cc = 1.0 / (1.0 - rhos**2)
        N = self.times.size
        self.prec_prior = np.zeros((2,N))
        self.prec_prior[1,:-1] = cc
        self.prec_prior[1,1:] += cc - 1.0
        self.prec_prior[1,-1] += 1.0
        self.prec_prior[0,1:] = -cc * rhos
        self.prec_prior *= tau
        self.cprec_prior = spl.cholesky_banded(self.prec_prior, lower=self.lower)
        self.prec_post = self.prec_prior.copy()
        self.prec_post[1,:] += obs_prec
        self.cprec_post = spl.cholesky_banded(self.prec_post, lower=self.lower)
        y_prec = obs_prec * self.obs
        self.mu_post = spl.cho_solve_banded((self.cprec_post, self.lower), y_prec)
        self.obs_prec = obs_prec # so can recompute mu_post in update_obs()
        # Evaluate p(y,f=0) and p(f=0|y).
        # TODO for numerical reasons may want to use f=mu_post instead.
        Ljoint = -0.5*np.dot(self.obs, y_prec) \
                + 0.5*self.prior_prec_logdet() \
                + 0.5*np.sum(np.log(obs_prec)) \
                - 0.5*(2*N)*np.log(2*np.pi)
        Lpost = -0.5*np.dot(y_prec, self.mu_post) \
                + 0.5*self.post_prec_logdet() \
                - 0.5*N*np.log(2*np.pi)
        self.Lprob_data = Ljoint - Lpost
        return self.Lprob_data
    def update_obs(self, obs):
        self.obs = obs
        y_prec = self.obs_prec * self.obs
        self.mu_post = spl.cho_solve_banded((self.cprec_post, self.lower), y_prec)
        # The posterior precision is unaffected by the observations.
    def _logdet(self, cb):
        """log-det of matrix with given banded Cholesky decomp."""
        if self.lower:
            diag_chol = cb[0]
        else:
            diag_chol = cb[-1]
        return 2.0*np.sum(np.log(diag_chol))
    def prior_prec_logdet(self):
        """log-det of prior precision matrix"""
        return self._logdet(self.cprec_prior)
    def post_prec_logdet(self):
        """log-det of posterior precision matrix"""
        return self._logdet(self.cprec_post)
    def _sample(self, cb):
        """Sample from Gaussian with given banded precision Cholesky decomp."""
        nu = np.random.randn(self.times.size)
        if self.lower:
            l_and_u = (1, 0)
        else:
            l_and_u = (0, 1)
        return spl.solve_banded(l_and_u, cb, nu)
    def post_sample(self):
        """Draw sample from Gaussian posterior"""
        return self._sample(self.cprec_post) + self.mu_post
    def prior_sample(self):
        """Draw sample from Gaussian prior"""
        return self._sample(self.cprec_prior)

