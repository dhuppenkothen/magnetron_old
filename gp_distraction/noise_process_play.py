# Quick sanity tests of noise_process.py
# Iain Murray, Jan 2014

from noise_process import NoiseModel
import numpy as np
import scipy as sp

# Compute Gaussian process covariance using pedestrian code and find log-prob of
# observations, for comparison to banded matrix computations on corresponding
# MRF precision.
#times = np.arange(5.0)
times = np.sort(np.random.randn(5))
N = times.size
lengthscale = 0.8
amplitude = 1.5
# These precisions were too high for my posterior-checking Markov chain to mix.
#obs_prec = 1.0 / (0.1*(1.0 + np.random.rand(N)))**2
obs_prec = (0.2 + 0.01*np.random.rand(N))**2

naive_K = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        naive_K[i,j] = (amplitude**2) * \
                np.exp(-np.abs(times[i]-times[j]) / lengthscale)
    naive_K[i,i] += 1.0 / obs_prec[i]
(sign, logdet) = np.linalg.slogdet(naive_K)
assert(sign > 0)
naive_P = np.linalg.inv(naive_K)
# banded = np.triu(np.tril(naive_P, 1), -1)
# Check (to within "numerical noise") that precision is banded:
# print('naive_P largest abs off_band =', np.abs(naive_P - banded).max())
# Isn't banded now I've added obs noise to diagonal.

yy = sp.random.multivariate_normal(np.zeros(N), naive_K)
#yy += np.random.randn(N) / np.sqrt(obs_prec)

naive_Lprob_data = -0.5*np.dot(yy, np.dot(naive_P, yy)) \
        - 0.5*N*np.log(2*np.pi) - 0.5*logdet
print('naive Lprob_data =', naive_Lprob_data)

nm = NoiseModel(times, yy, obs_prec, lengthscale, amplitude)
print('banded Lprob_data =', nm.Lprob_data)

# Check prior sampler:
# If I sample points, they should have roughly the covariance I expect:
S = 100000
xx = np.zeros((S,times.size))
for i in range(S):
    xx[i] = nm.prior_sample()
C = np.cov(xx, rowvar=0)
print(C[0:5,0:5])
print(naive_K[0:5,0:5] - np.diag(1.0 / obs_prec))


# Check posterior sampler:
# If sample from the joint prior over the process and observations, by sampling
# from the posterior, given observations, sampling new observations, and
# iterating, then the process should have roughly the covariance I expect.
# (Geweke's "Getting it Right")
S = 100000
xx = np.zeros((S,times.size))
for i in range(S):
    xx[i] = nm.post_sample()
    nm.update_obs(xx[i] + np.random.randn(times.size)/np.sqrt(nm.obs_prec))
C = np.cov(xx, rowvar=0)
print(C[0:5,0:5])


