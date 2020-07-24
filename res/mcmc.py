import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.stats as st

def pgauss(x, y, mus, sigmas):
    return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)

def sample_by_gauss(mus, sigmas):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    return np.random.multivariate_normal(mean=mus, cov=sigmas)

def metropolis_hastings(p, init, sample_func, iter, pi, mus, sigmas):
    '''
    Metropolis hastings algorithm for sampling beta, gamma
    '''
    init_sample = init
    x = init_sample[0]
    y = init_sample[1]
    samples = np.zeros((iter, 2))
    mus_star = [0, 0]
    mus = mus_star
    for i in range(iter):
        if i > 0:
          mus = mus_star
          mus_star = samples[i - 1]
        x_star, y_star = sample_func(mus=mus_star, sigmas=sigmas)
        eta = p(x, y, mus, sigmas) / p(x_star, y_star, mus_star, sigmas)
        eta = eta *  pi(abs(x_star- y_star)) / pi(abs(x - y))
        r = min(eta, 1)
        if np.random.uniform(low=0.0, high=1.0, size=1) <= r:
            x, y = x_star, y_star
        samples[i] = np.array([x, y])
    return samples

pi_distribution = st.uniform
init_sample = pi_distribution.rvs(loc=0, scale=1, size=2)
samples = metropolis_hastings(pgauss, init_sample, sample_by_gauss , iter=1000, pi=pi_distribution.pdf, mus=[0,0], sigmas=[[1, 0],  [0, 1]]) # samples of beta, gamma
