import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.stats as st
import euler

def metropolis_hastings(p, n):
    alpha, beta = 0., 0.
    samples = np.zeros((n , 2))
    for i in range(n):
        alpha_star, beta_star = np.array([alpha, beta]) + np.random.normal(size = 2)
        if np.random.rand() < p(alpha_star, beta_star) / p(alpha, beta)
            alpha, beta = alpha_star, beta_star
        samples[i] = np.([alpha, beta])
    return samples

test_samples = metropolis_hastings()
