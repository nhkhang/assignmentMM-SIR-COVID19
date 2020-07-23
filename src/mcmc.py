import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy.stats as st
import euler

# def metropolis_hastings(p, n):
#     alpha, beta = 0., 0.
#     samples = np.zeros((n , 2))
#     for i in range(n):
#         alpha_star, beta_star = np.array([alpha, beta]) + np.random.gamma(size = 2)
#         if np.random.rand() < min(1, p(alpha_star, beta_star) / p(alpha, beta)):
#             alpha, beta = alpha_star, beta_star
#         samples[i] = np.array([alpha, beta])
#     return samples

def check_hist(df):
    pass

def sample_by_gauss(s):
  return np.random.normal(loc=mus, scale=sigmas, size=s)

def pgauss(x, y):
    return st.multivariate_normal.pdf([x, y], mean=mus, cov=sigmas)

def metropolis_hastings(p, sample, iter=1000):
    init_sample = sample(s=2)
    x = init_sample[0]
    y = init_sample[1]
    samples = np.zeros((iter, 2))
    for i in range(iter):
        x_star, y_star = np.array([x, y]) + sample(s=2)
        r = min(1, p(x_star, y_star) / p(x, y))
        if np.random.uniform(low=0.0, high=1.0, size=1) <= r:
            x, y = x_star, y_star
        samples[i] = np.array([x, y])

    return samples

mus = np.array([0.002, 0.5])
sigmas = np.array([[1, 0.9], [0.9, 1]])
samples = metropolis_hastings(pgauss, sample_by_gauss, iter=10000)
print(samples)

# data_folder = Path(__file__).parent.absolute().parent.absolute() / "data"
# data_file = data_folder / "confirmed_global.csv"
# df = pd.read_csv(data_file)
# df = df.drop(columns=['Lat','Long', 'Province/State', 'Country/Region'])
# germany = df.iloc[[120]]
# data = germany.iloc[[0]].values
# data = data.T
# df_germany = pd.DataFrame(data)
# df_germany.plot()
# plt.show()



# test_samples = metropolis_hastings()
