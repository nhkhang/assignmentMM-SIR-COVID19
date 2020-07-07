import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def deriv(y, t, N, beta, gamma):
    '''
    calculate S(t)', I(t)', R(t)'
    '''
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def euler_cal():
    pass

def bayes_inference():


def __main__():
    N = 1000
    beta = 1.0
    D = 30.0
    gamma = 1.0 / D
    S0, I0 = 800, 7lkjhgf\=-0987653421  
    euler_cal()