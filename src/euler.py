import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

N = 999
beta = 1.0
D = 3.0
gamma = 1.0 / D
S0, I0, R0 = 999, 1, 0
y0 = S0,I0, R0
days = 20

def deriv(y, t, N, beta, gamma):
    '''
    calculate S(t)', I(t)', R(t)'
    '''
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def euler_cal(y0, N, beta, gamma, days):
    '''
    Approximate calculation using Euler method
        y1 = y0 + h*f(y0)
    y0 = S0, I0, R0
    step_size h = 1
    '''
    ret = []
    S, I, R = y0
    y0 = [S, I, R]
    ret.append(y0)
    for i in range(days):
        S = y0[0]
        I = y0[1]
        R = y0[2]
        dS, dI, dR = deriv(y0, None, N, beta, gamma)
        yi = [S+dS, I+dI, R+dR]
        ret.append(yi)
        y0 = yi
    return ret

def main():
    res = euler_cal(y0, N, beta, gamma, days)
    print(len(res))
    print(len(res[0]))
    df = pd.DataFrame(res, columns=['Nguy cơ', 'Nhiễm bệnh', 'Phục hồi'])
    df.index.names =['Ngày']
    df.plot()
    plt.show()
    df.to_csv("euler.csv")

if __name__ == "__main__":
    main()