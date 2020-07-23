import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

N = 856
beta = 1.0
D = 3.0
gamma = 1.0 / D
S0, I0, R0 = 800, 56, 0
y0 = S0,I0, R0
weeks = 10

def derive(y, N, beta, gamma):
    '''
    calculate S(t)', I(t)', R(t)'
    '''
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def euler_approximate(y0, N, beta, gamma, weeks):
    '''
    Approximate calculation using Euler method
    '''
    ret = []
    S, I, R = y0
    y0 = [S, I, R]
    ret.append(y0)
    for i in range(weeks):
        S = y0[0]
        I = y0[1]
        R = y0[2]
        dS, dI, dR = derive(y0, N, beta, gamma)
        yi = [S+dS, I+dI, R+dR]
        ret.append(yi)
        y0 = yi
    return ret
    
def main():
    res = euler_approximate(y0, N, beta, gamma, weeks)
    df = pd.DataFrame(res, columns=['Nguy cơ', 'Nhiễm bệnh', 'Phục hồi'])
    for index, row in df.iterrows():
        row['Nguy cơ'] = int(row['Nguy cơ']*10000)/10000
        row['Nhiễm bệnh'] = int(row['Nhiễm bệnh']*10000)/10000
        row['Phục hồi'] = int(row['Phục hồi']*10000)/10000
    df.index.names =['Tuần']
    df.plot()
    plt.xticks(list(df.index))
    df.to_csv("euler.csv")

if __name__ == "__main__":
    main()