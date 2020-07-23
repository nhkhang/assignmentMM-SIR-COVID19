import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import random
import math
from mcmc import *

# Prepare data
confirmed_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
recovered_path = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
death_path     = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'

confirmed_df = pd.read_csv(confirmed_path)
recovered_df = pd.read_csv(recovered_path)
death_df     = pd.read_csv(death_path)

confirmed_table = confirmed_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Confirmed").fillna('').drop(['Lat', 'Long'], axis=1)
death_table = death_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Deaths").fillna('').drop(['Lat', 'Long'], axis=1)
recovered_table = recovered_df.melt(id_vars=["Province/State", "Country/Region", "Lat", "Long"], var_name="Date", value_name="Recovered").fillna('').drop(['Lat', 'Long'], axis=1)

full_table = confirmed_table.merge(death_table).merge(recovered_table)
full_table['Date'] = pd.to_datetime(full_table['Date'])

def get_time_series(country):
    if full_table[full_table['Country/Region'] == country]['Province/State'].nunique() > 1:
        country_table = full_table[full_table['Country/Region'] == country]
        country_df = pd.DataFrame(pd.pivot_table(country_table, values = ['Confirmed', 'Deaths', 'Recovered'], index='Date', aggfunc=sum).to_records())
        return country_df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered']]

    df = full_table[(full_table['Country/Region'] == country) & (full_table['Province/State'].isin(['', country]))]
    return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered']]

def get_time_series_province(province):
    df = full_table[(full_table['Province/State'] == province)]
    return df.set_index('Date')[['Confirmed', 'Deaths', 'Recovered']]
  
def get_dataframe(df):
    if len(df) > 1 and df.iloc[-2,0] >= df.iloc[-1,0]:
        df.drop(df.tail(1).index,inplace=True)

    susceptible = []
    for idx, row in df.iterrows():
        s = N - row['Recovered'] - row['Confirmed'] - row['Deaths']
        susceptible.append(s)
    df = df.assign(Susceptible = susceptible)
    return df

country = 'Germany'
df = drop(get_time_series(country))
N = 83020000 # Germany population
betas = []
gammas = []
pi_distribution = st.uniform
init_sample = pi_distribution.rvs(loc=0, scale=1, size=2)

# Sampling beta and gamma
samples = metropolis_hastings(pgauss, init_sample, sample_by_gauss , iter=1000, pi=pi_distribution.pdf) # samples of beta, gamma

# Calculate E(r0)
r0 = 0 # E(r0)
Xs = df.iloc[5:]['Confirmed']
pi = 0
for i in range(len(Xs)):
    beta = samples[i][0]
    gamma = samples[i][1]
    X = Xs[i]
    if gamma != 0:
        pi = pow(gamma, beta) * pow(X, (beta-1)) * math.exp(-gamma*X) / math.gamma(beta)
        r0 += pi * beta / gamma
