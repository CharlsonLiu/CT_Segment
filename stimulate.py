from math import exp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def algorithm_1(n):
    t = 0.2 * n * np.log(n) + 0.5 * n**0.5 * np.sin(n/20) +\
        np.random.normal(0, 5)*np.random.poisson(lam=2.6,size=None)*0.05*exp(0.001*n)
    noise = np.random.normal(0, 20)
    t += noise
    if t < 0:
        t = 0
    if t > 800000:
        t = 800000
    t = (t + np.random.poisson(lam=1.5,size=None))*0.76
    return t

def algorithm_2(n):
    t = 0.1 * n * np.log(n) + 0.5 * n**0.5 * np.sin(n/20) + \
        np.random.normal(0, 5)*np.random.poisson(lam=4,size=None)*0.005** n * np.log(2*n)
    noise = np.random.normal(0, 30)
    t += noise
    if t < 0:
        t = 0
    if t > 480000:
        t = 480000
    t = (t + np.random.poisson(lam=2.3,size=None))*0.751
    return t

def algorithm_3(n):
    t = 0.05 * n**2 + 100 * np.sin(n/20) + np.random.normal(0, 100)
    noise = np.random.normal(0, 200)
    t += noise
    if t < 0:
        t = 0
    if t > 360000:
        t = 360000
    t =  t*0.08 + np.random.poisson(lam=1.5,size=None)
    return t

def algorithm_4(n):
    t = 0.02 * n**2 + np.random.normal(0, 50) + \
        np.random.poisson(lam=15 , size=None) * np.exp(0.005*n) + \
        np.random.normal(0, 30)*np.random.poisson(lam=4,size=None)*0.005** n * np.log(2*n)

    noise = np.random.normal(0, 50)
    t += noise
    if t < 0:
        t = 0
    if t > 800000:
        t = 800000
    t = t*0.145 + np.random.poisson(lam=3 , size=None)
    return t

df = pd.DataFrame(columns=['algorithm_1', 'algorithm_2', 'algorithm_3', 'algorithm_4'])

for n in range(1, 601):
    df.loc[n] = [algorithm_1(n), algorithm_2(n), algorithm_3(n), algorithm_4(n)]

df.to_csv('processing_times.csv', index_label='Number of Images')

plt.plot(df['algorithm_1'], label='algorithm_1')
plt.plot(df['algorithm_2'], label='algorithm_2')
plt.plot(df['algorithm_3'], label='algorithm_3')
plt.plot(df['algorithm_4'], label='algorithm_4')
plt.legend()
plt.xlabel('Number of Images')
plt.ylabel('Processing Time (seconds)')
plt.show()