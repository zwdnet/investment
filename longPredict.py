# -*- coding:utf-8 -*-
# 看上证指数的长期趋势。


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import itertools
from sklearn.linear_model import LinearRegression
from LIS import LIS

df_SH = pd.read_csv('000001.csv')

highp = df_SH['high'].values
lowp = df_SH['low'].values
openp = df_SH['open'].values
closep = df_SH['close'].values

lr = LinearRegression()
x = np.atleast_2d(np.linspace(0, len(closep), len(closep))).T
lr.fit(x, closep)
print(lr)
xt = np.atleast_2d(np.linspace(0, len(closep)+200, len(closep)+200)).T
yt = lr.predict(xt)
plt.plot(xt, yt, '-g', linewidth=5)
plt.plot(closep)
plt.show()


def LIS2(X):
    n = len(X)
    m = [0]*n
    result = []
    pos = []
    for x in range(n-2, -1, -1):
        for y in range(n-1, x, -1):
            if X[x] < X[y] and m[x] <= m[y]:
                m[x] += 1
        max_value = max(m)
        for i in range(n):
            if m[i] == max_value:
                result.append(X[i])
                pos.append(i)
                max_value -= 1
    return result, pos



bV = []
bP = []
for i in range(1, len(highp)-1):
    if (highp[i] <= highp[i-1] and
       highp[i] < highp[i+1] and
       lowp[i] <= lowp[i-1] and
       lowp[i] < lowp[i+1]):
       bV.append(lowp[i])
       bP.append(i)
d,p = LIS(bV)
idx = []
for i in range(len(p)):
    idx.append(bP[p[i]])

plt.plot(closep)
plt.plot(idx, d, 'ko')
plt.show()

lr = LinearRegression()
X = np.atleast_2d(np.array(idx)).T
Y = np.array(d)
lr.fit(X, Y)
estV = lr.predict(xt)

plt.plot(closep)
plt.plot(idx, d, 'ko')
plt.plot(xt, estV, '-r', linewidth=5)
plt.plot(xt, yt, '-g', linewidth=5)
plt.show()
