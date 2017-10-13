# -*- coding:utf-8 -*-
# 看上证指数的长期趋势。


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import itertools
from sklearn.linear_model import LinearRegression

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


def LIS(X):
    N = len(X)
    P = [0]*N
    M = [0]*(N+1)
    L = 0
    for i in range(N):
        lo = 1
        hi = L
        while lo <= hi:
            mid = (lo+hi)//2
            if (X[M[mid]] < X[i]):
                lo = mid+1
            else:
                hi = mid-1
        newL = lo
        P[i] = M[newL - 1]
        M[newL] = i

        if (newL > L):
            L = newL

    S = []
    pos = []
    k = M[L]
    for i in range(L-1, -1, -1):
        S.append(X[k])
        pos.append(k)
        k = P[k]
    return S[::-1], pos[::-1]


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
