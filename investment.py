# -*- coding:utf-8 -*-
# 指数基金定投模拟程序

import tushare as ts
import matplotlib.pyplot as pyplot
import pandas as pd

df = pd.read_csv('399300.csv')
df_etf = pd.read_csv('510300.csv')
df_SH = pd.read_csv('000001.csv')
df_SH['close'] = df_SH['close']/1000.0 # 因为指数数值太大了，缩小1000倍，方便模拟。

p1 = pyplot.subplot(311)
p2 = pyplot.subplot(312)
p3 = pyplot.subplot(313)
p1.plot(df['close'])
p2.plot(df_etf['close'])
p3.plot(df_SH['close'])
pyplot.show()

print(len(df), len(df_etf), len(df_SH))

import numpy as np

print("沪深300指数与沪深300etf的相关性分析")
# covariance = np.cov(df['close'], df_etf['close'])
# print(covariance)
corrc = np.corrcoef(df['close'], df_etf['close'])
print(corrc)

print("上证综指指数与沪深300etf的相关性分析")
# covariance = np.cov(df_etf['close'][0:len(df_SH)], df_SH['close'])
# print(covariance)
corrc = np.corrcoef(df_etf['close'], df_SH['close'][len(df_SH)-len(df_etf):])
print(corrc)

# nask_data = df_nask['close'].values[::-1]
# print(len(df['close']))
# print(len(nask_data))
# covariance2 = np.cov(df_etf['close'][0:732], nask_data)
# print(covariance2)
# corrc2 = np.corrcoef(df_etf['close'][0:732], nask_data)
# print(corrc2)

# 定投模拟情况
# 1.初始日期开始，每隔30天投一次，计算总盈利。
n = len(df_SH['close'])
num = 1000.0  # 每次定投投入金额
fee = 3.0/10000.0 # 手续费率
total = [] # 每一期的总市值
number = 0 # 持股总数
stock = [] # 每一期购入的股票总数
invest = 0 # 总投入资金
rate = [] # 每期的收益率
cost = [] # 累计交易成本
costrate = [] # 每期的累计成本率

j = 0
for i in range(0, n, 30):
    stock.append(int((num/df_SH['close'].values[i])/100)*100)
    number += stock[j]
    investnow = stock[j] * df_SH['close'].values[i]   # 本期买的股票市值
    costnow = investnow * fee
    if costnow < 0.1:
        costnow = 0.1
    if j == 0:
        cost.append(costnow)
    else:
        cost.append(cost[j-1] + costnow)
    all = number * df_SH['close'].values[i] + (num - investnow)  - costnow  # 股票总市值与现金之和，扣除了成本。
    total.append(all)
    invest += num
    rate.append(total[j]/invest)
    costrate.append(cost[j]/float(invest))
    print(i, total[j], rate[j], costrate[j]) # 输出每期投资后的总市值以及收益率
    j = j+1
p3 = pyplot.subplot(4,1,1)
p4 = pyplot.subplot(4,1,2)
p5 = pyplot.subplot(4,1,3)
p6 = pyplot.subplot(4,1,4)
p3.plot(total)
p4.plot(rate)
p5.plot(cost)
p6.plot(costrate)
pyplot.show()