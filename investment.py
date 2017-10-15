# -*- coding:utf-8 -*-
# 指数基金定投模拟程序

import tushare as ts
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from LIS import LIS
from sklearn.linear_model import LinearRegression

df = pd.read_csv('399300.csv')
df_etf = pd.read_csv('510300.csv')
df_SH = pd.read_csv('000001.csv')
df_SH['close'] = df_SH['close'] / 1000.0  # 因为指数数值太大了，缩小1000倍，方便模拟。
df_SH['low'] = df_SH['low'] / 1000.0
df_SH['high'] = df_SH['high'] / 1000.0
df_SH['open'] = df_SH['open'] / 1000.0

# 处理数据，计算5日均值，10日均值，30日均值，60日均值
def dataDeal(df_data):
    MA5 = []
    MA10 = []
    MA30 = []
    MA60 = []
    n = len(df_data)
    for i in range(n):
        # 计算MA5
        if i < 5:
            MA5.append(np.mean(df_data['close'][0:i]))
        else:
            MA5.append(np.mean(df_data['close'][i - 5 + 1:i]))
        # 计算MA10
        if i < 10:
            MA10.append(np.mean(df_data['close'][0:i]))
        else:
            MA10.append(np.mean(df_data['close'][i - 10 + 1:i]))
        # 计算MA30
        if i < 30:
            MA30.append(np.mean(df_data['close'][0:i]))
        else:
            MA30.append(np.mean(df_data['close'][i - 30 + 1:i]))
        # 计算MA60
        if i < 60:
            MA60.append(np.mean(df_data['close'][0:i]))
        else:
            MA60.append(np.mean(df_data['close'][i - 60 + 1:i]))
        if i == 0:
            num = df_data['close'][0]
            MA5[0] = MA10[0] = MA30[0] = MA60[0] = num
    df_data['MA5'] = MA5
    df_data['MA10'] = MA10
    df_data['MA30'] = MA30
    df_data['MA60'] = MA60


dataDeal(df_SH)
print(df_SH.head())

p1 = plt.subplot(311)
p2 = plt.subplot(312)
p3 = plt.subplot(313)
p1.plot(df['close'])
p2.plot(df_etf['close'])
p3.plot(df_SH['close'])
p3.plot(df_SH['MA5'])
p3.plot(df_SH['MA10'])
p3.plot(df_SH['MA30'])
p3.plot(df_SH['MA60'])
plt.show()

print(len(df), len(df_etf), len(df_SH))

import numpy as np

print("沪深300指数与沪深300etf的相关性分析")
corrc = np.corrcoef(df['close'], df_etf['close'])
print(corrc)

print("上证综指指数与沪深300etf的相关性分析")
corrc = np.corrcoef(df_etf['close'], df_SH['close'][len(df_SH) - len(df_etf):])
print(corrc)


# 定投模拟情况
# 1.初始日期开始，每隔30天投一次，计算总盈利。

class Module(object):
    def __init__(self, df_data):
        self.n = len(df_data['close'])
        self.df_data = df_data
        self.num = 1000.0  # 每次定投投入金额
        self.fee = 3.0 / 10000.0  # 手续费率
        self.total = []  # 每一期的总市值
        self.number = 0  # 持股总数
        self.stock = []  # 每一期购入的股票总数
        self.invest = 0  # 总投入资金
        self.input = []  # 每期累计投入的总资金
        self.rate = []  # 每期的收益率
        self.cost = []  # 累计交易成本
        self.costrate = []  # 每期的累计成本率
        # 评价模型的指标
        self.maxDrawdown = 0.0  # 最大回撤

    # 运行模拟
    def run(self):
        j = 0
        for i in range(0, self.n, 30):
            self.stock.append(int((self.num / self.df_data['close'].values[i]) / 100) * 100)
            self.number += self.stock[j]
            self.investnow = self.stock[j] * self.df_data['close'].values[i]  # 本期买的股票市值
            self.costnow = self.investnow * self.fee
            if self.costnow < 0.1:
                self.costnow = 0.1
            if j == 0:
                self.cost.append(self.costnow)
            else:
                self.cost.append(self.cost[j - 1] + self.costnow)
            all = self.number * self.df_data['close'].values[i] + \
                  (self.num - self.investnow) - self.costnow  # 股票总市值与现金之和，扣除了成本。
            self.total.append(all)
            self.invest += self.num
            self.input.append(self.invest)
            self.rate.append((self.total[j] - self.invest) / self.invest)
            self.costrate.append(self.cost[j] / float(self.invest))
            # print(i, self.total[j], self.rate[j], self.costrate[j]) # 输出每期投资后的总市值以及收益率
            j = j + 1

    # 作图
    def draw(self):
        # plt.rcParams['font.sans-serifSj'] = ['SimHei']
        p1 = plt.subplot(4, 1, 1)
        p2 = plt.subplot(4, 1, 2)
        p3 = plt.subplot(4, 1, 3)
        p4 = plt.subplot(4, 1, 4)
        p1.plot(self.total)
        p1.plot(self.input)
        p1.set_title("All money of graph")
        p2.plot(self.rate)
        p2.set_title("All interst of graph")
        p3.plot(self.cost)
        p3.set_title("Cost graph")
        p4.plot(self.costrate)
        p4.set_title("Cost rate graph")
        plt.show()

    # 评价模型
    def Judge(self):
        # 计算最大回撤
        maxValue = max(self.total)
        maxIndex = self.total.index(maxValue)
        minValue = min(self.total[maxIndex:])
        self.maxDrawdown = (maxValue - minValue) / minValue

    # 返回模拟结果
    def getResult(self):
        self.Judge()
        return self.total[-1], self.rate[-1], self.costrate[-1], self.maxDrawdown


module = Module(df_SH)
module.run()
module.draw()
result = module.getResult()
print(result)


# 另一个模拟，增加止盈和止损
class Module2(Module):
    def __init__(self, df_data):
        Module.__init__(self, df_data)
        self.bV = []  # 历史底部值
        self.bP = []  # 历史底部位置
        self.highp = df_data['high'].values
        self.lowp = df_data['low'].values
        self.openp = df_data['open'].values
        self.closep = df_data['close'].values
        # 算历史底部的数据
        self.xt = []
        self.est = []
        self.d = []
        self.idx = []

    def findBottom(self):
        for i in range(1, len(self.highp) - 1):
            if (self.highp[i] <= self.highp[i - 1] and
                        self.highp[i] < self.highp[i + 1] and
                        self.lowp[i] <= self.lowp[i - 1] and
                        self.lowp[i] < self.lowp[i + 1]):
                self.bV.append(self.lowp[i])
                self.bP.append(i)
        self.d,p = LIS(self.bV)
        self.idx = []
        for i in range(len(p)):
            self.idx.append(self.bP[p[i]])
        lr = LinearRegression()
        X = np.atleast_2d(np.array(self.idx)).T
        Y = np.array(self.d)
        lr.fit(X, Y)
        self.xt = np.atleast_2d(np.linspace(0, len(self.closep)+200, len(self.closep)+200)).T
        self.estV = lr.predict(self.xt)

    def draw(self):
        # self.findBottom()
        plt.plot(self.closep)
        plt.plot(self.idx, self.d, 'ko')
        plt.plot(self.xt, self.estV, '-r', linewidth=5)
        plt.show()


module2 = Module2(df_SH)
module2.findBottom()
module2.draw()
