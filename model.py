# -*- coding:utf-8 -*-
# 重新写一个交易模拟程序吧，太乱了。

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class simulation(object):
    def __init__(self):
        self.df_data = []    # 交易数据
        self.get_data()

    def get_data(self): # 获取交易数据
        self.df_data = pd.read_csv('000001.csv')
        # 将数据缩小1000倍，方便模拟
        self.df_data['close'] = self.df_data['close']/1000.0
        # 计算日均线值
        MA5 = []
        MA10 = []
        MA30 = []
        MA60 = []
        n = len(self.df_data)
        for i in range(n):
            # 计算MA5
            if i < 5:
                MA5.append(np.mean(self.df_data['close'][0:i]))
            else:
                MA5.append(np.mean(self.df_data['close'][i - 5 + 1:i]))
            # 计算MA10
            if i < 10:
                MA10.append(np.mean(self.df_data['close'][0:i]))
            else:
                MA10.append(np.mean(self.df_data['close'][i - 10 + 1:i]))
            # 计算MA30
            if i < 30:
                MA30.append(np.mean(self.df_data['close'][0:i]))
            else:
                MA30.append(np.mean(self.df_data['close'][i - 30 + 1:i]))
            if i < 60:
                MA60.append(np.mean(self.df_data['close'][0:i]))
            else:
                MA60.append(np.mean(self.df_data['close'][i - 60 + 1:i]))
            if i == 0:
                num = self.df_data['close'][0]
                MA5[0] = MA10[0] = MA30[0] = MA60[0] = num
        self.df_data['MA5'] = MA5
        self.df_data['MA10'] = MA10
        self.df_data['MA30'] = MA30
        self.df_data['MA60'] = MA60

    # 画出数据
    def display_data(self):
        plt.plot(self.df_data['close'])
        plt.plot(self.df_data['MA5'])
        plt.plot(self.df_data['MA10'])
        plt.plot(self.df_data['MA30'])
        plt.plot(self.df_data['MA60'])
        plt.show()

    # 执行模拟
    def run(self):
        pass


simu = simulation()
simu.display_data()
