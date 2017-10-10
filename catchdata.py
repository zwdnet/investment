# -*- coding:utf-8 -*-
# 抓取股票数据，存入csv文件。
import tushare as ts


df = ts.get_k_data('399300', index=True, start='2013-03-15')
df_etf = ts.get_k_data('510300', index=False, start='2013-03-15')
df_SH = ts.get_k_data('000001', index=True, start='2000-01-01')

df.to_csv('399300.csv')
df_etf.to_csv('510300.csv')
df_SH.to_csv('000001.csv')