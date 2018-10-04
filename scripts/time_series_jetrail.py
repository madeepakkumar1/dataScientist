# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 21:35:37 2018

@author: deepak
"""

import  pandas as pd
import matplotlib.pyplot as plt

csv_file = r'E:\DjangoProject\dataScientist\dataset\%s'

df_train = pd.read_csv(csv_file % 'Train_time_series_analysis.csv')
df_test = pd.read_csv(csv_file % 'Test_time_series_analysis.csv')

df_train.dtypes 
df_test.dtypes

df_train.Datetime = pd.to_datetime(df_train.Datetime, format="%d-%m-%Y %H:%M")
df_test.Datetime = pd.to_datetime(df_test.Datetime, format="%d-%m-%Y %H:%M")

df_train.dtypes
df_test.dtypes

for i in (df_train, df_test):
    i['year']=i.Datetime.dt.year 
    i['month']=i.Datetime.dt.month 
    i['day']=i.Datetime.dt.day
    i['Hour']=i.Datetime.dt.hour 

df_train['day of week'] = df_train['Datetime'].dt.dayofweek
temp = df_train['Datetime']

def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0
temp2 = df_train['Datetime'].apply(applyer)
df_train['weekend'] = temp2

df_train.index = df_train.Datetime
df = df_train.drop('ID', axis=1)
ts = df.Count
plt.figure(figsize=(16,10))
plt.plot(ts, label="Passenger Count")

df.columns
df.groupby('year')['Count'].mean().plot.bar()
df.groupby('month')['Count'].mean().plot.bar()

temp3 = df.groupby(['year', 'month'])['Count'].mean()
temp3.plot()

df.groupby('day')['Count'].mean().plot.bar()

df.groupby('Hour')['Count'].mean().plot.bar()

df.groupby('weekend')['Count'].mean().plot.bar()

df.groupby('day of week')['Count'].mean().plot.bar()

