# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:39:33 2018

@author: deepak
"""

import pandas as pd
import matplotlib.pyplot as plt

csv_file = r'E:\DjangoProject\dataScientist\dataset\%s'

df_train = pd.read_csv(csv_file % 'train_boston_housing.csv')

df_test = pd.read_csv(csv_file % 'test_boston_housing.csv')

data = df_train.append(df_test)

data.reindex

df_train.columns

df_train = df_train.drop('ID', axis=1)

df_train.head()

df_train.describe()

import seaborn as sns
sns.pairplot(df_train, size=1.5);
plt.show()

df_train.columns
col_study = ['zn', 'indus', 'nox', 'rm',]
sns.pairplot(df_train[col_study], size=1.5);
plt.show()

col_assign = ['ptratio', 'black', 'medv']
sns.pairplot(df_train[col_assign], size=1.5)
plt.show()

df_train.corr()
pd.options.display.float_format = '{:,.3f}'.format
df_train.corr()

plt.figure(figsize=(16,10))
sns.heatmap(df_train.corr(), annot=True)
plt.show()

X = df_train.rm.values.reshape(-1,1)
y = df_train.medv.values

from sklearn.linear_model import LinearRegression

regresor = LinearRegression()
regresor.fit(X, y)
regresor.coef_
regresor.intercept_

plt.figure(figsize=(12,10))

sns.regplot(X,y)

sns.jointplot(x = 'rm', y='medv', data=df_train, kind='reg', size=10)