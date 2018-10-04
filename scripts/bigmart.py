# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 22:27:09 2018

@author: deepak
"""

# Problem: Predict the sales of a Store

import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

pwd = r'E:\DjangoProject\dataScientist\dataset\%s'
df_train = pd.read_csv(pwd % 'train_bigmart.csv')
df_test = pd.read_csv(pwd % 'test_bigmart.csv')

data = df_train.append(df_test)
data.reset_index(drop=True, inplace=True)

data.columns
data.isnull().sum()

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='mean')
data.Item_Weight = imputer.fit_transform(data.loc[:, ['Item_Weight']])

data.Item_Weight.isnull().sum()

data.Outlet_Size.mode()[0]
data.Outlet_Size.isnull().sum()
imputer1 = Imputer(strategy='mode')
data.Outlet_Size = imputer1.fit_transform(data.loc[:, ['Outlet_Size.mode']])