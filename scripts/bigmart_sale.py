# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:16:37 2018

@author: kumadee
"""

#Problem: predict the big mart sale
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

pwd = r'C:\Users\kumadee\Downloads\datasets\%s'

df_train = pd.read_csv(pwd % 'train.csv')
#df_test = pd.read_csv(pwd % 'test.csv')


#df_train = df_train.drop('Item_Outlet_Sales', axis=1)

# data = pd.concat([df_train, df_test], ignore_index=True)
# data = df_train.append(df_test)
data = df_train

data['Item_Fat_Content'].unique()

data.Outlet_Establishment_Year.unique()

data.Outlet_Size.unique()
data.describe()

data.Item_Visibility.hist(bins=20)


data['Item_Fat_Content'].value_counts()
data.Outlet_Size.value_counts()

data.Outlet_Size.mode()[0]

data.Outlet_Size = data.Outlet_Size.fillna(data.Outlet_Size.mode()[0])

data.Outlet_Size.isnull().sum()

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean')

imputer = imputer.fit(data.iloc[:, 1:2])

data['Item_Weight'] = imputer.transform(data.iloc[:, [1]])

data['Item_Weight'].sum()

Q1 = data['Item_Visibility'].quantile(0.25)
Q3 = data['Item_Visibility'].quantile(0.75)
IQR = Q3 - Q1
filt_train = data.query('(@Q1 - 1.5 * @IQR) <= Item_Visibility <= (@Q3 + 1.5 * @IQR)')

print(filt_train.shape)
print(df_train.shape)
train = filt_train

print(train.shape)

train.columns

train['Item_Visibility_bins'] = pd.cut(train['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])
train['Item_Visibility_bins'] 
train['Item_Visibility_bins'] = train['Item_Visibility_bins'].replace(np.NaN, 'Low Viz')
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace('reg', 'Regular')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['Item_Fat_Content'].unique()
train['Item_Fat_Content'] = le.fit_transform(train['Item_Fat_Content'])
train['Item_Visibility_bins'] = le.fit_transform(train['Item_Visibility_bins'])

train['Outlet_Size'] = le.fit_transform(train['Outlet_Size'])

train['Outlet_Location_Type'] = le.fit_transform(train['Outlet_Location_Type'])

dummy = pd.get_dummies(train['Outlet_Type'])
dummy.head()
train = pd.concat([train, dummy], axis=1)
train.corr()[((train.corr() < -0.85) | (train.corr() > 0.85)) & (train.corr() != 1)]
train.dtypes

train = train.drop(['Item_Identifier', 'Item_Type', 'Outlet_Identifier', 'Outlet_Type'], axis=1)
train.columns

X = train.drop('Item_Outlet_Sales', axis=1)

y = train.Item_Outlet_Sales

train.columns


#Test data set
test = pd.read_csv(pwd % 'Test.csv')
test['Outlet_Size'] = test['Outlet_Size'].fillna('Medium')
test.Outlet_Size.isnull().any()

test['Item_Visibility_bins'] = pd.cut(test['Item_Visibility'], [0.000, 0.065, 0.13, 0.2], labels=['Low Viz', 'Viz', 'High Viz'])

test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].mean())

test['Item_Visibility_bins'] = test['Item_Visibility_bins'].replace(np.NaN, 'Low Viz')
test['Item_Visibility_bins'].head()

test['Item_Fat_Content'] = test['Item_Fat_Content'].replace(['low fat', 'LF'], 'Low Fat')
test['Item_Fat_Content'] = test['Item_Fat_Content'].replace('reg', 'Regular')


#Label Encoding 
test['Item_Fat_Content'] = le.fit_transform(test['Item_Fat_Content'])


