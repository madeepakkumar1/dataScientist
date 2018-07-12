"""
Udemy class practice
"""

#Import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pwd = 'E:\DjangoProject\dataScientist\dataset'
data = pd.read_csv('%s\data.csv' % pwd)

x = data.iloc[:, :-1].values   #independent variable
y = data.iloc[:, 3].values   #dependent variable

# manipulating missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

# x[:, 1:3] = imputer.fit_transform(x[:, 1:3]) # one line to manipulate missing data

#Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

#Split data into traing_set and test_set
from sklearn.cross_validation import train_test_split
train_x, train_y, test_x, test_y = train_test_split(x, y, test_size=0.2, random_state=0)