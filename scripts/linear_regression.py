import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pwd = 'E:\DjangoProject\dataScientist\dataset'
dataset = pd.read_csv('%s\Salary_Data.csv' % pwd)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

if dataset.isnull().values.any():
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    x = imputer.fit_transform(x)
#
# from sklearn.cross_validation import train_test_split
# train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)
#
#
# #LinearRegression
#
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression
# regressor.fit(train_x, train_y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

# Feature scaling
# ---------------
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

print(X_train, X_test, y_train, y_test)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

predict = regressor.predict(X_test)