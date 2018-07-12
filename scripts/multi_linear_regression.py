import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pwd = 'E:\DjangoProject\dataScientist\dataset'
dataset = pd.read_csv('%s\\50_Startups.csv' % pwd)

x = dataset.iloc[:, :-1].values   #independent variable
y = dataset.iloc[:, 4].values    # dependent variable

if dataset.isnull().values.any():
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
    x = imputer.fit_transform(x)

#Categorical variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
x= onehotencoder.fit_transform(x).toarray()

#Avoiding dummy variable trap
x = x[:, 1:]
# #LinearRegression
#
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression
# regressor.fit(train_x, train_y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

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

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
