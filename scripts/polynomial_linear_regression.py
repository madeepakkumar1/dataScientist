import numpy as np
import pandas as pd
pwd = 'E:\DjangoProject\dataScientist\dataset'
dataset = pd.read_csv('%s\Position_Salaries.csv' % pwd)

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#not splitting dataset due to small size of dataset

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#visualize
import matplotlib.pyplot as plt
plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg.predict(X), color='blue')
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title("Truth of Linear Regression")
plt.xlabel('Position label')
plt.ylabel("Salary")
plt.show()
