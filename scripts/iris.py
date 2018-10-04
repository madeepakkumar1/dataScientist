# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 11:43:43 2018

@author: deepak
"""

#Iris dataset

#Problem: Predict the class of the flower based on available attributes.
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

pwd = r'E:\DjangoProject\dataScientist\dataset\%s'
dataset = 'iris.data'
df = pd.read_csv(pwd % dataset, names=['sepal length','sepal width','petal length','petal width','target'])
X = df.iloc[:,:4].values
y = df.target

iris = datasets.load_iris()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
onehotencoder = OneHotEncoder()
y = labelencoder.fit_transform(y)

df.describe()
df.groupby('target').size()
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()
df.hist()
plt.show()
from pandas.plotting import scatter_matrix
scatter_matrix(df)
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=7)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))


knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

predict = knn.predict(x_test)

print(accuracy_score(y_test, predict))
print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
from sklearn.metrics import r2_score
print(r2_score(y_test, predict))
