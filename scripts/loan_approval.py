# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 17:08:11 2018

@author: deepak
"""

#Problem: Predict if a loan will get approved or not.
#Note: 1. There are two datasets, need to combined in one dataset(test_loan_predict.csv, train_loan_predict.csv)
#      2. Need to fix the outlier, categorical data, missing value
#      3. Standardize the values
#      4. build a model
import pandas as pd
import matplotlib.pyplot as plt

pwd = r'E:\DjangoProject\dataScientist\dataset\%s'
train_data = 'train_loan_predict.csv'
test_data = 'test_loan_predict.csv'

#Read the datasets
df_train = pd.read_csv(pwd % train_data)
df_test = pd.read_csv(pwd % test_data)

#Checks the null values
df_train.isnull().sum()
df_test.isnull().sum()

df_train['Gender'].value_counts()
df_test['Gender'].value_counts()

#Combine the two dataset
def get_combine():
    df_train1 = pd.read_csv(pwd % train_data)
    df_test1 = pd.read_csv(pwd % test_data)
    target = df_train1.Loan_Status
    df_train1.drop('Loan_Status', 1, inplace=True)
    combined = df_train1.append(df_test1)
    combined.reset_index(inplace=True)
    combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)
    return combined, target

combined_data, target = get_combine()
print(combined_data.describe())

#Handling the Missing values, Use imputer from sklearn
def impute_gender():
    global combined_data
    combined_data['Gender'].fillna('Male', inplace=True)

def impute_marital_status():
    global combined_data
    combined_data['Married'].fillna('Yes', inplace=True)
    
def impute_employment():
    global combined_data
    combined_data['Self_Employed'].fillna('No', inplace=True)
    
def impute_LoanAmount():
    global combined_data
    combined_data['LoanAmount'].fillna(combined_data['LoanAmount'].median(), inplace=True)

def impute_credit_history():
    global combined_data
    combined_data['Credit_History'].fillna(2, inplace=True)
    
combined_data.Credit_History.value_counts()

impute_gender()

impute_marital_status()

impute_employment()

impute_LoanAmount()
impute_credit_history()

combined_data.isnull().sum()

def impute_dependents():
    global combined_data
    combined_data['Dependents'].fillna('', inplace=True)


#handling the categorical data from sklearn.preprocessing import LabelEncoder, OneHotEncoder    
def process_gender():
    global combined_data
    combined_data['Gender'] = combined_data['Gender'].map({'Male':1, 'Female':0})

def process_martial_status():
    global combined_data
    combined_data['Married'] = combined_data['Married'].map({'Yes':1,'No':0})

def process_dependents():
    global combined_data
    combined_data['Singleton'] = combined_data['Dependents'].map(lambda d: 1 if d=='1' else 0)
    combined_data['Small_Family'] = combined_data['Dependents'].map(lambda d: 1 if d=='2' or d == '2+' else 0)
    combined_data['Large_Family'] = combined_data['Dependents'].map(lambda d: 1 if d=='3+' else 0)
    combined_data.drop(['Dependents'], axis=1, inplace=True)

def process_education():
    global combined_data
    combined_data['Education'] = combined_data['Education'].map({'Graduate':1,'Not Graduate':0})

def process_employment():
    global combined_data
    combined_data['Self_Employed'] = combined_data['Self_Employed'].map({'Yes':1,'No':0})

def process_income():
    global combined_data
    combined_data['Total_Income'] = combined_data['ApplicantIncome'] + combined_data['CoapplicantIncome']
    combined_data.drop(['ApplicantIncome','CoapplicantIncome'], axis=1, inplace=True)

def process_loan_amount():
    global combined_data
    combined_data['Debt_Income_Ratio'] = combined_data['Total_Income'] / combined_data['LoanAmount']

combined_data['Loan_Amount_Term'].value_counts()

approved_term = df_train[df_train['Loan_Status']=='Y']['Loan_Amount_Term'].value_counts()
unapproved_term = df_train[df_train['Loan_Status']=='N']['Loan_Amount_Term'].value_counts()
df1 = pd.DataFrame([approved_term,unapproved_term])
df1.index = ['Approved','Unapproved']
df1.plot(kind='bar', stacked=True, figsize=(15,8))

def process_loan_term():
    global combined_data
    combined_data['Very_Short_Term'] = combined_data['Loan_Amount_Term'].map(lambda t: 1 if t<=60 else 0)
    combined_data['Short_Term'] = combined_data['Loan_Amount_Term'].map(lambda t: 1 if t>60 and t<180 else 0)
    combined_data['Long_Term'] = combined_data['Loan_Amount_Term'].map(lambda t: 1 if t>=180 and t<=300  else 0)
    combined_data['Very_Long_Term'] = combined_data['Loan_Amount_Term'].map(lambda t: 1 if t>300 else 0)
    combined_data.drop('Loan_Amount_Term', axis=1, inplace=True)

def process_credit_history():
    global combined_data
    combined_data['Credit_History_Bad'] = combined_data['Credit_History'].map(lambda c: 1 if c==0 else 0)
    combined_data['Credit_History_Good'] = combined_data['Credit_History'].map(lambda c: 1 if c==1 else 0)
    combined_data['Credit_History_Unknown'] = combined_data['Credit_History'].map(lambda c: 1 if c==2 else 0)
    combined_data.drop('Credit_History', axis=1, inplace=True)

def process_property():
    global combined_data
    property_dummies = pd.get_dummies(combined_data['Property_Area'], prefix='Property')
    combined_data = pd.concat([combined_data, property_dummies], axis=1)
    combined_data.drop('Property_Area', axis=1, inplace=True)

process_gender()

process_martial_status()

process_dependents()

process_education()

process_employment()

process_income()

process_loan_amount()

process_loan_term()

process_credit_history()

process_property()

combined_data[60:70]


#Feature scaling, fixing outlier, from sklearn.preprocessing import StandardScaler
def  feature_scaling(df):
    df -= df.min()
    df /= df.max()
    return df

combined_data['LoanAmount'] = feature_scaling(combined_data['LoanAmount'])
combined_data['Total_Income'] = feature_scaling(combined_data['Total_Income'])
combined_data['Debt_Income_Ratio'] = feature_scaling(combined_data['Debt_Income_Ratio'])


combined_data[200:210]


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import numpy as np

def compute_score(clf, x, y, scoring='accuracy'):
    xval = cross_val_score(clf, x, y, cv=5, scoring=scoring)
    return np.mean(xval)

def recover_train_test_target():
    global combined_data
    target1 = target.map({'Y':1, 'N':0})
    train2 = combined_data.head(614)
    test2 = combined_data.iloc[614:]
    return train2, test2, target1

train2, test2, target1 = recover_train_test_target()
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train2, target1)

features = pd.DataFrame()
features['Feature'] = train2.columns
features['Importance'] = clf.feature_importances_
features.sort_values(by=['Importance'], ascending=False, inplace=True)
features.set_index('Feature', inplace=True)

features.plot(kind='bar', figsize=(20,10))

model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train2)
print(train_reduced.shape)

test_reduced = model.transform(test2)
print(test_reduced.shape)
parameters = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

model = RandomForestClassifier(**parameters)
model.fit(train2, target1)

compute_score(model, train2, target1, scoring='accuracy')

output = model.predict(test2).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv(pwd % 'test_loan_predict.csv')
df_output['Loan_ID'] = aux['Loan_ID']
df_output['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output)
df_output[['Loan_ID','Loan_Status']].to_csv('output.csv',index=False)
