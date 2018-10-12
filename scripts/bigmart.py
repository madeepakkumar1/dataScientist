import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


file = r'C:\Users\kumadee\Desktop\assignment2-kmart_sales_forecast\%s'

train_df = pd.read_csv(file % 'train_kmart.csv')
test_df = pd.read_csv(file % 'test_kmart.csv')

train_df.head()
train_df.describe()
train_df.columns

# correlation
sns.heatmap(train_df.corr())

#train_df.Outlet_Size.unique()
# train_df.Outlet_Size.value_counts()
pd.Categorical(train_df.Outlet_Size).describe()
train_df.Outlet_Size = train_df.Outlet_Size.fillna('High', limit=565)
train_df.Outlet_Size = train_df.Outlet_Size.fillna('Medium', limit=923)
train_df.Outlet_Size = train_df.Outlet_Size.fillna('Small')
pd.Categorical(train_df.Outlet_Size).describe()

 pd.Categorical(train_df.Item_Fat_Content).describe()
 train_df.Item_Fat_Content = train_df.Item_Fat_Content.replace(('LF', 'low fat'), 'Low Fat')
 train_df.Item_Fat_Content = train_df.Item_Fat_Content.replace('reg', 'Regular')
 pd.Categorical(train_df.Item_Fat_Content).describe()
 
 
#Counting NA for Item_Weight
# train_df.Item_Weight.isnull().value_counts()[1]
sum(np.isnan(train_df.Item_Weight))
#Fill NA with mean value
train_df.Item_Weight = train_df.Item_Weight.fillna(train_df.Item_Weight.mean())

#Find counts of zeros
train_df[train_df.Item_Visibility == 0].Item_Visibility.value_counts()
# Replacing zeros with mean value 
train_df.Item_Visibility = train_df.Item_Visibility.replace(0, train_df.Item_Visibility.mean())
plt.hist(train_df.Item_Visibility)


mapping_Item_Fat_Content = {'Regular': 1, 'Low Fat': 0}
train_df.Item_Fat_Content = train_df.Item_Fat_Content.map(mapping_Item_Fat_Content)

mapping_Outlet_Size = {'Small': 0, "Medium": 1,'High': 2}
train_df.Outlet_Size = train_df.Outlet_Size.map(mapping_Outlet_Size)

pd.Categorical(train_df.Outlet_Type).describe()
mapping_Outlet_Type = {'Grocery Store': 0, 'Supermarket Type1': 0, 'Supermarket Type2': 1, 'Supermarket Type3': 2}
train_df.Outlet_Type = train_df.Outlet_Type.map(mapping_Outlet_Type)

# handing Outlet_Location_Type
train_df.Outlet_Location_Type.unique()
pd.Categorical(train_df.Outlet_Location_Type).describe()
mapping_Outlet_Location_Type = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
train_df.Outlet_Location_Type = train_df.Outlet_Location_Type.map(mapping_Outlet_Location_Type)

y = train_df.Item_Outlet_Sales
x = pd.get_dummies(train_df.Item_Type)

train_df = pd.concat([train_df.iloc[:,1:], x], axis=1)
train_df = train_df.drop(['Item_Type'], axis=1)
x = pd.get_dummies(train_df.Outlet_Identifier)
train_df = pd.concat([train_df, x], axis=1)
train_df = train_df.drop(['Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)


# Dividing data set into two parts train set and test set
train_x, test_x, train_y, test_y = train_test_split(train_df, y)


#Building a LinearRegresion model
lr = LinearRegression()
lr.fit(train_x, train_y)
# lr.coef_
pd.DataFrame(lr.coef_, train_df.columns, columns=['Coefficient']) #find the cofficient of independent variables
# lr.intercept_
predict = lr.predict(test_x)
predict
lr.score(test_x, test_y) #R squared value
rmse = np.sqrt(sum((predict - test_y)**2)/len(test_y))

# from sklearn.metrics import mean_squared_error, r2_score
# np.sqrt(mean_squared_error(test_y, predict))
# r2_score(test_y, predict)

# Regularization of linear model by Ridge
ridge = Ridge(alpha=0.01, normalize=True)
ridge.fit(train_x, train_y)
pd.DataFrame(ridge.coef_, train_df.columns, columns=['Coefficient']) #find the cofficient of independent variables
ridge_predict = ridge.predict(test_x)
ridge_predict
ridge.score(test_x, test_y) #R squared value
np.sqrt(sum((ridge_predict - test_y)**2)/len(test_y))  #mean squared error


# Regularization of linear model by Lasso
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(train_x, train_y)
pd.DataFrame(lasso.coef_, train_df.columns, columns=['Coefficient']) #find the cofficient of independent variables
lasso_predict = lasso.predict(test_x)
lasso_predict
lasso.score(test_x, test_y) #R squared value
np.sqrt(mean_squared_error(test_y, predict))  #mean squared error


# Regularization of linear model by Elastic Net
elastic = ElasticNet(alpha=0.01, l1_ratio=1)
elastic.fit(train_x, train_y)
pd.DataFrame(elastic.coef_, train_df.columns, columns=['Coefficient']) #find the cofficient of independent variables
elastic_predict = elastic.predict(test_x)
elastic_predict
elastic.score(test_x, test_y) #R squared value
np.sqrt(mean_squared_error(test_y, predict))  #mean squared error


#Random Forest
rfr = RandomForestRegressor(n_estimators=500, min_samples_split=350)
rfr.fit(train_x, train_y)
rfr_predict = rfr.predict(test_x)
np.sqrt(mean_squared_error(test_y, rfr_predict))
r2_score(test_y, rfr_predict)

### Neural network
import keras
from keras.models import Model,Sequential
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Reshape, Dropout, Activation
from keras.optimizers import SGD, Adam

train_x, test_x, train_y, test_y  = train_x.values, test_x.values, train_y.values, test_y.values

lr = 0.001 ##### learning rate
wd = 0.01
batch_size = 128
num_epochs = 50
model = Sequential()

model.add(Dense(512,input_dim=34, activation='relu', kernel_initializer='normal'))
model.add(Dense(256,input_dim=34, activation='relu', kernel_initializer='normal'))
model.add(Dense(512,input_dim=34, activation='relu', kernel_initializer='normal'))
model.add(Dense(256,input_dim=34, activation='relu', kernel_initializer='normal'))
model.add(Dense(780,input_dim=34, activation='relu', kernel_initializer='normal'))
model.add(Dense(220,input_dim=34, activation='relu', kernel_initializer='normal'))
model.add(Dense(1, kernel_initializer='normal'))
#sgd = SGD(lr=lr, decay=1e-3, momentum=0.9, nesterov=True, clipvalue=5.0)
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='mean_squared_error', optimizer=adam,metrics=['accuracy'])
print (model.summary())
model.fit(train_x, train_y, nb_epoch=num_epochs, batch_size=batch_size, shuffle=True, 
          validation_data=(test_x, test_y))

keras_predicts = model.predict(test_x)
keras_predicts = pd.DataFrame(keras_predicts)
np.sqrt(metrics.mean_squared_error(test_y, keras_predicts))



#Cleaning Test data set

pd.Categorical(test_df.Outlet_Size).describe()
test_df.Outlet_Size = test_df.Outlet_Size.fillna('High', limit=300)
test_df.Outlet_Size = test_df.Outlet_Size.fillna('Medium', limit=800)
test_df.Outlet_Size = test_df.Outlet_Size.fillna('Small')


pd.Categorical(test_df.Item_Fat_Content).describe()
test_df.Item_Fat_Content = test_df.Item_Fat_Content.replace(('low fat','LF'), 'Low Fat')
test_df.Item_Fat_Content = test_df.Item_Fat_Content.replace('reg', 'Regular')
sum(np.isnan(test_df.Item_Weight))   ### counting the NA
test_df.Item_Weight = test_df.Item_Weight.fillna(test_df.Item_Weight.mean())

test_df['Item_Visibility'] = test_df.Item_Visibility.replace(0, test_df.Item_Visibility.mean())

plt.hist(train_df.Item_Visibility)


mapping_Item_Fat_Content = {'Regular': 1, 'Low Fat': 0}
test_df.Item_Fat_Content = test_df.Item_Fat_Content.map(mapping_Item_Fat_Content)

mapping_Outlet_Size = {'Small': 0, "Medium": 1,'High': 2}
test_df.Outlet_Size = test_df.Outlet_Size.map(mapping_Outlet_Size)

# test_set['Outlet_Establishment_Year'] = 2013 - test_set['Outlet_Establishment_Year']
mapping_Outlet_Type = {'Grocery Store': 0, 'Supermarket Type1': 0, 'Supermarket Type2': 1, 'Supermarket Type3': 2}
test_df.Outlet_Type = test_df.Outlet_Type.map(mapping_Outlet_Type)

# handing Outlet_Location_Type
mapping_Outlet_Location_Type = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
test_df.Outlet_Location_Type = test_df.Outlet_Location_Type.map(mapping_Outlet_Location_Type)

x = pd.get_dummies(test_df.Item_Type)
test_df = pd.concat([test_df.iloc[:,1:], x], axis=1)
test_df = test_df.drop(['Item_Type'], axis=1)
x = pd.get_dummies(test_df.Outlet_Identifier)
test_df = pd.concat([test_df, x], axis=1)
test_df = test_df.drop(['Outlet_Identifier'], axis=1)
test_df.dtypes


lr.fit(train_df, y)
coeff_lr = pd.DataFrame(lr.coef_, train_df.columns,columns=['Coefficient'])
coeff_lr
lr_predict = lr.predict(test_df)
lr_predict
model1 = pd.DataFrame(lr_predict, columns=['Linear predict'])


ridge.fit(train_set,y)
coeff_df = pd.DataFrame(ridge.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
predict = ridge.predict(test_set)
predict
model2 = pd.DataFrame(predict,columns=['Ridge predict'])


lasso.fit(train_set,y)
coeff_df = pd.DataFrame(lasso.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
predict = lasso.predict(test_set)
predict
model3 = pd.DataFrame(predict,columns=['Lasso predict'])


elastic.fit(train_set,y)
coeff_df = pd.DataFrame(elastic.coef_,train_set.columns,columns=['Coefficient'])
coeff_df
predict = elastic.predict(test_set)
predict
model4 = pd.DataFrame(predict,columns=['Elastic predict'])

lrf.fit(train_set,y)
predict = lrf.predict(test_set)
model5 = pd.DataFrame(predict,columns=['RandomForest predict'])

X = train_set.values
Y = y.values
test = test_set.values
model.fit(X,Y, 
    nb_epoch=num_epochs, 
    batch_size=batch_size, 
    shuffle=True)

predicts = model.predict(test)
model6 = pd.DataFrame(predicts,columns=['NNetwork predict'])

test_final = pd.concat((model2,model1,model3,model4),axis=1)
test_final.to_csv("file1.csv")
