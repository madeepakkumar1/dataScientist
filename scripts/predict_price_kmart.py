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
import warnings
warnings.filterwarnings("ignore")

file = r'C:\Users\kumadee\Desktop\assignment2-kmart_sales_forecast\%s'

#Read train and test data set
train_df = pd.read_csv(file % 'train_kmart.csv')
test_df = pd.read_csv(file % 'test_kmart.csv')

print(train_df.head())
print(train_df.describe())
# train_df.columns

# correlation
sns.heatmap(train_df.corr())

def imputer_outlet_size(data, limit1, limit2):
    """Given limit1 and limit2 values after observing describe() """
    
    print('Before imputing outlet size: ', pd.Categorical(data.Outlet_Size).describe())
    data.Outlet_Size = data.Outlet_Size.fillna('High', limit=limit1) # 565
    data.Outlet_Size = data.Outlet_Size.fillna('Medium', limit=limit2) # 923
    data.Outlet_Size = data.Outlet_Size.fillna('Small')
    print('After Imputing outlet size: ',pd.Categorical(data.Outlet_Size).describe())


def imputer_item_fat_content(data):
    print('Before imputing item fat content: ', 
          pd.Categorical(data.Item_Fat_Content).describe())
    data.Item_Fat_Content = data.Item_Fat_Content.replace(('LF', 'low fat'), 'Low Fat')
    data.Item_Fat_Content = data.Item_Fat_Content.replace('reg', 'Regular')
    print('After imputing item fat content: ',
          pd.Categorical(data.Item_Fat_Content).describe())
 
 
def imputer_item_weight(data):
    # print('{Totoal NA: }'.format(data.Item_Weight.isnull().value_counts()[1]))
    print('Total NA in item weight: ', sum(np.isnan(data.Item_Weight)))
    # Fill NA with mean value
    data.Item_Weight = data.Item_Weight.fillna(data.Item_Weight.mean())


def imputer_item_visibility(data):
    #Find counts of zeros
    print('Total zeores count in item visibility: ', 
          data[data.Item_Visibility == 0].Item_Visibility.value_counts())
    # Replacing zeros with mean value 
    data.Item_Visibility = data.Item_Visibility.replace(0, train_df.Item_Visibility.mean())
    plt.hist(data.Item_Visibility)

def labelencoding_item_fat_content(data):
    print('label encoding for item fat content...')
    mapping_Item_Fat_Content = {'Regular': 1, 'Low Fat': 0}
    data.Item_Fat_Content = data.Item_Fat_Content.map(mapping_Item_Fat_Content)

def labelencoding_outlet_size(data):
    print('label encoding for outlet size...')
    mapping_Outlet_Size = {'Small': 0, "Medium": 1,'High': 2}
    data.Outlet_Size = data.Outlet_Size.map(mapping_Outlet_Size)

def labelencoding_Outlet_Type(data):
    print('description for outlet type: ',
          pd.Categorical(data.Outlet_Type).describe())
    print('label encoding for outlet type...')
    mapping_Outlet_Type = {'Grocery Store': 0, 'Supermarket Type1': 0, 
                           'Supermarket Type2': 1, 'Supermarket Type3': 2}
    data.Outlet_Type = data.Outlet_Type.map(mapping_Outlet_Type)

def labelencoding_outlet_location_type(data):
    # print(data.Outlet_Location_Type.unique())
    print('description of outlet location type: ',
          pd.Categorical(data.Outlet_Location_Type).describe())
    print('label encoding for outlot location type...')
    mapping_Outlet_Location_Type = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
    data.Outlet_Location_Type = data.Outlet_Location_Type.map(mapping_Outlet_Location_Type)

# Handling missing data
imputer_outlet_size(train_df, 565, 923)
imputer_item_fat_content(train_df)
imputer_item_weight(train_df)
imputer_item_visibility(train_df)

# Handling categorical data
labelencoding_item_fat_content(train_df)
labelencoding_outlet_size(train_df)
labelencoding_Outlet_Type(train_df)
labelencoding_outlet_location_type(train_df)

# features and label
y = train_df.Item_Outlet_Sales
x = pd.get_dummies(train_df.Item_Type)

train_df = pd.concat([train_df.iloc[:,1:], x], axis=1)
train_df = train_df.drop(['Item_Type'], axis=1)
x = pd.get_dummies(train_df.Outlet_Identifier)
train_df = pd.concat([train_df, x], axis=1)
train_df = train_df.drop(['Outlet_Identifier', 'Item_Outlet_Sales'], axis=1)

# Dividing data set into two parts train set and test set
train_x, test_x, train_y, test_y = train_test_split(train_df, y)

# Building models
lr = LinearRegression()
ridge = Ridge(alpha=0.01, normalize=True)
lasso = Lasso(alpha=0.1, normalize=True)
elasticnet = ElasticNet(alpha=0.01, l1_ratio=1)
rfr = RandomForestRegressor(n_estimators=500, min_samples_split=350)
models = (lr, ridge, lasso, elasticnet, rfr)

for model in models:
    print(f'Calling model object: {model}')
    model.fit(train_x, train_y)
    # print(pd.DataFrame(lr.coef_, train_df.columns, columns=['Coefficient']))
    predict = model.predict(test_x)
    print(np.sqrt(mean_squared_error(test_y, predict)))
    print(r2_score(test_y, predict))
    



# graph for check the prediction vs test target
rfr.fit(train_x, train_y)
predict = rfr.predict(test_x)
plt.hist([test_y, predict], color=['orange', 'green'])
plt.legend(['actual target', 'predicted target'])
plt.savefig('RandomForest Prediction image')
plt.show()


# Neural network model
import keras
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Reshape, Dropout, Activation
from keras.optimizers import SGD, Adam

train_x, test_x, train_y, test_y  = train_x.values, test_x.values, train_y.values, test_y.values

lr = 0.001 # learning rate
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
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(train_x, train_y, 
          epochs=num_epochs, 
          batch_size=batch_size, 
          shuffle=True,
          validation_data=(test_x, test_y))

keras_predicts = model.predict(test_x)
keras_predicts = pd.DataFrame(keras_predicts)
print(np.sqrt(metrics.mean_squared_error(test_y, keras_predicts)))



######## Handling Test data set #############

# handling missing data
imputer_outlet_size(test_df, 300, 800)
imputer_item_fat_content(test_df)
imputer_item_weight(test_df)
imputer_item_visibility(test_df)

# handling categorical data (label encoding)
labelencoding_item_fat_content(test_df)
labelencoding_outlet_size(test_df)
labelencoding_Outlet_Type(test_df)
labelencoding_outlet_location_type(test_df)

# Onehot encoding for Item_Type and Outlet_Identifier
x = pd.get_dummies(test_df.Item_Type)
test_df = pd.concat([test_df.iloc[:,1:], x], axis=1)
test_df = test_df.drop(['Item_Type'], axis=1)
x = pd.get_dummies(test_df.Outlet_Identifier)
test_df = pd.concat([test_df, x], axis=1)
test_df = test_df.drop(['Outlet_Identifier'], axis=1)

lr = LinearRegression()
ridge = Ridge(alpha=0.01, normalize=True)
lasso = Lasso(alpha=0.1, normalize=True)
elasticnet = ElasticNet(alpha=0.01, l1_ratio=1)
rfr = RandomForestRegressor(n_estimators=500, min_samples_split=350)


# linear regression model
lr.fit(train_df, y)
# coeff_lr = pd.DataFrame(lr.coef_, train_df.columns, columns=['Coefficient'])
# print(coeff_lr)
lr_predict = lr.predict(test_df)
model1 = pd.DataFrame(lr_predict, columns=['Linear predict'])
lr_coef1 = pd.Series(lr.coef_, train_df.columns).sort_values()
lr_coef1.plot(kind='bar', title='Model Coefficients')

#ridge model
ridge.fit(train_df, y)
# coeff_ridge = pd.DataFrame(ridge.coef_, train_df.columns, columns=['Coefficient'])
ridge_predict = ridge.predict(test_df)
model2 = pd.DataFrame(ridge_predict, columns=['Ridge predict'])
ridge_coef = pd.Series(ridge.coef_, train_df.columns).sort_values()
ridge_coef.plot(kind='bar', title='Model Coefficients')
#lasso model
lasso.fit(train_df, y)
coeff_lasso = pd.DataFrame(lasso.coef_,train_df.columns, columns=['Coefficient'])
lasso_predict = lasso.predict(test_df)
model3 = pd.DataFrame(lasso_predict, columns=['Lasso predict'])

#elastic net model
elasticnet.fit(train_df, y)
coeff_elastic = pd.DataFrame(elasticnet.coef_, train_df.columns,columns=['Coefficient'])
elastic_predict = elasticnet.predict(test_df)
model4 = pd.DataFrame(elastic_predict, columns=['Elastic predict'])

#random forest model
rfr.fit(train_df, y)
rfr_predict = rfr.predict(test_df)
model5 = pd.DataFrame(rfr_predict, columns=['RandomForest predict'])
rfr_coef = pd.Series(rfr.feature_importances_, train_df.columns).sort_values()
rfr_coef.plot(kind='bar', title='Model coefficients')
# Neuro network model
X = train_df.values
Y = y.values
test = test_df.values
model.fit(X, Y, nb_epoch=num_epochs, batch_size=batch_size, shuffle=True)
predicts = model.predict(test)
model6 = pd.DataFrame(predicts, columns=['NNetwork predict'])


# Campare the all model's predictions
final_test = pd.concat((model1, model2, model3, model4, model5, model6), axis=1)
final_test.to_csv('final_result.csv', index=False)
print(final_test.head())
print(final_test.mean().head())


