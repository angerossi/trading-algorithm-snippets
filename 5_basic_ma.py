import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stats
import pandas as pd
import numpy as np
from pandas_datareader import data
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

start_date = '2001-01-01'
end_date = '2018-01-01'
SRC_DATA_FILENAME = 'goog_data.pkl'

# load data
try:
    goog_data = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
    goog_data.to_pickle(SRC_DATA_FILENAME)
    
# labels for regression prediction (numerical prediction) -- change in price/day
def pop_regression_y(df):
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df = df.dropna()
    x = df[['Open-Close', 'High-Low']]
    y = df['Close'].shift(-1) - df['Close']
    x = x[:-1]
    y = y[:-1]
    return (x, y)

# labels for classification prediction (categorical prediction) -- 1: up, -1: down
def pop_classification_y(df):
    df['Open-Close'] = df['Open'] - df['Close']
    df['High-Low'] = df['High'] - df['Low']
    df = df.dropna()
    x = df[['Open-Close', 'High-Low']]
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
    x = x[:-1]
    y = y[:-1]
    return (x, y)  

# linear model
x, y = pop_regression_y(goog_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.8)
lrm = LinearRegression()
lrm.fit(x_train, y_train)
y_hat = lrm.predict(x_test)
print('linear:')
print('mse: %.2f'
      % mean_squared_error(y_test, y_hat))
print('r2: %.2f'
    % r2_score(y_test, y_hat))
print('-----------------------------')
    # lasso and ridge regression
x, y = pop_regression_y(goog_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.8)
lasso_rm = Lasso(alpha=0.1)
lasso_rm.fit(x_train, y_train)
y_hat = lasso_rm.predict(x_test)
print('lasso')
print('mse: %.2f'
      % mean_squared_error(y_test, y_hat))
print('r2: %.2f'
    % r2_score(y_test, y_hat))
print('-----------------------------')
    # ridge model
x, y = pop_regression_y(goog_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.8)
ridge_rm = Ridge(alpha=1e5)
ridge_rm.fit(x_train, y_train)
y_hat = ridge_rm.predict(x_test)
print('ridge ')
print('mse: %.2f'
      % mean_squared_error(y_test, y_hat))
print('r2: %.2f'
    % r2_score(y_test, y_hat))
print('-----------------------------')

# KNN
x, y = pop_classification_y(goog_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.8)
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(x_train, y_train)
acc_train = accuracy_score(y_train, knn.predict(x_train))
acc_test = accuracy_score(y_test, knn.predict(x_test))
print('knn')
print(f'acc score train: {acc_train}')
print(f'acc score test: {acc_test}')

# SVM
x, y = pop_classification_y(goog_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.8)
svc = SVC()
svc.fit(x_train, y_train)

# logistic regression