import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statistics as stats
from pandas_datareader import data
from collections import deque
from statsmodels.tsa.stattools import adfuller
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

# get mean cost per month
goog_data.reset_index(inplace=True)
goog_data['pct_change'] = goog_data['Adj Close'].pct_change()
goog_monthly_return = goog_data[['Date', 'pct_change']].groupby(
    [goog_data['Date'].dt.year,
    goog_data['Date'].dt.month]).mean()
goog_monthly_return.reset_index(level=0, inplace=True)
goog_monthly_return.drop(columns='Date', axis=1, inplace=True)
goog_monthly_return.rename(columns={'pct_change': 'monthly_return'}, inplace=True)
goog_monthly_return.index.names = ['month']
#sns.boxplot(data=goog_monthly_return, x=goog_monthly_return.index, y='monthly_return')

# rolling statistics
def plot_stats(s, windowsize=12):
    fig = plt.figure()
    plt.plot(s, color='r', label='original', lw=0.5)
    plt.plot(s.rolling(windowsize).mean(), color='b', label='rolling mean')
    plt.plot(s.rolling(windowsize).std(), color='k', label='rolling std')
    plt.legend()
goog_monthly_return_sequential = goog_monthly_return.reset_index()
goog_monthly_return_sequential.drop('month', axis=1, inplace=True)
#plot_stats(goog_monthly_return_sequential)
#plot_stats(goog_data['Adj Close'], windowsize=365)

# determine if there is a unit root --> if the series is non-stationary --> if the series has a non-constant mean&std
def test_stationarity(s):
    test = adfuller(s)
    output = pd.Series(test[0:4], index=['test stat', 'p-value', '# lags used', '# observations used'])
    print(output)
test_stationarity(goog_data['Adj Close'])
