'''buy when price decreases, sell when price increases'''

from pandas_datareader import data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

start_date = '2014-01-01'
end_date = '2018-01-01'
goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)

# signal
goog_data_signal = pd.DataFrame(index=goog_data.index)
goog_data_signal['price'] = goog_data['Adj Close']
goog_data_signal['daily_difference'] = goog_data_signal['price'].diff()
goog_data_signal['signal'] = 0.0
goog_data_signal['signal'] = np.where(goog_data_signal['daily_difference'] > 0, 0.0, 1.0) # 0:sell, 1:buy
goog_data_signal['positions'] = goog_data_signal['signal'].diff()

# backtesting
initial_capital = float(1000.0)
positions = pd.DataFrame(index=goog_data_signal.index).fillna(0.0)
portfolio = pd.DataFrame(index=goog_data_signal.index).fillna(0.0)
positions['GOOG'] = goog_data_signal['signal']
portfolio['positions'] = (positions.multiply(goog_data_signal['price'], axis=0))
portfolio['cash'] = initial_capital - (positions.diff().multiply(goog_data_signal['price'], axis=0)).cumsum()
portfolio['total'] = portfolio['positions'] + portfolio['cash']

# output
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Google price in $')
portfolio['total'].plot(ax=ax1, label='total')
portfolio['cash'].plot(ax=ax1, label='cash')
portfolio['positions'].plot(ax=ax1, label='positions')
goog_data_signal['price'].plot(ax=ax1, color='r', lw=2.)
# ax1.plot(goog_data_signal.loc[goog_data_signal['positions'] == 1.0].index,
#           goog_data_signal['price'][goog_data_signal['positions'] == 1.0], '^', markersize=5, color='m', label='buy')
# ax1.plot(goog_data_signal.loc[goog_data_signal['positions'] == -1.0].index,
#           goog_data_signal['price'][goog_data_signal['positions'] == -1.0], 'v', markersize=5, color='k', label='sell')
ax1.legend()
plt.show()
print(goog_data_signal.head(20))