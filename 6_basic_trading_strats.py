import numpy as np
import pandas as pd
from pandas_datareader import data
from matplotlib import pyplot as plt

start_date = '2014-01-01'
end_date = '2018-01-01'
SRC_DATA_FILENAME = 'goog_data.pkl'

# load data
try:
    goog_data = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
    goog_data.to_pickle(SRC_DATA_FILENAME)
    

# doubla ma
def double_ma(data, short_window, long_window):
    signals = pd.DataFrame(index=data.index)
    signals['short_ma'] = data['Close'].rolling(window=short_window).mean()
    signals['long_ma'] = data['Close'].rolling(window=long_window).mean()
    signals['signal'] = 0.0
    signals['signal'][short_window:] = np.where(signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1.0, 0.0)
    signals['orders'] = signals['signal'].diff()
    return signals

signals = double_ma(goog_data, 20, 100)

fig = plt.figure()
ax = fig.add_subplot(ylabel='GOOG price [$]', title='double ma')
goog_data['Close'].plot(ax=ax, color='g', lw=.5, label='Close')
signals['short_ma'].plot(ax=ax, color='r', lw=2, label='Short MA')
signals['long_ma'].plot(ax=ax, color='b', lw=2, label='Long MA')
ax.plot(signals.loc[signals['orders'] == 1.0].index,
        goog_data['Close'][signals['orders'] == 1.0],
        marker='^', markersize=7, color='k', lw=0, label='Buy')
ax.plot(signals.loc[signals['orders'] == -1.0].index,
        goog_data['Close'][signals['orders'] == -1.0],
        marker='v', markersize=7, color='k', lw=0, label='Sell')
ax.legend()

# naive trading strategy
def naive_momentum_trading(data, period_len):
    signals = pd.DataFrame(index=data.index)
    signals['orders'] = 0
    count = 0
    prev_price = 0
    init = True
    for i, price in enumerate(data['Close'].values):
        if init:
            prev_price = price
            init = False
        elif price > prev_price:
            if count < 0:
                count = 0
            count += 1
        elif price < prev_price:
            if count > 0:
                count = 0
            count -= 1
        if count == period_len:
            signals['orders'][i] = 1.0
        elif count == -period_len:
            signals['orders'][i] = -1.0
            
        prev_price = price
    return signals
signals = naive_momentum_trading(goog_data, 5)

fig = plt.figure()
ax = fig.add_subplot(ylabel='GOOG price [$]', title='naive momentum')
goog_data['Close'].plot(ax=ax, color='g', lw=.5, label='Close')
ax.plot(signals.loc[signals['orders'] == 1.0].index,
        goog_data['Close'][signals['orders'] == 1.0],
        marker='^', markersize=7, color='k', lw=0, label='Buy')
ax.plot(signals.loc[signals['orders'] == -1.0].index,
        goog_data['Close'][signals['orders'] == -1.0],
        marker='v', markersize=7, color='k', lw=0, label='Sell')
ax.legend()

# turtle strategy
def turtle_strat(data, window_entry, window_exit):
    signals = pd.DataFrame(index=data.index)
    signals['orders'] = 0
    signals['high'] = data['Close'].shift(1).rolling(window=window_entry).max()
    signals['low'] = data['Close'].shift(1).rolling(window=window_entry).min()
    signals['mean'] = data['Close'].shift(1).rolling(window=window_exit).mean()
    
    # entry: price > high for last [window] days
    #        price < low for last [window] days
    signals['long_entry'] = data['Close'] > signals['high']
    signals['short_entry'] = data['Close'] < signals['low']
    
    # exit: price crosses mean for last [window] days
    signals['long_exit'] = data['Close'] < signals['mean']
    signals['short_exit'] = data['Close'] > signals['mean']
    
    init = True
    pos = 0
    for i, price in enumerate(data["Close"]):
        if (signals['long_entry'][i]) and (pos == 0):
            signals['orders'][i] = 1
            pos = 1
        elif (signals['short_entry'][i]) and (pos == 0):
            signals['orders'][i] = -1
            pos = -1
        elif (signals['long_exit'][i]) and (pos == 1):
            signals['orders'][i] = -1
            pos = 0
        elif (signals['short_exit'][i]) and (pos == -1):
            signals['orders'][i] = 1
            pos = 0
        else:
            signals['orders'][i] = 0
    return signals
signals = turtle_strat(goog_data, 70, 20)

fig = plt.figure()
ax = fig.add_subplot(ylabel='GOOG price [$]', title='turtle')
goog_data['Close'].plot(ax=ax, color='g', lw=.5, label='close')
signals['low'].plot(ax=ax, color='r', label='low')
signals['high'].plot(ax=ax, color='g', label='high')
signals['mean'].plot(ax=ax, color='purple', label='mean')
ax.plot(signals.loc[signals['orders'] == 1.0].index,
        goog_data['Close'][signals['orders'] == 1.0],
        marker='^', markersize=7, color='k', lw=0, label='Buy')
ax.plot(signals.loc[signals['orders'] == -1.0].index,
        goog_data['Close'][signals['orders'] == -1.0],
        marker='v', markersize=7, color='k', lw=0, label='Sell')
ax.legend()