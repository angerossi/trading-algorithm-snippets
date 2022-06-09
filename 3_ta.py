import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stats
from pandas_datareader import data
from collections import deque
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

start_date = '2014-01-01'
end_date = '2018-01-01'
SRC_DATA_FILENAME = 'goog_data.pkl'

# load data
try:
    goog_data = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
    goog_data.to_pickle(SRC_DATA_FILENAME)
close_prices = goog_data['Adj Close'].to_list()

# sma
time_period = 20
history = deque(maxlen=time_period)
sma_values = []

for i in range(len(close_prices)):
    history.append(close_prices[i])
    sma_values.append(stats.mean(history))
goog_data['sma'] = sma_values

# ema
'''K = proportion of new ema value that is the new closevalue. 
(1 - K) = proportion of new ema that is old ema value. This formula used as default'''
K = 2/(time_period + 1) 
ema = 0
ema_values = []

for i in range(len(close_prices)):
    if (ema == 0): # first ema
        ema = close_prices[i]
    else:
        ema = ema + (close_prices[i] - ema)*K
    ema_values.append(ema)
goog_data['ema'] = ema_values

# apo
'''diff between a fast ema and slow ema. Big difference = either that prices are starting to trend/break out or that prices are far
from equilibrium (overbought or oversold)'''
time_period_fast = 10
time_period_slow = 40
K_fast = 2/(time_period_fast + 1)
K_slow = 2/(time_period_slow + 1)
ema_fast = 0
ema_slow = 0
apo_values = []

for i in range(len(close_prices)):
    if (ema_fast == 0) and (ema_slow == 0):
        ema_fast = close_prices[i]
        ema_slow = close_prices[i]
    else:
        ema_fast = ema_fast + (close_prices[i] - ema_fast)*K_fast
        ema_slow = ema_slow + (close_prices[i] - ema_slow)*K_slow
    apo_values.append(ema_fast - ema_slow)
goog_data['apo'] = apo_values
    
# macd
'''same idea as apo, however a singal macd is given by applying a smoothing ema
   to the apo. Then the apo is compared to the signal.'''

time_period_fast = 10
time_period_slow = 40
time_period_macd = 20
K_fast = 2/(time_period_fast + 1)
K_slow = 2/(time_period_slow + 1)
K_macd = 2/(time_period_macd + 1)
ema_fast = 0
ema_slow = 0
ema_macd = 0
macd_values = []
signal_values= []
macd_histogram_values = []

for i in range(len(close_prices)):
    if (ema_fast == 0) and (ema_slow == 0):
        ema_fast = close_prices[i]
        ema_slow = close_prices[i]
    else:
        ema_fast = ema_fast + (close_prices[i] - ema_fast)*K_fast
        ema_slow = ema_slow + (close_prices[i] - ema_slow)*K_slow
    macd = ema_fast - ema_slow
    
    if (ema_macd == 0):
        ema_macd = macd
    else:
        ema_macd = ema_macd + (macd - ema_macd)*K_macd
    
    macd_values.append(macd)
    signal_values.append(ema_macd)
    macd_histogram_values.append(macd - ema_macd)
goog_data['macd'] = macd_values
goog_data['macd_signal'] = signal_values
goog_data['macd_histo'] = macd_histogram_values

# bb
time_period = 20
std_factor = 2
history = deque(maxlen=time_period)
sma_values = []
upper_band = []
lower_band = []

for i in range(len(close_prices)):
    history.append(close_prices[i])
    sma = stats.mean(history)
    sma_values.append(stats.mean(history))
    
    if (len(upper_band) <= 1):
        std = 0
        upper_band.append(sma)
        lower_band.append(sma)
    else:
        std = stats.stdev(history)
        upper_band.append(sma + std_factor*std)
        lower_band.append(sma - std_factor*std)  
goog_data['sma'] = sma_values
goog_data['upper_band'] = upper_band
goog_data['lower_band'] = lower_band

# rsi
'''based on price changes over periods to capture strength/magnitude of price
 moves. >50 % indicate an uptrend, <50 % indicate downtrend. 50 % is the point where
 there are more price increases than price decreases in the history period'''

rsi_time_period = 20
gains_history = deque(maxlen=rsi_time_period)
losses_history = deque(maxlen=rsi_time_period)
rsi_values = []
last_price = 0

for i in range(len(close_prices)):
    if (last_price == 0):
        last_price = close_prices[i]
    gains_history.append(max(0, close_prices[i] - last_price))
    losses_history.append(max(0, last_price - close_prices[i]))
    last_price = close_prices[i]
    
    avg_gains = stats.mean(gains_history)
    avg_losses =stats.mean(losses_history) 
    if avg_losses == 0 :
        rs = 0
    else:
        rs = avg_gains/avg_losses
    rsi_values.append(100 - 100/(1 + rs))
    
goog_data['rsi'] = rsi_values

# std
'''standard deviation is a basic measure of price volatility and can improve other indicators'''
time_period_std = 20
history = deque(maxlen=time_period_std)
std_values = []

for i in range(len(close_prices)):
    history.append(close_prices[i])
    
    if (len(std_values) <= 1):
        std = 0
    else:
        std = stats.stdev(history)
    std_values.append(std)
goog_data['std'] = std_values

# mom
'''momentum is the difference in price between the current and that n periods ago'''
time_period_mom = 20
history = deque(maxlen=20)
mom_values = []

for i in range(len(close_prices)):
    history.append(close_prices[i])
    mom_values.append(close_prices[i] - history[0])
goog_data['mom'] = mom_values   

# plot
fig = plt.figure(tight_layout=True, dpi=50)
ax1 = fig.add_subplot(611, ylabel='Google price in $')
ax2 = fig.add_subplot(612, ylabel='macd')
ax3 = fig.add_subplot(613, ylabel='macd histo')
ax4 = fig.add_subplot(614, ylabel='rsi')
ax5 = fig.add_subplot(615, ylabel='std')
ax6 = fig.add_subplot(616, ylabel='mom')
goog_data['Adj Close'].plot(ax=ax1, color='g', lw=2)
#goog_data['sma'].plot(ax=ax1, color='r', lw=2)
#goog_data['ema'].plot(ax=ax1, color='b', lw=2)
#goog_data['apo'].plot(ax=ax2, color='k', lw=2)
goog_data['macd'].plot(ax=ax2, color='k', lw=2)
goog_data['macd_signal'].plot(ax=ax2, color='g', lw=2)
ax3.bar(goog_data.index, goog_data['macd_histo'], color='r', label='macd_histo')
goog_data['sma'].plot(ax=ax1, color='b')
goog_data['upper_band'].plot(ax=ax1, color='g')
goog_data['lower_band'].plot(ax=ax1, color='r')
goog_data['rsi'].plot(ax=ax4, color='b')
goog_data['std'].plot(ax=ax5, color='b')
goog_data['mom'].plot(ax=ax6, color='g')
ax4.hlines(50, goog_data.index[0], goog_data.index[-1], color='k')
ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()
ax5.legend()