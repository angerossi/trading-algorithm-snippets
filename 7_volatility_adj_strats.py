import numpy as np
import pandas as pd
import matplotlib as mpl
import statistics as stats
from pandas_datareader import data
from matplotlib import pyplot as plt
from collections import deque
mpl.style.use('dark_background')

start_date = '2014-01-01'
end_date = '2018-01-01'
ticker_name = 'GOOG'
SRC_DATA_FILENAME = 'goog_data.pkl'

# load data
try:
    data = pd.read_pickle(SRC_DATA_FILENAME)
except FileNotFoundError:
    data = data.DataReader(ticker_name, 'yahoo', start_date, end_date)
    data.to_pickle(SRC_DATA_FILENAME)
    
'''volatility-sensitive mean-reversion apo signal'''
# buys and sells a set ammount per trade. Can make multiple buy/sell orders in a row

# sell: apo is above threshold and 
#       difference between last trade price and current price above threshold
#       or
#       we are in a long position and
#       the apo is at or above 0 or current pos is profitable enough to lock profit
# buy:  apo is below threshold and
#       difference between last trade price and current trade price above threshold
#       we are in a short position and
#       the apo is at or below 0 or current pos is profitable enough to lock profit
# volatility influence: v_factor = std/mean_std such that 1 is normal, <1 is less than normal, >1 more than normal
#                       k_fast/v_factor for inc response for high v
#                       APO_VALUE_FOR_BUY_ENTRY*v_factor for less aggresive entry for high v
#                       APO_VALUE_FOR_SELL_ENTRY*v_factor for less aggresive entry for high v
#                       MIN_PROFIT_CLOSE/v_factor to take profits more easily for high v

# parameters
STD_BASIS = 15
STD_PERIOD = 20
NUM_PERIODS_FAST = 10 
NUM_PERIODS_SLOW = 40
APO_VALUE_FOR_BUY_ENTRY = -10
APO_VALUE_FOR_SELL_ENTRY = 10
MIN_PRICE_MOVE_FROM_LAST_TRADE = 10 # to prevent overtrading (min price change since last trade before considering trading again)
MIN_PROFIT_TO_CLOSE = 10 # min unrealised profit to lock profits
NUM_SHARES_PER_TRADE = 10

k_fast = 2/(NUM_PERIODS_FAST + 1)
k_slow = 2/(NUM_PERIODS_SLOW + 1)
ema_fast = 0
deque_costs = deque(maxlen=STD_PERIOD)
list_ema_fast = []
ema_slow = 0
list_ema_slow = []
list_apo = []
list_orders = [] # 1 = buy, -1 = sell
list_positions = [] # + = long, - = short, 0 = no position
list_pnl = [] # total pnls (pnls already locked in and open-market pnl)
last_buy_price = 0
last_sell_price = 0
position = 0 # current position (number of shares currently owned)
buy_sum_value = 0 # value of buys since flat
sell_sum_value = 0 # value of sells since flat
buy_sum_qty = 0 # quantity of buys since flat
sell_sum_qty = 0 # quantity of sells since flat
open_pnl = 0
pnl = 0 # realised pnl so far

for close in data['Close']:
    # v_factor
    deque_costs.append(close)
    if len(deque_costs) == 1:
        v_factor = 1
    else:
        std = stats.stdev(deque_costs)
        v_factor = std/STD_BASIS

    # calc apo
    if ema_fast == 0 :
        ema_fast = close
        ema_slow = close
    else:
        ema_fast = (close - ema_fast)*k_fast/v_factor + ema_fast
        ema_slow = (close - ema_slow)*k_slow + ema_slow
    list_ema_fast.append(ema_fast)
    list_ema_slow.append(ema_slow)
    apo = ema_fast - ema_slow
    list_apo.append(apo)
   
    #v_factor = 1
    # orders
    if (apo >= APO_VALUE_FOR_SELL_ENTRY*v_factor and abs(close - last_sell_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE) or ((position > 0) and (apo >= 0 or open_pnl > MIN_PROFIT_TO_CLOSE/v_factor)):
        order = -1
        last_sell_price = close
        position -= NUM_SHARES_PER_TRADE
        sell_sum_value += close*NUM_SHARES_PER_TRADE 
        #print(f'sell: {NUM_SHARES_PER_TRADE} @ {close}, position: {position}')
    
    elif (apo < APO_VALUE_FOR_BUY_ENTRY*v_factor and abs(close - last_buy_price) > MIN_PRICE_MOVE_FROM_LAST_TRADE) or ((position < 0) and (apo <= 0 or open_pnl > MIN_PROFIT_TO_CLOSE/v_factor)):
        order = 1
        last_buy_price = close
        position += NUM_SHARES_PER_TRADE
        buy_sum_value += close*NUM_SHARES_PER_TRADE
        #print(f'buy: {NUM_SHARES_PER_TRADE} @ {close}, position: {position}')
    
    else:
        order = 0
    list_orders.append(order)
    list_positions.append(position)
    
    # pnl
    if position == 0:
        pnl += sell_sum_value - buy_sum_value
        sell_sum_value = 0 
        buy_sum_value = 0
        open_pnl = 0
    elif position == 1:
        open_pnl = NUM_SHARES_PER_TRADE*(close - last_buy_price)
    elif position == -1:
        open_pnl == NUM_SHARES_PER_TRADE*(last_sell_price - close)
    list_pnl.append(pnl)

df_mean_rev = pd.DataFrame(data=data['Close'], index=data.index)
df_mean_rev['ema_fast'] = list_ema_fast      
df_mean_rev['ema_slow'] = list_ema_slow
df_mean_rev['apo'] = list_apo
df_mean_rev['orders'] = list_orders
df_mean_rev['position'] = list_positions
df_mean_rev['pnl'] = list_pnl

fig, axes = plt.subplots(3, 1, sharex=True)
df_mean_rev['Close'].plot(ax=axes[0], color='w', lw=3)
df_mean_rev['ema_fast'].plot(ax=axes[0], color='y', lw=3)
df_mean_rev['ema_slow'].plot(ax=axes[0], color='m', lw=3)
axes[0].plot(df_mean_rev.loc[df_mean_rev['orders'] == 1].index, 
        df_mean_rev['Close'][df_mean_rev['orders'] == 1], 
        color='r', lw=0, marker='^', markersize=20, label='buy')
axes[0].plot(df_mean_rev.loc[df_mean_rev['orders'] == -1].index, 
        df_mean_rev['Close'][df_mean_rev['orders'] == -1], 
        color='g', lw=0, marker='v', markersize=20, label='sell')
axes[0].set_ylabel('Close')

df_mean_rev['apo'].plot(color='pink', lw=3, ax=axes[1])
axes[1].plot(df_mean_rev.loc[df_mean_rev['orders'] == 1].index,
        df_mean_rev['apo'][df_mean_rev['orders'] == 1],
        color='r', lw=0, marker='^', markersize=20)
axes[1].plot(df_mean_rev.loc[df_mean_rev['orders'] == -1].index,
        df_mean_rev['apo'][df_mean_rev['orders'] == -1],
        color='g', lw=0, marker='v', markersize=20)
axes[1].axhline(y=0, lw=1, color='w')
for i in range(APO_VALUE_FOR_BUY_ENTRY, APO_VALUE_FOR_BUY_ENTRY*5, APO_VALUE_FOR_BUY_ENTRY):
    axes[1].axhline(y=i, lw=1, color='r')
for i in range(APO_VALUE_FOR_SELL_ENTRY, APO_VALUE_FOR_SELL_ENTRY*5, APO_VALUE_FOR_SELL_ENTRY):
    axes[1].axhline(y=i, lw=1, color='g')
axes[1].set_ylabel('apo')    

df_mean_rev['pnl'].plot(ax=axes[2], color='w', ls='dotted', lw=3)
axes[2].set_ylabel('pnl')
                    
fig.legend()
