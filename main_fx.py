#%%
!pip install ta oandapyV20 plotly ta-lib sklearn
# conda install -c conda-forge ta-lib

# %% Imports
import plotly.express as px
import numpy as np
import pandas as pd
import requests
import datetime as dt
import talib
import ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
import json
from datetime import datetime, timedelta, timezone
from tqdm import trange
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# %%
# Read in API Key
with open('secrets.json', 'r') as f:
  SECRETS = json.load(f)
API_KEY = SECRETS['OANDA_API']
ACCOUNT_ID = SECRETS['OANDA_ACCOUNT_ID']

#%% Retrieve dataset
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments


accountID = ACCOUNT_ID
api = API(access_token=API_KEY)

num_years = 3
date_today  = datetime.now() - timedelta(days = 1)
df = pd.DataFrame()

# Loop through past  months
for i in trange(num_years * 12 * 2):
    try:
        date_end = (date_today - timedelta(days = 15 * i)).strftime("%Y-%m-%d")
        params = {
            "count":5000,
            "granularity": "M5",
            "to": date_end
        }
        r = instruments.InstrumentsCandles(instrument = "GBP_JPY", params = params) # EUR_USD
        rv = api.request(r)
        sub_df = pd.DataFrame(rv['candles'])
        sub_df['open'] = sub_df['mid'].apply(lambda x: x['o'])
        sub_df['high'] = sub_df['mid'].apply(lambda x: x['h'])
        sub_df['low'] = sub_df['mid'].apply(lambda x: x['l'])
        sub_df['close'] = sub_df['mid'].apply(lambda x: x['c'])
        sub_df.drop(columns = ['mid'], inplace=True)
        if i == 0:
            df = sub_df
        else:
            df = pd.concat([df,sub_df], ignore_index= True)
    except:
        print(date_end)
        
# %% Data Manipulation
df = df.drop_duplicates()
print(df.shape)
df.head()

# %% Save df
df.to_csv('GBP_JPY.csv')

# %%
df = pd.read_csv('GBP_JPY.csv')

# %% Data manipulation
df['datetime'] = df['time'].apply(lambda x: datetime.fromisoformat(x[:-7]).astimezone(timezone.utc))
df.head()

# %% Plot datapoints on plotly

def plot_candlestick(df_in, start_date = None, end_date = None):
  # Get subset of data
  if (start_date != None) | (end_date != None):
    sub_df = df_in[
      (df_in['datetime'].dt.date >= pd.to_datetime(start_date)) & \
      (df_in['datetime'].dt.date <= pd.to_datetime(end_date))
    ]
  else:
    sub_df = df_in

  print(sub_df.shape)
  
  # Plot
  fig = go.Figure(
    data = [
      go.Candlestick(
        x = sub_df['datetime'],
        open = sub_df['open'],
        close = sub_df['close'],
        high = sub_df['high'],
        low = sub_df['low']
      )
    ]
  )
  fig.update_layout(height=1000)
  fig.write_html('candlestick.html')
  fig.show()

plot_candlestick(
  df, 
  start_date = '18/3/2023', # 'today', "MM/DD/YYYY"
  end_date = 'today'
)

# %% Plot datapoints on matplotlib
plt.figure(figsize=(20,20))
plt.scatter(
    x = df['datetime'],
    y = df['close'],
    alpha = 0.1
)
plt.grid()
plt.show()

# %% Plot with Plotly V2
fig = go.Figure(
  data = [
    go.Scatter(
      x = df['datetime'],
      y = df['close'],
      opacity = 0.3
    )
  ]
)
fig.show()

# %% Adding indicators

# RSI
df['RSI'] = ta.momentum.RSIIndicator(df['close'], window = 14).rsi()

# Bollinger Band
# bol = ta.volatility.BollingerBands(df['close'], window = 14)
# df['bol_upper'] = bol.bollinger_hband()
# df['bol_middle'] = bol.bollinger_mavg() 
# df['bol_lower'] = bol.bollinger_lband()

# MACD
# MACD = ema_26 - ema_12
# Signal = MACD - ema_9
# Hist = MACD - Signal
macd, signal, hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['MACD_hist'] = hist

# SMA
# df['SMA_l_200'] = ta.trend.SMAIndicator(df['low'], window = 200).sma_indicator()
# df['SMA_h_200'] = ta.trend.SMAIndicator(df['high'], window = 200).sma_indicator()
# df['hlc3'] = (df['high'] + df['low'] + df['close'])/3
# hlc = (df['high'] + df['low'] + df['close'])/3
# df['SMA_hlc3_200'] = ta.trend.SMAIndicator(hlc, window = 200).sma_indicator()

# EMA
# df['EMA_h_10'] = ta.trend.ema_indicator(df['high'], window = 10)
# df['EMA_l_10'] = ta.trend.ema_indicator(df['low'], window = 10)
# hlc = (df['high'] + df['low'] + df['close'])/3
# df['EMA_hlc3_10'] = ta.trend.ema_indicator(hlc, window = 10)

# EMA for close
# df['EMA_c_15'] = ta.trend.ema_indicator(df['close'], window = 15)
# df['EMA_c_20'] = ta.trend.ema_indicator(df['close'], window = 20)
# df['EMA_c_100'] = ta.trend.ema_indicator(df['close'], window = 100)
# df['EMA_c_150'] = ta.trend.ema_indicator(df['close'], window = 150)

# # Open - Close
# df['c_minus_o'] = df['close'] - df['open']
# df['h_minus_l'] = df['high'] - df['low']

# %% Add candlesticks
cs_patterns = {
  'hammer' : lambda openP, high, low, close: talib.CDLHAMMER(openP, high, low, close),
  'closingMarubozu' : lambda openP, high, low, close:  talib.CDLCLOSINGMARUBOZU(openP, high, low, close),
  'doji' : lambda openP, high, low, close: talib.CDLDOJI(openP, high, low, close),
  'engulfing' : lambda openP, high, low, close:  talib.CDLENGULFING(openP, high, low, close),
  'hangingMan' : lambda openP, high, low, close: talib.CDLHANGINGMAN(openP, high, low, close),
  'hammer' : lambda openP, high, low, close: talib.CDLHAMMER(openP, high, low, close),
  'invertedHammer' : lambda openP, high, low, close: talib.CDLINVERTEDHAMMER(openP, high, low, close),
  'marubozu' : lambda openP, high, low, close: talib.CDLMARUBOZU(openP, high, low, close),
  'beltHold': lambda openP, high, low, close: talib.CDLBELTHOLD(openP, high, low, close),
  'breakaway' : lambda openP, high, low, close: talib.CDLBREAKAWAY(openP, high, low, close),
  'inNeck' : lambda openP, high, low, close: talib.CDLINNECK(openP, high, low, close),
  'kicking': lambda openP, high, low, close: talib.CDLKICKING(openP, high, low, close),
  'kickingByLength' : lambda openP, high, low, close: talib.CDLKICKINGBYLENGTH(openP, high, low, close),
  'ladderBottom' : lambda openP, high, low, close: talib.CDLLADDERBOTTOM(openP, high, low, close), 
  'longLeggedDoji' : lambda openP, high, low, close: talib.CDLLONGLEGGEDDOJI(openP, high, low, close),
  'longLineCdl' : lambda openP, high, low, close: talib.CDLLONGLINE(openP, high, low, close),
  'shortLineCdl' : lambda openP, high, low, close: talib.CDLSHORTLINE(openP, high, low, close),
  'rickShawMan' : lambda openP, high, low, close: talib.CDLRICKSHAWMAN(openP, high, low, close),
  'spinningTop' : lambda openP, high, low, close: talib.CDLSPINNINGTOP(openP, high, low, close),
  'stalled' : lambda openP, high, low, close: talib.CDLSTALLEDPATTERN(openP, high, low, close),
  'stickSandwhich' : lambda openP, high, low, close: talib.CDLSTICKSANDWICH(openP, high, low, close),
  'takuri' : lambda openP, high, low, close: talib.CDLTAKURI(openP, high, low, close),
  'tasukiGap' : lambda openP, high, low, close: talib.CDLTASUKIGAP(openP, high, low, close),
  'stalled' : lambda openP, high, low, close: talib.CDLSTALLEDPATTERN(openP, high, low, close),
  'thrusting' : lambda openP, high, low, close: talib.CDLTHRUSTING(openP, high, low, close),
}

ochl_df = df[['open', 'high', 'low', 'close']]

for strat_name, fn in cs_patterns.items():
  df[f'CS_{strat_name}'] = abs(fn(df['open'], df['high'], df['low'], df['close']) / 100)
# %%Get win rate over next 20 minutes (4 periods)
working_dataset = df.dropna()

working_dataset['log_ret'] = np.log(working_dataset['close'] / working_dataset['close'].shift())
working_dataset['ret'] = (working_dataset['close'] / working_dataset['close'].shift()) - 1
working_dataset['win'] = working_dataset['ret'] > 0

# Reverse dataset, then calculate win rate
reversed_dataset = working_dataset.iloc[::-1]

win_rate_col_nm = 'win_rate_30m'
reversed_dataset[win_rate_col_nm] = reversed_dataset['win'].rolling(6).mean()
working_dataset = reversed_dataset.iloc[::-1]

working_dataset = working_dataset.dropna()
working_dataset

# %%
plt.figure(figsize = (15,8))
working_dataset[win_rate_col_nm][-100:].plot()
plt.show()

# %% Get MSE by predicting all 0.5 win rate (number to beat for model to be meaningful)
working_dataset[win_rate_col_nm].apply(lambda x: (x - 0.5) ** 2).mean()

# %% Specify column names
x_columns_to_scale = [
  # 'volume', 'RSI', 'bol_upper', 'bol_middle',
  # 'bol_lower', 'SMA_l_200', 'SMA_h_200', 'hlc3', 'SMA_hlc3_200',
  # 'EMA_h_10', 'EMA_l_10', 'EMA_hlc3_10', 'EMA_c_15', 'EMA_c_20',
  # 'EMA_c_100', 'EMA_c_150'
  'RSI', 'MACD_hist'
]

x_columns_scaled = [
  # Candlestick indicators
  'CS_hammer', 'CS_closingMarubozu', 'CS_doji',
  'CS_engulfing', 'CS_hangingMan', 'CS_invertedHammer', 'CS_marubozu',
  'CS_beltHold', 'CS_breakaway', 'CS_inNeck', 'CS_kicking',
  'CS_kickingByLength', 'CS_ladderBottom', 'CS_longLeggedDoji',
  'CS_longLineCdl', 'CS_shortLineCdl', 'CS_rickShawMan', 'CS_spinningTop',
  'CS_stalled', 'CS_stickSandwhich', 'CS_takuri', 'CS_tasukiGap',
  'CS_thrusting'
]
    
y_column = win_rate_col_nm

# %% Scale and create features and target
sc = StandardScaler()
scaled_features = sc.fit_transform(working_dataset[x_columns_to_scale])
print(scaled_features.shape)

features = working_dataset[x_columns_scaled].values
print(features.shape)

features = np.concatenate((features, scaled_features), axis = 1)
print(features.shape)

target = np.array(working_dataset[y_column])

# %% convert an array of values into a dataset matrix
# Number of time_steps in each set = look_back
def create_dataset(features_df, target_df, look_back=1):
  dataX, dataY = [], []

  for i in range(0, len(target) - look_back + 1):

    dataX.append(features_df[i : (i + look_back)])
    dataY.append(target_df[(i + look_back - 1)]) # Predict last day win-rate
  return np.array(dataX), np.array(dataY)

# %%
X_full, y_full = create_dataset(features, target, 12 * 5) # 5 hours of data looked back at

# Dataset for ML = front 120 - 1 removed
print(X_full.shape)

# %% Train test spliit
num_rows = len(X_full)

train_last_row = int(num_rows * 0.7)

X_train = np.array(X_full[:train_last_row])
y_train = np.array(y_full[:train_last_row])

X_test = np.array(X_full[train_last_row:])
y_test = np.array(y_full[train_last_row:])

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %% Save x and y files
# savetxt('X_train.csv', X_train, delimiter=',')
# savetxt('y_train.csv', y_train, delimiter=',')
# savetxt('X_test.csv', X_test, delimiter=',')
# savetxt('y_test.csv', y_test, delimiter=',')

np.savez_compressed('X_train.npz', X_train)
np.savez_compressed('y_train.npz', y_train)
np.savez_compressed('X_test.npz', X_test)
np.savez_compressed('y_test.npz', y_test)

# %% Testing loading of arrays
test = np.load('X_train.npz')['arr_0']
print(test)
test

#%% Get % of positives
(y_train >= 0.75).mean()
# (y_test >= 0.75).mean()

# %%
