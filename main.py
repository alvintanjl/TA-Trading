#%%
# Imports
import plotly.express as px
import numpy as np
import pandas as pd
import requests
import datetime as dt
import ta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
import json
warnings.filterwarnings("ignore")

#%%
# Read in API Key
with open('secrets.json', 'r') as f:
  SECRETS = json.load(f)

API_KEY = SECRETS['POLYGON_API']
#%%
def get_data(sym):
  '''
  Function to get stock data for specified symbol
  '''
  url = "https://api.polygon.io/v2/aggs/ticker/"
  # sym = "AAPL"
  t_interval_1min = "/range/1/minute/"
  t_interval_1hr = "/range/1/hour/"
  t_interval_1day = "/range/1/day/"

  d_range = 1000
  date_to = dt.datetime.now().strftime("%Y-%m-%d")
  date_from = (dt.datetime.now()- dt.timedelta(days = d_range)).strftime("%Y-%m-%d")
  date_input = date_from + "/" + date_to

  f_url = url + sym + t_interval_1day + date_input

  params = {
      'unadjusted':'true',
      'sort':'asc',
      'limit':'50000', # max 50000
      'apiKey':API_KEY
  }

  resp = requests.get(f_url, params)

  # print(resp.json().keys())
  print('API Call returned ' + str(resp.json()['resultsCount']) + ' rows')

  df = pd.DataFrame(resp.json()['results'])
  df['t'] = df['t'].apply(float)
  df['t'] = df['t'] / 1000
  df['t'] = df['t'].apply(dt.datetime.fromtimestamp)

  df.rename(columns = {'v':'vol', 'o':'open', 'c':'close', 'h':'high', 'l':'low', 't':'time', 'n':'num'}, inplace = True)
  df.time = pd.to_datetime(df.time)
  df = df.set_index('time')
  df = df.sort_index(ascending = True)
  df = df.reset_index()
  df_long = pd.melt(df, id_vars = ['time'], var_name = 'type', value_name = 'value')

  # Return 2 types of df
  return df, df_long

def add_ta_indicators(df):
  '''
  Function to add other technical analysis indicators from the TA package
  '''
  # RSI
  df['RSI'] = ta.momentum.RSIIndicator(df['close'], window = 14).rsi()

  # Bollinger Band
  bol = ta.volatility.BollingerBands(df['close'], window = 14)
  df['bol_upper'] = bol.bollinger_hband()
  df['bol_lower'] = bol.bollinger_lband()
  # plt.plot(df['time'], df['close'])
  # plt.plot(df['time'], df['bol_upper'], color = 'orange')
  # plt.plot(df['time'], df['bol_lower'], color = 'orange')
  # plt.show()

  # SMA
  df['SMA_Low_200'] = ta.trend.SMAIndicator(df['low'], window = 200).sma_indicator()
  df['SMA_High_200'] = ta.trend.SMAIndicator(df['high'], window = 200).sma_indicator()
  df['hlc3'] = (df['high'] + df['low'] + df['close'])/3
  df['SMA_hlc3_200'] = ta.trend.SMAIndicator(df['hlc3'], window = 200).sma_indicator()

  # EMA
  df['EMA_High_10'] = ta.trend.ema_indicator(df['high'], window = 10)
  df['EMA_Low_10'] = ta.trend.ema_indicator(df['low'], window = 10)
  df['EMA_hlc3_10'] = ta.trend.ema_indicator(df['hlc3'], window = 10)

  # ADX
  df['ADX_5'] = ta.trend.ADXIndicator(
    high = df['high'],
    low = df['low'],
    close = df['close'],
    window = 5,
  ).adx()

  # ADX Slope
  df['ADX Slope Up'] = False
  for idx, row in df.iterrows():
    if pd.isna(row['ADX_5']):
      continue
    elif idx == 0:
      continue
    elif pd.isna(df.loc[(idx - 1), 'ADX_5']):
      continue
    elif idx >= (df.shape[0] - 1):
      continue
    else:
      curr = row['ADX_5']
      prev = df.loc[(idx - 1), 'ADX_5']
      next = df.loc[(idx + 1), 'ADX_5']
      if (curr > prev) and (next > curr):
        df.loc[idx, 'ADX Slope Up'] = True

  # Return Data Frame
  return df

def add_cobra_strat(df):
  
  def entry_cond(EMA_low, EMA_high, SMA_high, cs_type, close, ADX, ADX_slope_up):
    # Test if any is NA
    for input_ in [EMA_low, EMA_high, SMA_high, cs_type, close, ADX, ADX_slope_up]:
      if pd.isna(input_):
        return False
    
    entry_cond = (EMA_low > SMA_high) and (cs_type == 'BULLISH') and \
        (close > EMA_high) and (ADX >= 40) and (ADX <= 75) and \
        (ADX_slope_up == True)
    
    return entry_cond
  
  df['[CO] Entry'] = df.apply(
    lambda row: entry_cond(
      row['EMA_Low_10'],
      row['EMA_High_10'],
      row['SMA_High_200'],
      row['Candlestick Type'],
      row['close'],
      row['ADX_5'],
      row['ADX Slope Up']
    ),
    axis = 1
  )

  def exit_cond(cs_type, close, EMA_low):
    if pd.isna(EMA_low):
      return False
    elif (cs_type == 'BEARISH') and (close < EMA_low):
      return True
    else:
      return False

  df['[CO] Exit'] = df.apply(
    lambda row: exit_cond(
      row['Candlestick Type'],
      row['close'],
      row['EMA_Low_10']
    ),
    axis = 1
  )

  return df

def add_entry_exit_price(df):
  '''
  Function to add entry and exit prices to dataframe
  '''
  df['Entry Price'] = df['open']
  df['Exit Price'] = df['open']
  return df

def plot_candlestick(df_in, start_date = None, end_date = None):
  # Get subset of data
  if (start_date != None) & (end_date != None):
    df = df_in[ (df_in['time'] >= start_date) & (df_in['time'] <= end_date) ]
  else:
    df = df_in

  # Plot
  fig = go.Figure(
    data = [
      go.Candlestick(
        x = df['time'],
        open = df['open'],
        close = df['close'],
        high = df['high'],
        low = df['low']
      )
    ]
  )
  fig.update_layout(height=1000)
  fig.write_html('candlestick.html')
  fig.show()

def cs_type(open, close):
  '''Funciton to return candlestick type'''
  if open < close:
    return 'BULLISH'
  elif open > close:
    return 'BEARISH'
  else:
    return 'NILL'

def add_candlestick(df):
  '''
  Function to add candlestick type and create boolean indicators for each of the candlestick patterns
  '''
  # Get candlestick type
  df['Candlestick Type'] = df.apply(lambda row: cs_type(row['open'], row['close']), axis = 1)

  # Create indicators columns signifying entry or exit
  df['[CS] Entry'] = False
  df['[CS] Exit'] = False

  # Initialise constant
  nrows = df.shape[0]

  ####################################################
  # Add in criteria to hit 5 first within each loop ##
  #####################################################

  ######################
  ## [BULLISH] Hammer ##
  ######################

  df['[BULLISH] Hammer'] = False

  for index, row in df.iterrows():

    if row['Candlestick Type'] == 'BULLISH':

      TRa = row['open'] - row['low']
      TRb = row['high'] - row['close']
      TRc = row['close'] - row['open']
      
      # Criteria 1 met
      if (TRc > 0) & (TRb < TRc) & (TRa > (2 * TRc)):

        # Look at next candlestick
        if ( index < (nrows - 1) ):

          if (df.loc[(index + 1), 'Candlestick Type'] == 'BULLISH'):
            # Look at past candlesticks
            temp = df.iloc[:index, :]
            past_bearish_df = temp[temp['Candlestick Type'] == 'BEARISH']

            # If no bearish candlestick in the past
            if past_bearish_df.empty == False:
              last_bearish_high = past_bearish_df.iloc[-1,]['high']
              next_day_close = df.loc[(index + 1), 'close']
              
              # If last bearish high lower than closing price
              if next_day_close > last_bearish_high:
                df.loc[index, '[BULLISH] Hammer'] = True

                # Entry
                if ( (index + 2) < nrows):
                  df.loc[(index + 2), '[CS] Entry'] = True

  ########################
  ## [BULLISH] Piercing ##
  ########################

  df['[BULLISH] Piercing'] = False

  for index, row in df.iterrows():

    if row['Candlestick Type'] == 'BEARISH':

      # Criteria 1 - Look at next candlestick
      if ( index < (nrows - 2) ):

        if (df.loc[(index + 1), 'Candlestick Type'] == 'BULLISH'):
          candle_2_open = df.loc[(index + 1), 'open']

          # Criteria 2
          if (row['low'] > candle_2_open):
            candle_1_body_mid = row['close'] + 0.5 * (row['open'] - row['close'])
            candle_2_close = df.loc[(index + 1), 'close']

            # Criteria 3
            if candle_2_close > candle_1_body_mid:

              # Criteria 4 (3rd candle)
              if (df.loc[(index + 2), 'Candlestick Type'] == 'BULLISH'):
                candle_3_close = df.loc[(index + 2), 'close']

                if (row['high'] < candle_3_close):
                  df.loc[index, '[BULLISH] Piercing'] = True

                  # Entry
                  if ( (index + 3) < nrows):
                    df.loc[(index + 3), '[CS] Entry'] = True
        
  #########################
  ## [BULLISH] Engulfing ##
  #########################

  df['[BULLISH] Engulfing'] = False
  nrows = df.shape[0]

  for index, row in df.iterrows():
    if row['Candlestick Type'] == 'BEARISH':

      # Criteria 1 - Look at next candlestick
      if ( index < (nrows - 1) ):

        if (df.loc[(index + 1), 'Candlestick Type'] == 'BULLISH'):
          candle_2_open = df.loc[(index + 1), 'open']

          # Criteria 2
          if (row['close'] > candle_2_open):
            candle_2_close = df.loc[(index + 1), 'close']

            # Criteria 3
            if (candle_2_close > row['open']):
              df.loc[index, '[BULLISH] Engulfing'] = True

              # Entry
              if ( (index + 2) < nrows):
                df.loc[(index + 2), '[CS] Entry'] = True


  #################################
  ## [BULLISH] One White Soldier ##
  #################################

  df['[BULLISH] One White Soldier'] = False
  nrows = df.shape[0]

  for index, row in df.iterrows():
    if row['Candlestick Type'] == 'BEARISH':

      # Criteria 1 - Look at next candlestick
      if ( index < (nrows - 1) ):

        if (df.loc[(index + 1), 'Candlestick Type'] == 'BULLISH'):
          candle_2_open = df.loc[(index + 1), 'open']

          # Criteria 2
          if (row['close'] < candle_2_open):
            candle_2_close = df.loc[(index + 1), 'close']

            # Criteria 3
            if (candle_2_close > row['high']):
              df.loc[index, '[BULLISH] One White Soldier'] = True

              # Entry
              if ( (index + 2) < nrows):
                df.loc[(index + 2), '[CS] Entry'] = True

  ##############################
  ## [BULLISH] Morning Gap Up ##
  ##############################

  df['[BULLISH] Morning Gap Up'] = False
  nrows = df.shape[0]

  for index, row in df.iterrows():
    if row['Candlestick Type'] == 'BEARISH':

      # Criteria 1 - Look at next candlestick
      if ( index < (nrows - 1) ):

        if (df.loc[(index + 1), 'Candlestick Type'] == 'BULLISH'):
          candle_2_open = df.loc[(index + 1), 'open']

          # Criteria 2
          if (row['high'] < candle_2_open):
            df.loc[index, '[BULLISH] Morning Gap Up'] = True

            # Entry
            if ( (index + 2) < nrows):
              df.loc[(index + 2), '[CS] Entry'] = True

  #############################
  ## [BEARISH] Shooting Star ##
  #############################

  df['[BEARISH] Shooting Star'] = False
  nrows = df.shape[0]

  for index, row in df.iterrows():

    # Same day indicators

    if row['Candlestick Type'] == 'BULLISH':
      upper_shadow = row['high'] - row['close']
      body = row['close'] - row['open']
      lower_shadow = row['open'] - row['low']
    
    else:
      upper_shadow = row['high'] - row['open']
      body = row['open'] - row['close']
      lower_shadow = row['close'] - row['low']
    
    # Same day criteria
    if (lower_shadow < body) & (upper_shadow > (2 * body)):
      
      # Next day criteria
      if ( index < (nrows - 1) ):
        if (df.loc[(index + 1), 'Candlestick Type'] == 'BEARISH'):

            # Look at past candlesticks
            temp = df.iloc[:index, :]
            past_bullish_df = temp[temp['Candlestick Type'] == 'BULLISH']

            # If no bearish candlestick in the past
            if past_bullish_df.empty == False:
              last_bullish_low = past_bullish_df.iloc[-1,]['low']
              next_day_close = df.loc[(index + 1), 'close']
              
              # If last bearish high lower than closing price
              if next_day_close < last_bullish_low:
                df.loc[index, '[BEARISH] Shooting Star'] = True

                # Exit
                if ( (index + 2) < nrows):
                  df.loc[(index + 2), '[CS] Exit'] = True

  ################################
  ## [BEARISH] Dark Cloud Cover ##
  ################################

  df['[BEARISH] Dark Cloud Cover'] = False
  nrows = df.shape[0]

  for index, row in df.iterrows():
    if row['Candlestick Type'] == 'BULLISH':

      # Criteria 1 - Look at next candlestick
      if ( index < (nrows - 2) ):

        if (df.loc[(index + 1), 'Candlestick Type'] == 'BEARISH'):
          candle_2_open = df.loc[(index + 1), 'open']

          # Criteria 2
          if (row['high'] < candle_2_open):

            candle_1_body_mid = row['open'] + 0.5 * (row['close'] - row['open'])
            candle_2_close = df.loc[(index + 1), 'close']

            # Criteria 3
            if candle_2_close < candle_1_body_mid:

              # Criteria 4 (3rd candle)
              if (df.loc[(index + 2), 'Candlestick Type'] == 'BEARISH'):
                candle_3_close = df.loc[(index + 2), 'close']

                if (row['low'] > candle_3_close):
                  df.loc[index, '[BEARISH] Dark Cloud Cover'] = True

                  # Exit
                  if ( (index + 3) < nrows):
                    df.loc[(index + 3), '[CS] Exit'] = True

  #########################
  ## [BEARISH] Engulfing ##
  #########################

  df['[BEARISH] Engulfing'] = False
  nrows = df.shape[0]

  for index, row in df.iterrows():
    if row['Candlestick Type'] == 'BULLISH':

      # Criteria 1 - Look at next candlestick
      if ( index < (nrows - 1) ):

        if (df.loc[(index + 1), 'Candlestick Type'] == 'BEARISH'):
          candle_2_open = df.loc[(index + 1), 'open']

          # Criteria 2
          if (row['close'] < candle_2_open):
            candle_2_close = df.loc[(index + 1), 'close']

            # Criteria 3
            if candle_2_close < row['open']:
              df.loc[index, '[BEARISH] Engulfing'] = True

              # Exit
              if ( (index + 2) < nrows):
                df.loc[(index + 2), '[CS] Exit'] = True

  ##############################
  ## [BEARISH] One Black Crow ##
  ##############################

  df['[BEARISH] One Black Crow'] = False
  nrows = df.shape[0]

  for index, row in df.iterrows():
    if row['Candlestick Type'] == 'BULLISH':

      # Criteria 1 - Look at next candlestick
      if ( index < (nrows - 1) ):

        if (df.loc[(index + 1), 'Candlestick Type'] == 'BEARISH'):
          candle_2_open = df.loc[(index + 1), 'open']

          # Criteria 2
          if (row['close'] > candle_2_open):
            candle_2_close = df.loc[(index + 1), 'close']

            # Criteria 3
            if candle_2_close < row['low']:
              df.loc[index, '[BEARISH] One Black Crow'] = True

              # Exit
              if ( (index + 2) < nrows):
                df.loc[(index + 2), '[CS] Exit'] = True

  ############################
  ## [BEARISH] Evening fall ##
  ############################

  df['[BEARISH] Evening fall'] = False
  nrows = df.shape[0]

  for index, row in df.iterrows():
    if row['Candlestick Type'] == 'BULLISH':

      # Criteria 1 - Look at next candlestick
      if ( index < (nrows - 1) ):

        if (df.loc[(index + 1), 'Candlestick Type'] == 'BEARISH'):
          candle_2_open = df.loc[(index + 1), 'open']

          # Criteria 2
          if (row['low'] > candle_2_open):
            candle_2_close = df.loc[(index + 1), 'close']

            # Criteria 3
            if candle_2_close < row['open']:
              df.loc[index, '[BEARISH] Evening fall'] = True

              # Exit
              if ( (index + 2) < nrows):
                df.loc[(index + 2), '[CS] Exit'] = True

  ######################
  ## Return dataframe ##
  ######################
  return df

def lds(arr, ascending):
  '''
  Longest Decreasing Subsequence Algorithm (Dynamic Programming)
  Time Complexity = O(n2)
  # Testing LDS:
  # lst = [1,3,2,1,4,6,5,7]
  # print(lds(lst, False))
  '''
  n = len(arr)
  lds = [0] * n

  # Initialise all subsequence length to 1
  for i in range(n):
    lds[i] = 1
  
  if ascending == True:
    # Compute LDS from every index in bottom up manner
    for i in range(1, n):
        for j in range(i):
            if (arr[i] > arr[j] and lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
  
  else:
    for i in range(1, n):
        for j in range(i):
            if (arr[i] < arr[j] and lds[i] < lds[j] + 1):
                lds[i] = lds[j] + 1
 
  # Select the maximum of all the LDS values
  max = 0
  for i in range(n):
      if (max < lds[i]):
          max = lds[i]
 
  # Returns the length of the LDS
  return max

def get_num_pre_signals(df_in, strat_type, end_date, num_days):
  '''
  Function to get number of pre signals
  '''
  # Make adjustments because date in is for 0000HRS
  start_date = end_date - dt.timedelta(num_days)
  end_date = end_date + dt.timedelta(1)
  temp = df_in[ (df_in['time'] >= start_date) & (df_in['time'] <= end_date)]

  # If strat type is bullish, looking for bearish
  if strat_type == 'BULLISH':
    close_lst = list(temp.loc[(temp['Candlestick Type'] == 'BEARISH'), 'close'])
    return lds(close_lst, False)

  # If strat type is bullish, looking for bearish
  elif strat_type == 'BEARISH':
    close_lst = list(temp.loc[(temp['Candlestick Type'] == 'BULLISH'), 'close'])
    return lds(close_lst, True)
  
  # Else error
  else:
    return 0

def backtest(df):
  '''
  Function to backtest data based on entry and exit signals in algorithm
  '''

  # STOP LOSS FUNCTION

  # Extract Action Days
  action_days = df.loc[ \
    (df['Entry'] | df['Exit']), 
    ['time', 'Entry', 'Entry Price', 'Exit', 'Exit Price']
  ].reset_index(drop = True)

  # Set default values
  action_days['Profit'] = np.nan
  action_days['Action'] = False
  profit = 0
  stock_held = False
  stock_value = 0

  # Loop through all action days
  for idx, row in action_days.iterrows():

    # Entry with criterias
    if (stock_held == False) & (row['Entry'] == True):
      stock_held = True
      stock_value = row['Entry Price']
      action_days.loc[idx, 'Profit'] = profit
      action_days.loc[idx, 'Action'] = True
    
    # Exit with criterias
    elif (stock_held == True) & (row['Exit'] == True):
      stock_held = False
      profit = profit - stock_value + row['Exit Price']
      stock_value = row['Entry Price']
      action_days.loc[idx, 'Profit'] = profit
      action_days.loc[idx, 'Action'] = True

  action_days = action_days[action_days['Action']].drop(columns = 'Action').reset_index(drop = True)
  # action_days.to_excel('Backtest.xlsx')
  return action_days

def run(symbol):
  df, df_long = get_data(symbol)
  df = add_ta_indicators(df)
  df = add_candlestick(df)
  df = add_cobra_strat(df)
  df = add_entry_exit_price(df)

  print(df.columns.values)

  # Plot candlesticks
  plot_candlestick(df)

  # Manipulate data to fit backtest (takes in Entry, Exit)
  df['Entry'] = df['[CS] Entry']
  df['Exit'] = df['[CS] Exit']

  # Calculate outputs
  output = backtest(df)
  first_buy_price = output.loc[0, 'Entry Price']
  profit = round(output.iloc[-1,-1], 2)
  print('Start Price:  ' + str(first_buy_price))
  print('Final Profit: ' + str(profit))
  print('% Returns:    ' + str(round((profit * 100) / first_buy_price, 2)) + '%')
  return backtest(df)

  # Testing
  # print(df.columns.values)
  # candlestick_cols = list(filter( lambda x: x[0] == '[', list(df.columns.values)))
  # print(candlestick_cols)

  # Future enhancements

  # # For testing each strategy
  # # Year, Month, Day, Hour, Minute
  # start_date_in = dt.datetime(2019, 1, 1)
  # end_date_in = dt.datetime(2021, 7, 5, 23, 59)
  # plot_candlestick(df, start_date_in, end_date_in)

  # # For 5 day analysis
  # end_date_in = dt.datetime(2020, 5, 3)
  # print(get_num_pre_signals(df, 'BEARISH', end_date_in, 10))

# %%
run('AAPL')
# %%
