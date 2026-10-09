# %% TRADING VOLUME STATISTICS.

# Downloading Ethereum Closing Price (1 Jan 2021 TO 31 Dec 2025).
import yfinance as yf
import numpy as np

# Downloading Ethereum Closing Price (1 Jan 2021 TO 31 Dec 2025).
ethereum_data = yf.download("ETH-USD", start= "2021-01-01", end = "2025-12-31")

# Flattening the Data.
ethereum_data.columns = ethereum_data.columns.get_level_values(0)

# Converting the Raw Closing Price into Stationary Daily Log Returns.
ethereum_trading_volume = np.log(ethereum_data['Volume'])
ethereum_trading_volume.dropna(inplace=True)

# Calculating Mean.
print ("The mean is " + str(ethereum_trading_volume.mean()))

# Calculating Median.
print ("The median is " + str(ethereum_trading_volume.median()))

# Calculating Standard Deviation.
print ("The SD is " + str(ethereum_trading_volume.std()))

# Calculating Variance
print ("The variance is " + str(ethereum_trading_volume.var()))

# Calculating Maximum Value.
print ("The maximum value is " + str(ethereum_trading_volume.max()))

# Calculating Minimum Value.
print ("The minimum value is " + str(ethereum_trading_volume.min()))

# Calculating 7-Day Simple Moving Average for Trading Volume
seven_day_moving_average = ethereum_trading_volume.rolling(7).mean()
print("The 7-Day moving average of the trading volume is:" + str(seven_day_moving_average.tail()))