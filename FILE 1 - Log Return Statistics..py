# %% CLOSING PRICE STATISTICS.

# Importing Essential Libraries.
import yfinance
import yfinance as yf
import numpy as np

# Downloading Ethereum Closing Price (1 Jan 2021 TO 31 Dec 2025).
ethereum_data = yf.download("ETH-USD", start= "2021-01-01", end = "2025-12-31")

# Flattening the Data.
ethereum_data.columns = ethereum_data.columns.get_level_values(0)

# Converting the Raw Closing Price into Stationary Daily Log Returns.
ethereum_closing_price = np.log(ethereum_data['Close']/ethereum_data['Close'].shift(1))
ethereum_closing_price.dropna(inplace=True)

# Calculating Mean.
print ("The mean is " + str(ethereum_closing_price.mean()))

# Calculating Median.
print ("The median is " + str(ethereum_closing_price.median()))

# Calculating Standard Deviation.
print ("The SD is " + str(ethereum_closing_price.std()))

# Calculating Variance.
print ("The variance is " + str(ethereum_closing_price.var()))

# Calculating Maximum Value.
print ("The maximum value is " + str(ethereum_closing_price.max()))

# Calculating Minimum Value.
print ("The minimum value is " + str(ethereum_closing_price.min()))

# Calculating 7-Day Simple Moving Average for Closing Price
seven_day_moving_average = (ethereum_closing_price.rolling(7).mean())
print("The 7-Day moving average of the closing price is:" + str(seven_day_moving_average.tail()))





