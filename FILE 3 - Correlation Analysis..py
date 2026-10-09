# %% CORRELATION ANALYSIS.

# Importing Essential Libraries.
import yfinance as yf
import numpy as np

# Downloading Ethereum Closing Price and Trading Volume (1 Jan 2021 TO 31 Dec 2025).
ethereum_data = yf.download("ETH-USD", start="2021-01-01", end="2025-12-31")

# Flattening the Data.
ethereum_data.columns = ethereum_data.columns.get_level_values(0)

# Mathematically Transforming the Data.
ethereum_data["Closing Price"] = np.log(ethereum_data["Close"]/ethereum_data["Close"].shift(1))
ethereum_data["Trading Volume"] = np.log(ethereum_data["Volume"])
ethereum_data.dropna(inplace=True)

# Calculating the Correlation Coefficient.
correlation_coefficient = ethereum_data["Closing Price"].corr(ethereum_data["Trading Volume"])
print("The correlation coefficient is " + str(round(correlation_coefficient, 2)))

