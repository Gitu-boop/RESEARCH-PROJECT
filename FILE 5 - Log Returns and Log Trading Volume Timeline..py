# %% PRICE AND VOLUME TIMELINE

# Importing Essential Libraries.
import matplotlib.pyplot as plt

# Downloading Closing Price and Trading Volume Data.
import yfinance as yf
import numpy as np
import yfinance as yf

# Downloading Ethereum Closing Price and Trading Volume (1 Jan 2021 TO 31 Dec 2025).
ethereum_data = yf.download("ETH-USD", start= "2021-01-01", end = "2025-12-31")

# Flattening the Data,
ethereum_data.columns = ethereum_data.columns.get_level_values(0)

# Converting the Raw Data into Stationary Values.
ethereum_data["Returns"] = np.log(ethereum_data["Close"]/ethereum_data["Close"].shift(1))
ethereum_data["Log Volume"] = np.log(ethereum_data["Volume"])
ethereum_data.dropna(inplace=True)

# EXTRACTING SQUEEZED ARRAYS
ethereum_closing_price = ethereum_data["Returns"].squeeze()
ethereum_trading_volume = ethereum_data["Log Volume"].squeeze()

# Setting Up Graph Size.
figure, axes = plt.subplots(figsize=(10,5))

# Plotting Closing Price on the Left Axis.
color = '#1f77b4'
axes.set_xlabel ('DATE (2021-2025)', fontsize=12)
axes.set_ylabel('ETHEREUM DAILY LOG RETURNS)',
                color=color, fontsize = 12)
axes.plot(ethereum_data.index,
          ethereum_closing_price,
          color=color, linewidth =1.5, label = 'Closing Price')
axes.tick_params(axis='y', labelcolor=color)

# Plotting Trading Volume as a Shaded Area on the Right Axis.
axes_volume = axes.twinx()
color_volume = '#ff7f0e'
axes_volume.set_ylabel ('LOG TRADING VOLUME',
                        color=color_volume, fontsize = 12)
axes_volume.fill_between(ethereum_data.index,
                         ethereum_trading_volume,
                         color= color_volume, alpha=0.3, label = 'TRADING VOLUME')
axes_volume.tick_params(axis='y', labelcolor=color_volume)

# Final Adjustments.
plt.title('LOG RETURNS AND LOG TRADING VOLUME TIMELINE',
          fontsize = 14, fontweight = 'bold')
figure.tight_layout()
plt.savefig('Ethereum Price and Volume Timeline', dpi=300)
plt.show()