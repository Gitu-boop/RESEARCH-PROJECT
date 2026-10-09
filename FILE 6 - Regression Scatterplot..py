# %% PRICE AND VOLUME SCATTERPLOT WITH REGRESSION LINE

# Importing Essential libraries.
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf

# # Downloading Ethereum Closing Price and Trading Volume (1 Jan 2021 TO 31 Dec 2025).
ethereum_data = yf.download("ETH-USD", start= "2021-01-01", end = "2025-12-31")

# Flattening the Data:
ethereum_data.columns = ethereum_data.columns.get_level_values(0)

# Converting the Raw Data into Stationary Values.
ethereum_data["Returns"] = np.log(ethereum_data["Close"]/ethereum_data["Close"].shift(1))
ethereum_data["Log Volume"] = np.log(ethereum_data["Volume"])
ethereum_data.dropna(inplace=True)

# Setting Up Graph Size.
plt.figure(figsize = (10,10))
sns.regplot(x=ethereum_data["Log Volume"], y=ethereum_data["Returns"],
            scatter_kws= {'alpha':0.4, 'color':'gray'},
            line_kws={'color':'red', 'linewidth':2})

# Final Adjustments
plt.title('LOG RETURNS AND LOG TRADING VOLUME SCATTERPLOT (CORRELATION = -0.01)', fontsize = 14, fontweight = 'bold')
plt.xlabel('ETHEREUM DAILY LOG RETURNS', size=12)
plt.ylabel('LOG TRADING VOLUME', size = 12)
plt.grid(True, linestyle= '--', alpha=0.5)
plt.tight_layout()
plt.savefig('Ethereum Closing Price and Volume Scatter Plot with Regression Line', dpi=300)
plt.show()