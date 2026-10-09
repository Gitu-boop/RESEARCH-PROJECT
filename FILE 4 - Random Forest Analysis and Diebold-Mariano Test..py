# %% MACHINE LEARNING ANALYSIS.

# Importing Essential Libraries.
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm

# Downloading Data from YFinance and Flattening it.
ethereum_data = yf.download("ETH-USD", start = "2021-01-01", end = "2025-12-31")
ethereum_data.columns = ethereum_data.columns.get_level_values(0)

# Converting the Raw Closing Price into Stationary Daily Log Returns.
ethereum_data["Returns"] = np.log(ethereum_data["Close"]/ethereum_data['Close'].shift(1))
ethereum_data["Log Volume"] = np.log(ethereum_data["Volume"])

# Defining Target Variable.
ethereum_data["target_price"] = ethereum_data["Returns"].shift(-1)

# Creating Features for the ML Model.
ethereum_data["price_one_day_before"] = ethereum_data["Returns"]
ethereum_data["volume_one_day_before"] = ethereum_data["Log Volume"]
ethereum_data["sevenday_moving_average_price"] = ethereum_data["Returns"].rolling(window=7).mean()

# Cleaning the Data off Empty Rows.
ethereum_data.dropna(inplace = True)

# Defining Training Data and Testing Data.
training_data = ethereum_data[ethereum_data.index < "2025-01-01"]
testing_data = ethereum_data[ethereum_data.index >= "2025-01-01"]

# Defining Inputs (X) and Outputs (Y).
features = ["price_one_day_before",
            "volume_one_day_before",
            "sevenday_moving_average_price"]
X_train, y_train = (training_data[features],
                    training_data["target_price"])
X_test, y_test = (testing_data[features],
                  testing_data["target_price"])

# Training the ML Model.
model = RandomForestRegressor(n_estimators=100, random_state=5)
model.fit(X_train, y_train)

"""
100 individual decision trees built, as 100 is the standard industry default.
The random seed is 5, guaranteeing that the exact same results will be produced when you run it.
Usually 42 is used as the random seed as it is a famous joke in the data science community that 42 will solve 
the ultimate question of life. The number 42 could be changed to anything, in this case 5, and the code will still work the same.
"""

# Generating Predictions.
moving_average_predictions = testing_data["sevenday_moving_average_price"]
machine_learning_predictions = model.predict(X_test)

# Calculating Error Rates.
mean_absolute_error_moving_average = mean_absolute_error(y_test, moving_average_predictions)
mean_absolute_error_machine_learning = mean_absolute_error(y_test, machine_learning_predictions)

print("The 7-Day Moving Average method has a Mean Absolute Error of " +
      str(round(mean_absolute_error_moving_average,6)))
print("The Machine Learning method has a Mean Absolute Error of " +
      str(round(mean_absolute_error_machine_learning,6)))

# Calculating Daily Absolute Error Vectors.
y_test_flattened = np.asarray(y_test).flatten()
moving_average_flattened = np.asarray(moving_average_predictions).flatten()
machine_learning_flattened = np.asarray(machine_learning_predictions).flatten()

# Calculating the Difference across Vectors.
moving_average_error = np.abs(y_test_flattened - moving_average_flattened)
machine_learning_error = np.abs(y_test_flattened - machine_learning_flattened)

# Implementing Statistical Significance Mathematical Function.
difference_vector = (moving_average_error**2) - (machine_learning_error**2)
mean_of_difference_vector = float(np.mean(difference_vector))
variance_of_difference_vector = float(np.var(difference_vector)/len(difference_vector))

diebold_marino_statistic = mean_of_difference_vector/np.sqrt(variance_of_difference_vector)
p_value = float(2 * (1 - norm.cdf(np.abs(diebold_marino_statistic))))

print("The DieBold-Marino test statistic is " + str(np.round(diebold_marino_statistic, 4)))
print("The P-value is " + str(np.round(p_value, 4)))


#=====================================================================================================#

#  LINE GRAPH COMPARING ACTUAL VALUES, MOVING AVERAGE PREDICTION AND ML PREDICTION FOR 2025.

# Importing Essential Libraries.
import pandas as pd
import matplotlib.pyplot as plt

# Variables for 2025 Test Frame.
monthly_test_data = testing_data.copy()
monthly_predictions = machine_learning_predictions

# Setting Up Graph.
plt.figure(figsize=(10,5))
plt.plot(monthly_test_data.index,
         monthly_test_data['target_price'],
         label='Actual Next-Day Price', color='red', linewidth=2)
plt.plot(monthly_test_data.index,
         monthly_test_data['sevenday_moving_average_price'],
         label='7-Day Moving Average Predictions',
         color='blue', linewidth=1.5)
plt.plot(monthly_test_data.index,
         monthly_predictions,
         label='Random Forest Predictions',
         color='green', linewidth=1.5)

# Final Adjustments.
plt.title('2025 RETURNS PREDICTIONS FORECAST ',
          fontsize=12, fontweight='bold')
plt.xlabel('DATE', fontsize=12)
plt.ylabel('ETHEREUM PRICE ($)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Comparison Line Graph', dpi=300)
plt.show()

#====================================================================================================#

#  LINE GRAPH COMPARING ACTUAL VALUES, MOVING AVERAGE PREDICTION AND ML PREDICTION FOR JANUARY 2025.

# Variables for January 2025 Test Frame.
monthly_test_data = testing_data.head(30)
monthly_predictions = machine_learning_predictions[:30]

# Setting Up Graph.
plt.figure(figsize=(10,5))
plt.plot(monthly_test_data.index,
         monthly_test_data['target_price'],
         label='Actual Next-Day Price', color='red', linewidth=2)
plt.plot(monthly_test_data.index,
         monthly_test_data['sevenday_moving_average_price'],
         label='7-Day Moving Average Predictions',
         color='blue', linewidth=1.5)
plt.plot(monthly_test_data.index,
         monthly_predictions,
         label='Random Forest Predictions',
         color='green', linewidth=1.5)

# Final Adjustments.
plt.title('JANUARY 2025 RETURNS PREDICTIONS FORECAST',
          fontsize=12, fontweight='bold')
plt.xlabel('DATE', fontsize=12)
plt.ylabel('ETHEREUM PRICE ($)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('Comparison Line Graph', dpi=300)
plt.show()