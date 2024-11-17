11.Weather Prediction  
Scenario: You create a simple linear regression model to predict daily temperatures based on historical weather data.  
Question: If your predictions are consistently inaccurate because they do not capture seasonal variations, what does this indicate about bias? How might you improve your model's accuracy?

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the weather data
file_path = "/mnt/data/weather_data.csv"
weather_data = pd.read_csv(file_path)

# Prepare the data (extracting day of the year for simple linear regression)
weather_data['DayOfYear'] = weather_data['Date'].apply(lambda x: pd.to_datetime(x).dayofyear)

# Linear regression model
X = weather_data[['DayOfYear']]  # Feature: Day of the year
y = weather_data['Temperature']  # Target: Temperature

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
weather_data['Predicted_Temperature'] = model.predict(X)

# Plot the actual vs predicted temperatures
plt.figure(figsize=(10, 6))
plt.plot(weather_data['Date'], weather_data['Temperature'], label='Actual Temperature', color='blue')
plt.plot(weather_data['Date'], weather_data['Predicted_Temperature'], label='Predicted Temperature', color='red', linestyle='--')
plt.title('Actual vs Predicted Temperature (Linear Regression)', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.xticks(weather_data['Date'][::30], rotation=45)  # Show every 30th date for better visibility
plt.tight_layout()
plt.show()
