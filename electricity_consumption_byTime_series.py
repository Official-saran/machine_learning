2. Time Series Data  
Scenario: You have monthly data on electricity consumption over several years and want to predict future consumption based on trends and seasonal patterns.  
 Question: Can linear regression be effectively used in this scenario? If so, how would you incorporate time as a variable in your model?



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate synthetic time series data
np.random.seed(42)

# Generate monthly data over 5 years (60 months)
months = pd.date_range(start="2018-01-01", periods=60, freq="M")
time_index = np.arange(1, 61)  # Time as sequential numbers (1 to 60)

# Create a seasonal pattern and trend
seasonal_pattern = 10 * np.sin(2 * np.pi * time_index / 12)  # Seasonal variation (annual cycle)
trend = 50 + 0.5 * time_index  # Linear upward trend
random_noise = np.random.normal(0, 5, len(time_index))  # Random noise
electricity_consumption = trend + seasonal_pattern + random_noise

# Create DataFrame
time_series_data = pd.DataFrame({
    "Month": months,
    "Time_Index": time_index,
    "Electricity_Consumption": electricity_consumption
})

# Save to CSV
time_series_csv_path = 'electricity_consumption_data.csv'
time_series_data.to_csv(time_series_csv_path, index=False)

# Step 2: Load the time series data
data = pd.read_csv(time_series_csv_path)

# Add seasonal features (sine and cosine of time index)
data["Sin_Month"] = np.sin(2 * np.pi * data["Time_Index"] / 12)
data["Cos_Month"] = np.cos(2 * np.pi * data["Time_Index"] / 12)

# Define features and target
X = data[["Time_Index", "Sin_Month", "Cos_Month"]]
y = data["Electricity_Consumption"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Extract coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Step 4: Print results
print("Time Series Linear Regression Model Results")
print("-------------------------------------------")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Intercept: {intercept:.2f}")
print("Coefficients:")
print(f"  Time Index (Trend): {coefficients[0]:.2f}")
print(f"  Sine of Month (Seasonality): {coefficients[1]:.2f}")
print(f"  Cosine of Month (Seasonality): {coefficients[2]:.2f}")

# Step 5: Save the model predictions and actual values for comparison
predictions_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Month": data.loc[y_test.index, "Month"]
}).sort_values(by="Month")

predictions_csv_path = 'time_series_predictions.csv'
predictions_df.to_csv(predictions_csv_path, index=False)

print(f"\nPrediction results saved to {predictions_csv_path}")
