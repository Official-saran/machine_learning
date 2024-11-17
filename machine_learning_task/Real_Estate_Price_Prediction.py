9. Real Estate Price Prediction  
Scenario: A real estate company wants to predict house prices based on features like location, size, and number of bedrooms.  
Question: If your model's R-squared value is 0.85, what does this indicate about the model's performance? Are there any limitations to using R-squared as the sole metric?

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load the data
csv_path = 'real_estate_data.csv'
data = pd.read_csv(csv_path)

# One-hot encode categorical 'Location'
encoder = OneHotEncoder(drop="first", sparse=False)
location_encoded = encoder.fit_transform(data[["Location"]])

# Combine encoded data with numerical features
X = pd.concat([
    pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out()),
    data[["Size_sqft", "Bedrooms", "Age_years"]]
], axis=1)
y = data["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# R-squared value
r_squared = r2_score(y_test, y_pred)
print(f"R-squared value: {r_squared:.2f}")

# Additional metrics
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")

# Coefficients and interpretation
coefficients = pd.DataFrame({
    "Feature": list(X.columns),
    "Coefficient": model.coef_
})
print("\nModel Coefficients:")
print(coefficients)
