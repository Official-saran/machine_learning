1.  Analyzing Student Performance:  
Scenario: You are analyzing factors that affect student performance in a standardized test. You collect data on study hours, attendance rates, and socioeconomic background.  
Question: How would you set up your linear regression model? What considerations would you make regarding the interpretation of coefficients in this context?


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate random data for student performance
np.random.seed(42)

# Number of data points
num_students = 100

# Generate data
study_hours = np.random.uniform(1, 10, num_students)  # Study hours between 1 and 10
attendance_rate = np.random.uniform(50, 100, num_students)  # Attendance rates between 50% and 100%
socioeconomic_index = np.random.uniform(1, 5, num_students)  # Socioeconomic background index (1 to 5)
test_scores = (
    20 + (5 * study_hours) + (0.8 * attendance_rate) + (10 * socioeconomic_index) + np.random.normal(0, 5, num_students)
)  # Test scores with noise

# Create DataFrame
data = pd.DataFrame({
    "Study_Hours": study_hours,
    "Attendance_Rate": attendance_rate,
    "Socioeconomic_Index": socioeconomic_index,
    "Test_Scores": test_scores
})

# Save to CSV
csv_path = 'student_performance_data.csv'
data.to_csv(csv_path, index=False)

# Step 2: Load data and prepare for model
data = pd.read_csv(csv_path)

# Define features and target variable
X = data[["Study_Hours", "Attendance_Rate", "Socioeconomic_Index"]]
y = data["Test_Scores"]

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
print("Linear Regression Model Results")
print("--------------------------------")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Intercept: {intercept:.2f}")
print("Coefficients:")
print(f"  Study Hours: {coefficients[0]:.2f}")
print(f"  Attendance Rate: {coefficients[1]:.2f}")
print(f"  Socioeconomic Index: {coefficients[2]:.2f}")
