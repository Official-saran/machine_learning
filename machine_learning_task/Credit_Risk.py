
7. Credit Risk Assessment  
Scenario: A bank uses logistic regression to determine the probability of a loan applicant defaulting on their loan based on their credit score, income level, and employment status. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the data
csv_path = 'credit_risk_data.csv'
data = pd.read_csv(csv_path)

# Features and target
X = data[["Credit_Score", "Income", "Employment_Status"]]
y = data["Loan_Default"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Get coefficients and feature names
coefficients = model.coef_[0]
features = X.columns

# Display the coefficients
print("Logistic Regression Coefficients:")
for feature, coef in zip(features, coefficients):
    print(f"{feature}: {coef:.4f}")

# Interpreting coefficients
print("\nCoefficient Interpretation:")
for feature, coef in zip(features, coefficients):
    if coef > 0:
        print(f"- {feature} has a positive impact on the likelihood of default (default risk increases as {feature} increases).")
    else:
        print(f"- {feature} has a negative impact on the likelihood of default (default risk decreases as {feature} increases).")
