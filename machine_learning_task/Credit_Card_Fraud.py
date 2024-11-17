12. Predicting Credit Card Fraud
Scenario: A financial institution is using logistic regression to identify fraudulent transactions based on transaction amount, location, and user behavior.  
Question: In setting up your data pipeline, how would you ensure that the model is trained on balanced classes? What techniques could you employ to address class imbalance?

import pandas as pd
# Load the dataset
file_path = "credit_card_fraud_data.csv"
fraud_data = pd.read_csv(file_path)

# Check class distribution
class_distribution = fraud_data['FraudulentTransaction'].value_counts()
print(f"Class distribution:\n{class_distribution}")
from sklearn.utils import resample

# Separate majority and minority classes
non_fraudulent = fraud_data[fraud_data['FraudulentTransaction'] == 0]
fraudulent = fraud_data[fraud_data['FraudulentTransaction'] == 1]

# Oversample minority class (fraudulent transactions)
fraudulent_oversampled = resample(fraudulent, 
                                  replace=True, 
                                  n_samples=len(non_fraudulent), 
                                  random_state=42)

# Combine the majority class with the oversampled minority class
balanced_data = pd.concat([non_fraudulent, fraudulent_oversampled])

# Check new class distribution
print(f"Balanced class distribution:\n{balanced_data['FraudulentTransaction'].value_counts()}")
# Undersample majority class (non-fraudulent transactions)
non_fraudulent_undersampled = resample(non_fraudulent, 
                                       replace=False, 
                                       n_samples=len(fraudulent), 
                                       random_state=42)

# Combine the undersampled majority class with the minority class
balanced_data_undersampled = pd.concat([non_fraudulent_undersampled, fraudulent])

# Check new class distribution
print(f"Balanced class distribution (Undersampling):\n{balanced_data_undersampled['FraudulentTransaction'].value_counts()}")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Prepare the data (features and target)
X = fraud_data[['TransactionAmount', 'Location', 'UserBehavior']]
y = fraud_data['FraudulentTransaction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Logistic Regression model with class weights
model = LogisticRegression(class_weight='balanced', random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate performance (using precision, recall, and F1-score)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
