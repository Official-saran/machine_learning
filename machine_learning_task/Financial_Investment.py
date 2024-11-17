4. Financial Investment Decisions  
Scenario: An investment firm uses a decision tree to analyze potential investment opportunities based on market trends and economic indicators.  
Question: What process would you follow to update the decision tree as market conditions change? How would you visualize these changes for clarity among team members?


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the generated CSV data
csv_path = "financial_investment_data.csv"
data = pd.read_csv(csv_path)

# Prepare the data for modeling
X = data[["Economic_Growth", "Market_Volatility", "Sector_Performance", "Investment_Cost"]]
y = data["Investment_Decision"]

# Encode target variable: 'Invest' -> 1, 'Do Not Invest' -> 0
y = y.map({"Invest": 1, "Do Not Invest": 0})

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the initial decision tree
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=4)
decision_tree.fit(X_train, y_train)

# Step 4: Evaluate the initial tree
y_pred = decision_tree.predict(X_test)

print("Initial Model Performance")
print("-------------------------")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the initial tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, feature_names=X.columns, class_names=["Do Not Invest", "Invest"], filled=True)
plt.title("Initial Decision Tree")
plt.show()

# Step 5: Re-train the tree with updated data (simulate new data scenario)
# Adding a small noise to simulate changes in market conditions
X_updated = X + np.random.normal(0, 0.5, X.shape)
X_train_updated, X_test_updated, y_train_updated, y_test_updated = train_test_split(X_updated, y, test_size=0.2, random_state=42)

decision_tree_updated = DecisionTreeClassifier(random_state=42, max_depth=4)
decision_tree_updated.fit(X_train_updated, y_train_updated)

# Step 6: Evaluate the updated tree
y_pred_updated = decision_tree_updated.predict(X_test_updated)

print("Updated Model Performance")
print("-------------------------")
print(f"Accuracy: {accuracy_score(y_test_updated, y_pred_updated):.2f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_updated, y_pred_updated))
print("\nClassification Report:")
print(classification_report(y_test_updated, y_pred_updated))

# Visualize the updated tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree_updated, feature_names=X.columns, class_names=["Do Not Invest", "Invest"], filled=True)
plt.title("Updated Decision Tree")
plt.show()

# Step 7: Visualize Feature Importance
feature_importance = decision_tree_updated.feature_importances_
plt.bar(X.columns, feature_importance)
plt.title("Feature Importance in Updated Model")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
