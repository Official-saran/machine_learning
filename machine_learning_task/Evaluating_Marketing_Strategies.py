3. Evaluating Marketing Strategies  
Scenario: A marketing team has created a decision tree to evaluate different advertising strategies for a new product launch.  
Question: How would you assess the effectiveness of the current decision tree structure? What specific metrics or outcomes would you analyze to determine if any adjustments are necessary?


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Step 1: Load the generated CSV data
csv_path = "marketing_strategy_data.csv"
data = pd.read_csv(csv_path)

# Step 2: Prepare the data for modeling
X = data[["Budget", "Channel_Effectiveness", "Market_Competition", "Target_Audience_Size"]]
y = data["Effectiveness"]

# Encode the target variable ("Effective" -> 1, "Ineffective" -> 0)
y = y.map({"Effective": 1, "Ineffective": 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42, max_depth=3)
decision_tree.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = decision_tree.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Step 5: Display results
print("Decision Tree Model Performance")
print("--------------------------------")
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(report)

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, feature_names=X.columns, class_names=["Ineffective", "Effective"], filled=True)
plt.title("Decision Tree for Marketing Strategies")
plt.show()

# Display the tree structure as text
tree_rules = export_text(decision_tree, feature_names=list(X.columns))
print("\nDecision Tree Rules:")
print(tree_rules)
