5. Disease Diagnosis  
Scenario: A healthcare provider uses patient data (symptoms, age, medical history) to diagnose diseases using K-NN.  
Question: What considerations should you take into account when choosing the value of $$ K $$? How would you ensure that the model is robust against overfitting?


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the generated CSV file
csv_path = 'disease_diagnosis_data.csv'
data = pd.read_csv(csv_path)

# Features and target variable
X = data[["Age", "Fever", "Cough", "Fatigue", "Breathing_Difficulty", "Medical_History"]]
y = data["Disease_Diagnosis"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the features to prevent dominance of certain variables in distance calculation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Find the optimal value of K using cross-validation
k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(np.mean(scores))

# Plot the cross-validation scores to visualize the optimal K
plt.plot(k_range, cv_scores)
plt.xlabel('Value of K')
plt.ylabel('Cross-validated Accuracy')
plt.title('Choosing the Optimal K for K-NN')
plt.show()

# The optimal K is the one with the highest cross-validation score
optimal_k = k_range[np.argmax(cv_scores)]
print(f"Optimal K: {optimal_k}")

# Train the final K-NN model with the optimal K and evaluate
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train_scaled, y_train)
accuracy = final_knn.score(X_test_scaled, y_test)

print(f"Model Accuracy with K={optimal_k}: {accuracy:.2f}")
