6. Image Classification**  
  Scenario: You are developing an application that classifies images of animals (e.g., cats vs. dogs) using K-NN.  
Question: Given the high dimensionality of image data, what techniques could you use to optimize the performance of K-NN? How would you measure the accuracy of your model?

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the generated CSV file
csv_path = 'image_classification_data.csv'
data = pd.read_csv(csv_path)

# Features and labels
X = data.drop('Label', axis=1)
y = data['Label']

# Normalize the pixel data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=50)  # Reduce to 50 principal components
X_pca = pca.fit_transform(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Train K-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict on test set
y_pred = knn.predict(X_test)

# Measure accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation to assess model performance
cv_scores = cross_val_score(knn, X_pca, y, cv=5, scoring='accuracy')
print("\nCross-validated Accuracy Scores:", cv_scores)
print("Mean Cross-validated Accuracy:", np.mean(cv_scores))
