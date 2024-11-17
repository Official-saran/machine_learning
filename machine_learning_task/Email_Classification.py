8. Email Classification  
Scenario: You are tasked with building a spam filter using SVM to classify emails as either "spam" or "not spam."  
Question: How would you decide between using a linear SVM and a non-linear SVM for this classification problem? What features would you consider important for your model?


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

# Step 1: Load dataset (example data structure)
emails = pd.DataFrame({
    "Email_Text": ["Congratulations, you won a prize!", "Meeting agenda for tomorrow", "..."],
    "Label": ["spam", "not spam", "..."]
})

# Step 2: Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(emails["Email_Text"]).toarray()
y = emails["Label"].apply(lambda x: 1 if x == "spam" else 0)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train linear SVM
linear_svm = SVC(kernel="linear", C=1)
linear_svm.fit(X_train, y_train)

# Step 5: Train non-linear SVM (RBF)
rbf_svm = SVC(kernel="rbf", C=1, gamma="scale")
rbf_svm.fit(X_train, y_train)

# Step 6: Evaluate both models
for name, model in [("Linear SVM", linear_svm), ("Non-linear SVM (RBF)", rbf_svm)]:
    y_pred = model.predict(X_test)
    print(f"{name} Results:")
    print(classification_report(y_test, y_pred))
    if hasattr(model, "decision_function"):
        roc_auc = roc_auc_score(y_test, model.decision_function(X_test))
        print(f"ROC-AUC: {roc_auc:.2f}\n")
