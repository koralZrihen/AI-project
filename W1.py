import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA

# Load the data
testing_data = pd.read_csv("C:\\Users\\kor12\\Desktop\\Testing.csv")
training_data = pd.read_csv("C:\\Users\\kor12\\Desktop\\Training.csv")

# Separate features and target variable
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X_train = training_data[features]
y_train = training_data['Outcome']

X_test = testing_data[features]
y_test = testing_data['Outcome']

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = rf.feature_importances_

# Select features based on importance threshold
threshold = 0.05  # Adjust threshold as needed
selected_features_rf = [feature for i, feature in enumerate(features) if feature_importances[i] > threshold]

# Train a new Random Forest model with selected features
X_train_selected_rf = X_train[selected_features_rf]
X_test_selected_rf = X_test[selected_features_rf]
rf_selected = RandomForestClassifier(random_state=42)
rf_selected.fit(X_train_selected_rf, y_train)

# Predict on the test set using Random Forest
y_pred_selected_rf = rf_selected.predict(X_test_selected_rf)

# Calculate accuracy for Random Forest
accuracy_selected_rf = accuracy_score(y_test, y_pred_selected_rf)
print("Accuracy with selected features (Random Forest):", accuracy_selected_rf)


# Feature Selection using SelectKBest
selector_kbest = SelectKBest(score_func=f_classif, k=5)  # any k
X_train_selected_kbest = selector_kbest.fit_transform(X_train, y_train)
X_test_selected_kbest = selector_kbest.transform(X_test)

# Train a new Random Forest model with selected features from SelectKBest
rf_kbest = RandomForestClassifier(random_state=42)
rf_kbest.fit(X_train_selected_kbest, y_train)

# Predict on the test set using SelectKBest
y_pred_selected_kbest = rf_kbest.predict(X_test_selected_kbest)

# Calculate accuracy for SelectKBest
accuracy_selected_kbest = accuracy_score(y_test, y_pred_selected_kbest)
print("Accuracy with selected features (SelectKBest):", accuracy_selected_kbest)


# Feature Selection using PCA
pca = PCA(n_components=5)  # any n 
X_train_selected_pca = pca.fit_transform(X_train)
X_test_selected_pca = pca.transform(X_test)

# Train a new Random Forest model with selected features from PCA
rf_pca = RandomForestClassifier(random_state=42)
rf_pca.fit(X_train_selected_pca, y_train)

# Predict on the test set using PCA
y_pred_selected_pca = rf_pca.predict(X_test_selected_pca)

# Calculate accuracy for PCA
accuracy_selected_pca = accuracy_score(y_test, y_pred_selected_pca)
print("Accuracy with selected features (PCA):", accuracy_selected_pca)


# Feature Selection using Recursive Feature Elimination (RFE)
rfe_selector = RFE(estimator=rf, n_features_to_select=5, step=1)
X_train_selected_rfe = rfe_selector.fit_transform(X_train, y_train)
X_test_selected_rfe = rfe_selector.transform(X_test)

# Train a new Random Forest model with selected features from RFE
rf_rfe = RandomForestClassifier(random_state=42)
rf_rfe.fit(X_train_selected_rfe, y_train)

# Predict on the test set using RFE
y_pred_selected_rfe = rf_rfe.predict(X_test_selected_rfe)

# Calculate accuracy for RFE
accuracy_selected_rfe = accuracy_score(y_test, y_pred_selected_rfe)
print("Accuracy with selected features (RFE):", accuracy_selected_rfe)
