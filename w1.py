import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics  

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
selected_features = [feature for i, feature in enumerate(features) if feature_importances[i] > threshold]

# Train a new Random Forest model with selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
rf_selected = RandomForestClassifier(random_state=42)
rf_selected.fit(X_train_selected, y_train)

# Predict on the test set
y_pred_selected = rf_selected.predict(X_test_selected)

# Calculate accuracy
accuracy_selected = accuracy_score(y_test, y_pred_selected)
print("Selected Features:", selected_features)
print("Accuracy with selected features:", accuracy_selected)
