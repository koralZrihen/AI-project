import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
training_data = pd.read_csv("C:\\Users\\kor12\\Desktop\\Training2.csv")
testing_data = pd.read_csv("C:\\Users\\kor12\\Desktop\\Testing2.csv")

# Impute missing values before feature selection
imputer = SimpleImputer(strategy='median')
for dataset in [training_data, testing_data]:
    dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = imputer.fit_transform(dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])

# Encoding the categorical outcome variable ('Outcome') using get_dummies
training_data = pd.get_dummies(training_data, columns=['Outcome'], drop_first=True)
testing_data = pd.get_dummies(testing_data, columns=['Outcome'], drop_first=True)

# Prepare X and y for feature selection
X_train = training_data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
X_test = testing_data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

# Determine the target variable ('Outcome_YES' or 'Outcome_1')
if 'Outcome_YES' in training_data.columns:
    y_train = training_data['Outcome_YES']
else:
    raise ValueError("Neither 'Outcome_YES' nor 'Outcome_1' found in training data columns.")

if 'Outcome_YES' in testing_data.columns:
    y_test = testing_data['Outcome_YES']
else:
    raise ValueError("Neither 'Outcome_YES' nor 'Outcome_1' found in testing data columns.")

# Apply SelectKBest for feature selection
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X_train, y_train)
selected_features = X_train.columns[selector.get_support()]
print("Selected features:", selected_features)

# Standardizing features for PCA
X_scaled = StandardScaler().fit_transform(X_new)

# Performing PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_scaled)

# Creating a DataFrame for PCA results visualization
principalDf = pd.DataFrame(data=principalComponents, columns=['Principal Component 1', 'Principal Component 2'])

# Visualizing the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(principalDf['Principal Component 1'], principalDf['Principal Component 2'], s=50)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Dataset')
plt.show()
