# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Load the Titanic dataset
titanic_data = pd.read_csv('titanic.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')  # You can choose a different strategy based on your data
titanic_data[['Age', 'Fare']] = imputer.fit_transform(titanic_data[['Age', 'Fare']])

# Preprocess the data (encode categorical variables, feature engineering, etc.)

# Select features and target variable
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
target = 'Survived'

X = titanic_data[features]
y = titanic_data[target]

# Encode categorical variables
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

# Print model evaluation metrics
print(f'Accuracy: {accuracy}')
print('Classification Report:\n', report)

# Feature importance plot
feature_importance = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance, y=feature_names)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

# Confusion Matrix
conf_matrix = pd.crosstab(y_test, predictions, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
