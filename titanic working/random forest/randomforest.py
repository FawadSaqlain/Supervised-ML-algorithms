# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_graphviz

# Load the training dataset
train_file_path = '../train.csv'
train_data = pd.read_csv(train_file_path)

# Load the testing dataset (without target column) and actual results
test_file_path = '../test.csv'  # Replace with the correct path if needed
test_data = pd.read_csv(test_file_path)

# Load the actual results for the test data
gender_submission_file_path = '../gender_submission.csv'  # Replace with the correct path if needed
actual_results = pd.read_csv(gender_submission_file_path)

# Preprocess the training data
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Cabin'].fillna('Unknown', inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Initialize and fit LabelEncoders for each column needing encoding
label_encoders = {}
for column in ['Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])
    label_encoders[column] = le

# Select features and target variable for training
X_train = train_data[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
y_train = train_data['Survived']

# Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Preprocess the test data (similar preprocessing steps as training data)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Cabin'].fillna('Unknown', inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# Transform the test data using the fitted LabelEncoders
for column in ['Sex', 'Cabin', 'Embarked', 'Name', 'Ticket']:
    if column in label_encoders:
        le = label_encoders[column]
        # Use .fit_transform on training and .transform on test, handling unseen labels safely
        test_data[column] = test_data[column].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

X_test = test_data[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the model using the actual results
y_test = actual_results['Survived']

# Print evaluation metrics
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Export the decision tree from the random forest (for one tree)
export_graphviz(rf_classifier.estimators_[0],
                out_file="tree.dot",
                feature_names=X_train.columns,
                class_names=['0', '1'],
                filled=True)


# Export the decision tree from the random forest (for one tree)
export_graphviz(rf_classifier.estimators_[99],
                out_file="tree_99.dot",
                feature_names=X_train.columns,
                class_names=['0', '1'],
                filled=True)