# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
# Import necessary libraries for evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_graphviz
import pandas as pd

# Load the training dataset
train_file_path = 'E:/one drive/OneDrive - Happy English/course subjects/Machine Learning CSC354/project/Supervised-ML-algorithms/titanic working/train.csv'
train_df = pd.read_csv(train_file_path)

# Load the testing dataset
test_file_path = 'E:/one drive/OneDrive - Happy English/course subjects/Machine Learning CSC354/project/Supervised-ML-algorithms/titanic working/test.csv'  # Update this path with your actual test file path if needed
test_df = pd.read_csv(test_file_path)

# Data preprocessing for training set
# Fill missing age values with the median
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill missing embarked values with the most common port
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' due to too many missing values
train_df.drop('Cabin', axis=1, inplace=True)

# Encode categorical variables ('gender' and 'Embarked')
label_encoder = LabelEncoder()
train_df['gender'] = label_encoder.fit_transform(train_df['gender'])
train_df['Embarked'] = label_encoder.fit_transform(train_df['Embarked'])

# Features and target for training
features = ['PassengerId', 'Pclass', 'gender', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_train = train_df[features]
y_train = train_df['Survived']

# Initialize and train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Data preprocessing for the test set
# Fill missing age values with the median from the training set
test_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill missing embarked values with the most common port from the training set
test_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Drop 'Cabin' as done with the training set
test_df.drop('Cabin', axis=1, inplace=True)

# Encode categorical variables in the test set
test_df['gender'] = label_encoder.fit_transform(test_df['gender'])
test_df['Embarked'] = label_encoder.fit_transform(test_df['Embarked'])

# Select features for the test set
X_test = test_df[features]

# Predict on the test set
y_pred = clf.predict(X_test)

# Create a DataFrame for the output
output = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_pred
})

# Display the prediction results
print(output)

# Save the output to a CSV file
output.to_csv('titanic_predictions.csv', index=False)

# Load the actual results from 'gender_submission.csv'
actual_results_file_path = 'E:/one drive/OneDrive - Happy English/course subjects/Machine Learning CSC354/project/Supervised-ML-algorithms/titanic working/gender_submission.csv'  # Update this path if necessary
actual_results_df = pd.read_csv(actual_results_file_path)

# Ensure that the 'PassengerId' is used to align predictions and actual results
y_test = actual_results_df.set_index('PassengerId')['Survived'].reindex(X_test['PassengerId']).values

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Generate the classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Export the decision tree to a .dot file for visualization
export_graphviz(clf,
                out_file="tree.dot",
                feature_names=features,
                class_names=['0', '1'],
                filled=True)

# Note: You can use tools like Graphviz or online viewers to visualize the 'tree.dot' file.

# Print the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Export the decision tree to a .dot file for visualization
export_graphviz(clf,
                out_file="tree.dot",
                feature_names=features,
                class_names=['0', '1'],
                filled=True)

# Note: You can use tools like Graphviz or online viewers to visualize the 'tree.dot' file.
