import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training and test datasets
train_file_path = '../train.csv'  # Update with your train dataset file path
test_file_path = '../test.csv'  # Update with your test dataset file path
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# Fill NaN values for 'Age' and 'Embarked' in both datasets
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# Impute missing values for 'Fare' in the test dataset (if any)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Combine train and test data to fit LabelEncoder on all possible categories
combined_data = pd.concat([train_data, test_data], axis=0, copy=True)

# Fit LabelEncoder on combined data for 'Sex' and 'Embarked'
label_encoder_sex = LabelEncoder()
label_encoder_embarked = LabelEncoder()
combined_data['Sex'] = label_encoder_sex.fit_transform(combined_data['Sex'])
combined_data['Embarked'] = label_encoder_embarked.fit_transform(combined_data['Embarked'])

# Split the data back into train and test sets
train_data = combined_data.iloc[:len(train_data)].copy()
test_data = combined_data.iloc[len(train_data):].copy()

# Drop non-numeric or non-useful columns
train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Ensure there are no NaN values in training and test data
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(train_data.drop('Survived', axis=1))
y_train = train_data['Survived'].copy()

# Drop 'Survived' and 'PassengerId' from the test set before transforming
X_test = imputer.transform(test_data.drop(['PassengerId', 'Survived'], axis=1, errors='ignore'))

# Initialize and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Load the gender_submission.csv file for evaluation (if available)
submission_file_path = '../gender_submission.csv'  # Update with your actual submission file path
submission_data = pd.read_csv(submission_file_path)

# Merge predictions with actual results based on 'PassengerId'
predictions_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Predicted_Survived': y_pred})

# Evaluate the model (assuming you have actual results in submission_data)
if 'Survived' in submission_data.columns:
    predictions_df = predictions_df.merge(submission_data[['PassengerId', 'Survived']], on='PassengerId')
    print("Accuracy Score:", accuracy_score(predictions_df['Survived'], predictions_df['Predicted_Survived']))
    print("\nClassification Report:\n", classification_report(predictions_df['Survived'], predictions_df['Predicted_Survived']))
    print("\nConfusion Matrix:\n", confusion_matrix(predictions_df['Survived'], predictions_df['Predicted_Survived']))

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(predictions_df['Survived'], predictions_df['Predicted_Survived']),
                annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

