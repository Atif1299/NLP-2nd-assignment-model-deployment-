"""
This script trains and evaluates an SVC model on the Car Evaluation dataset
using simple label encoding for preprocessing.
"""
import pandas as pd
import numpy as np
import pickle
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score

# --- Step 1: Fetch Dataset ---
print("Fetching Car Evaluation dataset...")
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets

# Combine features and target for easier processing
data = pd.concat([X, y], axis=1)
print("Original Data Head:")
print(data.head())
print("\n" + "="*30 + "\n")

# --- Step 2: Pre-process and Encode the Data ---
print("Encoding categorical data using LabelEncoder...")
encoded_data = pd.DataFrame()
label_encoders = {}

# Encode all features and the target variable
for column in data.columns:
    le = LabelEncoder()
    encoded_data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

print("Encoded Data Head:")
print(encoded_data.head())
print("\n" + "="*30 + "\n")

# --- Step 3: Split Data into Training and Testing Sets ---
print("Splitting data into training (80%) and testing (20%) sets...")
features = encoded_data.drop('class', axis=1)
target = encoded_data['class']

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    shuffle=True
)
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")
print("\n" + "="*30 + "\n")


# --- Step 4: Train the Support Vector Classifier ---
print("Training the Support Vector Classifier...")
svc_model = svm.SVC(gamma='auto', random_state=42)
svc_model.fit(X_train, y_train)
print("Model training complete.")
print(svc_model)
print("\n" + "="*30 + "\n")


# --- Step 5: Save the Trained Model ---
print("Saving the trained model to 'car_svc_model.pkl'...")
with open('car_svc_model.pkl', 'wb') as model_file:
    pickle.dump(svc_model, model_file)
print("Model saved successfully.")
print("\n" + "="*30 + "\n")


# --- Step 6: Evaluate the Model ---
print("Evaluating the model on the test data...")
with open('car_svc_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

y_pred = loaded_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
