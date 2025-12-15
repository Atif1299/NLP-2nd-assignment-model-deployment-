"""
This script trains an SVC model on the Car Evaluation dataset using an
NLP-inspired Bag-of-Words approach (implemented via One-Hot Encoding)
for feature preprocessing.
"""
import pandas as pd
import numpy as np
import pickle
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.metrics import accuracy_score

# --- Step 1: Fetch Dataset ---
print("Fetching Car Evaluation dataset...")
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets

print("Original Features Head:")
print(X.head())
print("\n" + "="*50 + "\n")

# --- Step 2: NLP-Inspired Feature Engineering (One-Hot Encoding) ---
print("Applying 'Bag-of-Words' concept using One-Hot Encoder...")

# Preprocessor to apply OneHotEncoder to all categorical features.
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), X.columns)],
    remainder='passthrough'
)

# Encode the target variable.
le = LabelEncoder()
y_encoded = le.fit_transform(np.ravel(y))
print(f"Target classes mapping: {list(le.classes_)} -> {list(range(len(le.classes_)))}")


# --- Step 3: Split Data into Training and Testing Sets ---
print("\nSplitting data into training (80%) and testing (20%) sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, shuffle=True
)
print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")


# --- Step 4: Create and Train the SVM Pipeline ---
print("\nCreating and training the SVM pipeline...")
# The pipeline bundles preprocessing and the classifier.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm.SVC(gamma='auto', random_state=42))
])
pipeline.fit(X_train, y_train)
print("Model training complete.")


# --- Step 5: Save the Trained Pipeline ---
print("\nSaving the trained pipeline to 'car_nlp_pipeline.pkl'...")
with open('car_nlp_pipeline.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)
print("Pipeline saved successfully.")


# --- Step 6: Evaluate the Model ---
print("\nEvaluating the pipeline on the test data...")
with open('car_nlp_pipeline.pkl', 'rb') as model_file:
    loaded_pipeline = pickle.load(model_file)

y_pred = loaded_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy with NLP-style preprocessing: {accuracy:.4f}")
