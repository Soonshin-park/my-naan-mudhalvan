import pandas as pd

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

data = load_data('accident_data.csv')
if data is not None:
    print("Data loaded successfully.")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def preprocess_data(df):
    """Preprocesses the accident data."""
    if df is None:
   return None, None, None, None
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])
    X = df.drop('accident_occurred', axis=1, errors='ignore')
    y = df['accident_occurred'] if 'accident_occurred' in df.columns else None
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X, None, None, None

if data is not None:
    X_train, X_test, y_train, y_test = preprocess_data(data.copy())
    if X_train is not None:
        print("Data preprocessed and split.")
        print(f"Training data shape: {X_train.shape}, Training target shape: {y_train.shape}")
        print(f"Testing data shape: {X_test.shape}, Testing target shape: {y_test.shape}")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(X_train, y_train):
    """Trains a Logistic Regression model."""
    if X_train is None or y_train is None:
        print("Error: Training data not available.")
        return None
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    return model

if X_train is not None and y_train is not None:
    model = train_model(X_train, y_train)
    if model:
        print("Model trained successfully.")
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model."""
    if model is None or X_test is None or y_test is None:
        print("Error: Model or testing data not available.")
        return
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on the test set: {accuracy:.4f}")

if model and X_test is not None and y_test is not None:
    evaluate_model(model, X_test, y_test)


def predict(model, new_data):
    """Makes predictions using the trained model."""
    if model is None or new_data is None:
        print("Error: Model or new data not available.")
        return None
    predictions = model.predict(new_data)
    return predictions

if model and X_test is not None:
    sample_prediction = predict(model, X_test.iloc[[0]]) # Predict on the first row of the test set
    print(f"Sample prediction: {sample_prediction}")
