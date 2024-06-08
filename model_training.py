import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

df = pd.read_csv('students_data.csv')

df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

X = df.drop('drug_addiction', axis=1)
y = df['drug_addiction']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1 Score: {f1_score(y_test, y_pred)}')

joblib_file = "best_model.pkl"
joblib.dump(model, joblib_file)
print(f"Model saved to {joblib_file}")

scaler_file = "scaler.pkl"
joblib.dump(scaler, scaler_file)
print(f"Scaler saved to {scaler_file}")
