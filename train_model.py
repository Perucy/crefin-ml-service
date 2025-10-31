"""
    Training a linear regression model to predict payment time
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os

print("=" * 60)
print("            TRAINING PAYMENT TIME PREDICTION MODEL")
print("=" * 60)

print("Loading training data.........")
df = pd.read_csv('data/training_data.csv')
print(f"✅ Loaded {len(df)} invoices ")
print(f" Columns: {list(df.columns)}")

#features (X) and target (Y)
#Features: what we use to predict
#        : invoice_amount, day_of_week, month, client_reliability_score

# mapping for client names to numbers
client_mapping = {client: idx for idx, client in enumerate(df['client_id'].unique())}
df['client_id_numeric'] = df['client_id'].map(client_mapping)

feature_columns = ['invoice_amount', 'day_of_week', 'month', 'client_reliability_score', 'client_id_numeric']
X = df[feature_columns]

# target: what we want to predict
y = df['payment_days']

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f" Features we're using: {feature_columns}")

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Testing set: {len(X_test)} samples")

# create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model
print("\n" + "=" * 60)
print("             MODEL PERFORMANCE METRICS")
print("=" * 60)