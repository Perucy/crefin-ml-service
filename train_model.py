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

# ========================================================================================================
# LOAD DATA
# ========================================================================================================
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

# mean absolute error: avg error in days
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.2f} days")
print(f"On average, predictions are off by {mae:.2f} days")

# root mean squared error: penalize larger errors more
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} days")
print(f"Standard deviation of errors")

# R² Score: How much variance is explained (0-1, higher is better)
r2 = r2_score(y_test, y_pred)
print(f"\n🎯 R² Score: {r2:.4f}")
print(f"   → Model explains {r2*100:.2f}% of the variance")
if r2 > 0.7:
    print("   → 🟢 GOOD model!")
elif r2 > 0.5:
    print("   → 🟡 DECENT model")
else:
    print("   → 🔴 NEEDS IMPROVEMENT")

# Show some example predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (First 10 test samples)")
print("=" * 60)
print(f"{'Actual':<10} {'Predicted':<10} {'Error':<10}")
print("-" * 30)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error = abs(actual - predicted)
    print(f"{actual:<10.0f} {predicted:<10.1f} {error:<10.1f}")

# Save the model
os.makedirs('models/saved', exist_ok=True)
model_path = 'models/saved/payment_predictor_v1.joblib'
joblib.dump({
    'model': model,
    'feature_columns': feature_columns,
    'client_mapping': client_mapping,
    'metrics': {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
}, model_path)
print(f"   ✅ Model saved to: {model_path}")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
