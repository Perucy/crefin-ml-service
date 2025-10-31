"""
    Generating mock training data for payment time predictions
    Simulates historical client payment data
"""

import pandas as pd
import numpy as np

#set random seed for reproducibility
np.random.seed(42)

#number of samples to consider
n_samples = 200

data = {
    'client_id': np.random.choice(['client_1', 'client_2', 'client_3', 'client_4', 'client_5'], n_samples),
    'invoice_amount': np.random.uniform(500, 5000, n_samples),
    'day_of_week': np.random.randint(0, 7, n_samples),
    'month': np.random.randint(1, 13, n_samples), 
    'client_reliability_score': np.random.uniform(0.3, 1.0, n_samples),
}

#create dataframe
df = pd.DataFrame(data)

# available columns
print(f" Available columns: {list(df.columns)}")
# Generate target: payment_days (what we want to predict)
# Formula: Base days + random variation + client effect + amount effect
df['payment_days'] = (
    15 +  # Base: 15 days average
    np.random.randint(-5, 10, n_samples) +  # Random variation
    (1 - df['client_reliability_score']) * 20 +  # Unreliable clients take longer
    (df['invoice_amount'] / 1000) * 2  # Bigger invoices take slightly longer
).astype(int)

# Make sure payment days is positive and realistic
df['payment_days'] = df['payment_days'].clip(5, 60)

#save to csv
df.to_csv('data/training_data.csv', index=False)

print("✅ Mock training data generated!")
print(f"📊 Total samples: {len(df)}")
print(f"📈 Payment days range: {df['payment_days'].min()} - {df['payment_days'].max()}")
print(f"📊 Average payment days: {df['payment_days'].mean():.1f}")
print("\nFirst 5 rows:")
print(df.head())
