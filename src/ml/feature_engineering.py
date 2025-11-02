"""
    Feature engineering for payment prediction
    Extracts client history features from invoice data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_client_history(invoices_df, reference_date_col='issue_date'):
    """
        For each invoice calculate features based on client's past behavior

        Args: 
            invoices_df: df with cols [client_id, issue_date, payment_days, amount]
            reference_date_col: col to use as "current" date (usually issue date)

        Returns:
            df with engineered client history features
    """

    # sort by date to ensure chronological order
    df = invoices_df.sort_values(reference_date_col).reset_index(drop=True)

    # will store client features for each invoice
    client_features = []

    print(f"📊 Engineering features for {len(df)} invoices...")

    for idx, current_invoice in df.iterrows():
        client_id = current_invoice['client_id']
        current_date = current_invoice[reference_date_col]

        # get all past invoices of this client (before current invoice)
        past_invoices = df[
            (df['client_id'] == client_id) &
            (df[reference_date_col] < current_date)
        ]

        if len(past_invoices) >= 1:
            # client has history - calc features
            features = {
                #avg payment time
                'client_avg_payment_days': past_invoices['payment_days'].mean(),

                # payment consistency (lower = more consistent)
                'client_payment_std': past_invoices['payment_days'].std() if len(past_invoices) > 1 else 0,

                # late payment rate (% of invoices paid late)
                'client_late_payment_rate': (past_invoices['payment_days'] > 30).mean(),

                # total experience with this client
                'client_total_invoices': len(past_invoices),

                # recency: days since last invoice
                'days_since_last_invoice': (current_date - past_invoices[reference_date_col].max()).days,

                # amount patterns
                'client_avg_amount': past_invoices['amount'].mean(),
                'client_max_amount': past_invoices['amount'].max(),

                # is this client improving or getting worse
                'client_payment_trend': calculate_payment_trend(past_invoices),
            }
        else:
            # new client - no history available
            # use neutral/default values
            features = {
                'client_avg_payment_days': 30.0,  # Assume industry average
                'client_payment_std': 5.0,         # Assume moderate variance
                'client_late_payment_rate': 0.5,   # Assume 50% late (cautious)
                'client_total_invoices': 0,        # Flag as new
                'days_since_last_invoice': 999,    # No previous invoice
                'client_avg_amount': current_invoice['amount'],  # Use current
                'client_max_amount': current_invoice['amount'],
                'client_payment_trend': 0,         # Neutral
            }
        
        client_features.append(features)

        # progress indicator
        if (idx + 1) % 50 == 0:
            print(f"  Proceed {idx + 1}/{len(df)} invoices...")

    features_df = pd.DataFrame(client_features)

    print(f"✅ Generated {len(features_df.columns)} client history features")
    print(f"   Features: {list(features_df.columns)}")
    
    return features_df

def calculate_payment_trend(past_invoices, window=3):
    """
        Calculates if client is paying faster or slower over time

        Returns:
            positive number = getting worse (slower payments)
            negative number = getting better (faster payments)
            0 = stable
    """
    if len(past_invoices) < 2:
        return 0
    
    # compare recent vs older payments
    sorted_invoices = past_invoices.sort_values('issue_date')

    if len(sorted_invoices) >= window * 2:
        recent = sorted_invoices.tail(window)['payment_days'].mean()
        older = sorted_invoices.tail(window)['payment_days'].mean()
        trend = recent - older
    else:
        # too few data points - use simpler linear trend
        payment_days = sorted_invoices['payment_days'].values
        trend = payment_days[-1] - payment_days[0]

    return trend

def add_temporal_features(invoices_df, date_col='issue_date'):
    """
        Add time-based features (day of week, month, end-of-month etc.)
    """
    df = invoices_df.copy()

    # extract datetime features
    df['day_of_week'] = df[date_col].dt.dayofweek  #0=Monday, 6=sunday
    df['day_of_month'] = df[date_col].dt.day
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter

    # business timing patterns
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Month patterns
    df['is_start_of_month'] = (df['day_of_month'] <= 5).astype(int)
    df['is_end_of_month'] = (df['day_of_month'] >= 25).astype(int)
    df['week_of_month'] = ((df['day_of_month'] - 1) // 7 + 1)

    # Quarterly patterns
    df['is_quarter_end'] = df['month'].isin([3, 6, 9, 12]).astype(int)

    # Seasonal patterns
    df['is_holiday_season'] = df['month'].isin([11, 12]).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

    print(f"✅ Added {13} temporal features")
    
    return df
