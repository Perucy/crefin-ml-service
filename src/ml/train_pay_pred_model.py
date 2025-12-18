"""
Implementation of the Payment Prediction Model Training
Professional ML training pipeline with proper error handling
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestRegressor  # ← Fixed spelling!
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Import our modules
from ml.features.feature_engineering import prepare_features_for_training
from ml.data.loader import load_invoices_from_database, check_data_quality


print("=" * 60)
print("      TRAINING PAYMENT TIME PREDICTION MODEL V2")
print("=" * 60)


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n📊 Loading invoice data from database...")

invoices_df = load_invoices_from_database()

print(f"✅ Loaded {len(invoices_df)} invoices")
print(f"\nColumns: {list(invoices_df.columns)}")
print(f"\nFirst few rows:")
print(invoices_df.head())


# ============================================================================
# STEP 2: DATA QUALITY CHECK
# ============================================================================
data_quality_ok = check_data_quality(invoices_df)

if not data_quality_ok:
    print("\n⚠️  Warning: Data quality issues detected.")
    response = input("Continue training anyway? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        exit()


# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n🔧 Engineering features...")

X, y, feature_names = prepare_features_for_training(invoices_df)

print(f"\n✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")


# ============================================================================
# STEP 4: TRAIN/TEST SPLIT (WITH SAFETY CHECK)
# ============================================================================
print("\n📊 Splitting train/test...")

# Check if we have enough data to split
if len(X) < 10:
    print(f"\n⚠️  Only {len(X)} samples - too few for train/test split!")
    print("   Using all data for training (no test set)")
    print("   This is ONLY for testing the pipeline!\n")
    
    X_train = X
    X_test = X  # Use same data for "testing" (just to make code work)
    y_train = y
    y_test = y
    
    print(f"✅ Training set: {len(X_train)} samples")
    print(f"⚠️  Test set: Using same data (not a real test!)")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"✅ Training set: {len(X_train)} samples")
    print(f"✅ Testing set: {len(X_test)} samples")


# ============================================================================
# STEP 5: TRAIN MODEL (RANDOM FOREST)
# ============================================================================
print("\n🤖 Training Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,        # More trees
    max_depth=8,             # Shallower (less overfitting)
    min_samples_split=10,    # More conservative
    min_samples_leaf=5,      # Smoother predictions
    max_features='sqrt',     # Feature sampling
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# Show out-of-bag score (another validation metric)
if hasattr(model, 'oob_score_'):
    print(f"   Out-of-bag R²: {model.oob_score_:.4f}")


# ============================================================================
# STEP 6: EVALUATE PERFORMANCE
# ============================================================================
print("\n" + "=" * 60)
print("             MODEL PERFORMANCE METRICS")
print("=" * 60)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics on test set
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)

# Metrics on training set (to check overfitting)
mae_train = mean_absolute_error(y_train, y_pred_train)
r2_train = r2_score(y_train, y_pred_train)

print(f"\n📊 TEST SET PERFORMANCE:")
print(f"   Mean Absolute Error (MAE): {mae:.2f} days")
print(f"   Root Mean Squared Error (RMSE): {rmse:.2f} days")
print(f"   R² Score: {r2:.4f}")

print(f"\n📊 TRAINING SET PERFORMANCE:")
print(f"   MAE: {mae_train:.2f} days")
print(f"   R²: {r2_train:.4f}")

# Check for overfitting
if r2_train - r2 > 0.15:
    print(f"\n⚠️  WARNING: Possible overfitting detected!")
    print(f"   Training R² ({r2_train:.4f}) >> Test R² ({r2:.4f})")
else:
    print(f"\n✅ Good generalization (no overfitting)")


# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 60)
print("             FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Get feature importances
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 10 Most Important Features:")
print(importances.head(10).to_string(index=False))

print("\n📊 Bottom 5 Least Important Features:")
print(importances.tail(5).to_string(index=False))


# ============================================================================
# STEP 8: CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 60)
print("             CROSS-VALIDATION")
print("=" * 60)

# Only do cross-validation if we have enough data
if len(X) >= 20:
    cv_scores = cross_val_score(
        model, X, y, 
        cv=min(5, len(X) // 4),  # 5-fold or less if not enough data
        scoring='r2'
    )
    
    print(f"\n📊 Cross-Validation R² Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score:.4f}")
    
    print(f"\n✅ Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
else:
    print(f"\n⚠️  Not enough data for cross-validation (need 20+, have {len(X)})")


# ============================================================================
# STEP 9: SAMPLE PREDICTIONS
# ============================================================================
print("\n" + "=" * 60)
print("      SAMPLE PREDICTIONS (First 10 test samples)")
print("=" * 60)

print(f"\n{'Actual':<10} {'Predicted':<12} {'Error':<10}")
print("-" * 32)

for actual, predicted in list(zip(y_test, y_pred_test))[:10]:
    error = abs(actual - predicted)
    print(f"{actual:<10.0f} {predicted:<12.1f} {error:<10.1f}")


# ============================================================================
# STEP 10: SAVE MODEL
# ============================================================================
print("\n💾 Saving model...")

# Create models directory if it doesn't exist
os.makedirs('models/saved', exist_ok=True)

model_path = 'models/saved/payment_predictor_v2.joblib'
joblib.dump(model, model_path)

# Also save feature names and metadata (important for production!)
metadata = {
    'feature_names': feature_names,
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'model_type': 'RandomForestRegressor',
    'n_features': len(feature_names),
    'n_training_samples': len(X_train),
    'training_date': pd.Timestamp.now().isoformat()
}

joblib.dump(metadata, 'models/saved/payment_predictor_v2_metadata.joblib')

print(f"   ✅ Model saved to: {model_path}")
print(f"   ✅ Metadata saved")


# ============================================================================
# STEP 11: IMPROVEMENT SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("             IMPROVEMENT SUMMARY")
print("=" * 60)

print(f"\n📈 MODEL COMPARISON:")
print(f"   Previous R² (v1, Linear Regression): ~0.40")
print(f"   Current R² (v2, Random Forest):      {r2:.4f}")

if r2 > 0.40:
    improvement = ((r2 - 0.40) / 0.40) * 100
    print(f"   Improvement: +{improvement:.1f}%")
    
    if r2 >= 0.65:
        print(f"\n🎉 SUCCESS! Model is PRODUCTION-READY!")
        print(f"   ✅ R² ≥ 0.65 (industry standard met)")
    elif r2 >= 0.55:
        print(f"\n✅ GOOD! Model is usable but could improve")
        print(f"   Consider collecting more data or tuning hyperparameters")
    else:
        print(f"\n⚠️  Model improved but still needs work")
        print(f"   Target: R² ≥ 0.65 for production")
else:
    print(f"\n⚠️  R² did not improve significantly")
    print(f"   Possible issues:")
    print(f"   - Not enough training data")
    print(f"   - Clients don't have enough history")
    print(f"   - Need more diverse data")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)