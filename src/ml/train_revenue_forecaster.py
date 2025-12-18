"""
Training script for Revenue Forecaster
Tests the model with actual invoice data
"""
import sys
import os
from pathlib import Path

# add parent directory to path so we can import our models
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
from datetime import datetime, timedelta
import random
import joblib
from models.revenue_forecaster import RevenueForecaster

def generate_sample_data():
    """
    Generate sample invoice data for testing
    Replace this with actual database query later
    """
    print("\n📦 Generating sample invoice data...")

    # simulate 12 months of invoice history
    start_date = datetime.now() - timedelta(days=365)
    invoices = []

    base_revenue = 5000
    growth_rate = 300

    for month in range(12):
        # date for this month
        invoice_date = start_date + timedelta(days=30 * month)

        # number of invoices this month (3-7)
        num_invoices = random.randint(3, 7)

        # monthly revenue (with growth + some randomness)
        monthly_target = base_revenue + (growth_rate * month) + random.randint(-500, 500)

        # split monthly revenue across invoices 
        for i in range(num_invoices):
            amount = monthly_target / num_invoices + random.randint(-200, 200)
            payment_days = random.randint(15, 45)

            invoices.append({
                'issue_date': invoice_date + timedelta(days=random.randint(0, 25)),
                'amount': max(500, amount),
                'payment_days': payment_days
            })
    
    df = pd.DataFrame(invoices)
    print(f"✅ Generated {len(df)} invoices over 12 months")

    return df

def main():
    """
    Main training script
    """
    print("=" * 60)
    print("🚀 REVENUE FORECASTER TRAINING")
    print("=" * 60)
    
    # Step 1: Load data (using sample for now)
    invoices_df = generate_sample_data()
    
    print(f"\n📊 Data Summary:")
    print(f"   Total Invoices: {len(invoices_df)}")
    print(f"   Date Range: {invoices_df['issue_date'].min().date()} to {invoices_df['issue_date'].max().date()}")
    print(f"   Total Revenue: ${invoices_df['amount'].sum():,.2f}")
    print(f"   Avg Invoice: ${invoices_df['amount'].mean():,.2f}")
    
    # Step 2: Initialize and train forecaster
    print("\n" + "=" * 60)
    forecaster = RevenueForecaster()
    forecaster.fit(invoices_df)
    
    # Step 3: Generate forecasts
    print("\n" + "=" * 60)
    print("🔮 GENERATING 6-MONTH FORECAST")
    print("=" * 60)
    
    forecast_df = forecaster.forecast(months_ahead=6)
    
    print("\n📈 Revenue Predictions:")
    print("-" * 60)
    for _, row in forecast_df.iterrows():
        print(f"\n{row['month']}:")
        print(f"   Predicted: ${row['predicted_revenue']:,.2f}")
        print(f"   Range: ${row['lower_bound']:,.2f} - ${row['upper_bound']:,.2f}")
        print(f"   Confidence: {row['confidence']}%")
    
    # Step 4: Get insights
    print("\n" + "=" * 60)
    print("💡 BUSINESS INSIGHTS")
    print("=" * 60)
    
    insights = forecaster.get_insights()
    
    print(f"\n📊 Current State:")
    print(f"   This Month: ${insights['current_month_revenue']:,.2f}")
    print(f"   Monthly Average: ${insights['avg_monthly_revenue']:,.2f}")
    print(f"   Growth Rate: {insights['growth_rate']:+.1f}% per month")
    
    print(f"\n📈 Trend Analysis:")
    print(f"   Direction: {insights['trend']}")
    print(f"   Volatility: {insights['volatility']}")
    
    print(f"\n🏆 Performance:")
    print(f"   Best Month: {insights['best_month']} (${insights['best_month']['revenue']:,.2f})")
    print(f"   Worst Month: {insights['worst_month']} (${insights['worst_month']['revenue']:,.2f})")
    
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    
    # Save model (optional)
    save_choice = input("\n💾 Save trained model? (y/n): ").lower()
    if save_choice == 'y':
        model_dir = Path(__file__).parent.parent.parent / 'models' / 'saved'
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / 'revenue_forecaster.joblib'
        joblib.dump(forecaster, str(model_path))
        print(f"✅ Model saved to: {model_path}")


if __name__ == "__main__":
    main()