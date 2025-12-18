"""
Client Risk Scoring - Training & Analysis
Calculates risk scores for all clients based on payment history
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# Setup paths
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# Import our modules
from data.loader import load_invoices_from_database, check_data_quality
from features.feature_engineering import calculate_client_history
from models.risk_scorer import ClientRiskScorer


print("=" * 60)
print("      CLIENT RISK SCORING ANALYSIS")
print("=" * 60)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n📊 Loading invoice data...")

invoices_df = load_invoices_from_database()

print(f"✅ Loaded {len(invoices_df)} invoices")
print(f"   Unique clients: {invoices_df['client_id'].nunique()}")

# ============================================================================
# DATA QUALITY CHECK
# ============================================================================
data_quality_ok = check_data_quality(invoices_df)

if not data_quality_ok:
    print("\n⚠️  Warning: Data quality issues detected.")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Analysis cancelled.")
        exit()


# ============================================================================
# CALCULATE CLIENT FEATURES
# ============================================================================
print("\n🔧 Calculating client payment features...")

# Ensure datetime
if not pd.api.types.is_datetime64_any_dtype(invoices_df['issue_date']):
    invoices_df['issue_date'] = pd.to_datetime(invoices_df['issue_date'])

# Calculate features for each client (using their FULL history)
# We'll get the most recent features for each client
invoices_df = invoices_df.sort_values('issue_date')

# Get client history features
client_features_all = calculate_client_history(invoices_df)

# Combine with original data
df_with_features = pd.concat([invoices_df, client_features_all], axis=1)

# Get the LATEST features for each client (most recent invoice)
client_latest_features = df_with_features.groupby('client_id').last().reset_index()

print(f"✅ Calculated features for {len(client_latest_features)} clients")


# ============================================================================
# STEP 4: CALCULATE RISK SCORES
# ============================================================================
print("\n🎯 Calculating risk scores...")

# Initialize risk scorer
risk_scorer = ClientRiskScorer()

# Calculate risk for each client
risk_results = []

for idx, row in client_latest_features.iterrows():
    client_features = {
        'client_avg_payment_days': row['client_avg_payment_days'],
        'client_payment_std': row['client_payment_std'],
        'client_late_payment_rate': row['client_late_payment_rate'],
        'client_total_invoices': row['client_total_invoices'],
        'client_payment_trend': row['client_payment_trend'],
        'days_since_last_invoice': row['days_since_last_invoice']
    }
    
    # Calculate risk
    risk_data = risk_scorer.calculate_risk_score(client_features)
    
    # Add client info
    risk_data['client_id'] = row['client_id']
    risk_data['avg_payment_days'] = row['client_avg_payment_days']
    risk_data['total_invoices'] = row['client_total_invoices']
    
    risk_results.append(risk_data)

risk_df = pd.DataFrame(risk_results)

print(f"✅ Calculated risk scores for {len(risk_df)} clients")


# ============================================================================
# STEP 5: ANALYZE RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("             RISK SCORE ANALYSIS")
print("=" * 60)

# Overall statistics
print(f"\n📊 Risk Score Distribution:")
print(f"   Mean: {risk_df['risk_score'].mean():.1f}/100")
print(f"   Median: {risk_df['risk_score'].median():.1f}/100")
print(f"   Min: {risk_df['risk_score'].min()}/100")
print(f"   Max: {risk_df['risk_score'].max()}/100")

# Risk level breakdown
print(f"\n📊 Risk Level Breakdown:")
risk_counts = risk_df['risk_level'].value_counts()
for level in ['low', 'medium', 'high']:
    count = risk_counts.get(level, 0)
    pct = (count / len(risk_df)) * 100
    emoji = '✅' if level == 'low' else '⚠️' if level == 'medium' else '🚨'
    print(f"   {emoji} {level.capitalize()}: {count} clients ({pct:.1f}%)")


# ============================================================================
# STEP 6: SHOW TOP/BOTTOM CLIENTS
# ============================================================================
print("\n" + "=" * 60)
print("             CLIENT RANKINGS")
print("=" * 60)

# Top 5 safest clients
print(f"\n✅ TOP 5 SAFEST CLIENTS (Lowest Risk):")
print("─" * 60)
safest = risk_df.nsmallest(5, 'risk_score')
for idx, client in safest.iterrows():
    print(f"\n{client['client_id']}")
    print(f"   Risk Score: {client['risk_score']}/100 ({client['risk_level'].upper()})")
    print(f"   Avg Payment: {client['avg_payment_days']:.1f} days")
    print(f"   Total Invoices: {int(client['total_invoices'])}")
    print(f"   Recommendation: {client['recommendation']}")

# Top 5 riskiest clients
print(f"\n🚨 TOP 5 RISKIEST CLIENTS (Highest Risk):")
print("─" * 60)
riskiest = risk_df.nlargest(5, 'risk_score')
for idx, client in riskiest.iterrows():
    print(f"\n{client['client_id']}")
    print(f"   Risk Score: {client['risk_score']}/100 ({client['risk_level'].upper()})")
    print(f"   Avg Payment: {client['avg_payment_days']:.1f} days")
    print(f"   Total Invoices: {int(client['total_invoices'])}")
    print(f"   Recommendation: {client['recommendation']}")


# ============================================================================
# STEP 7: DETAILED BREAKDOWN FOR ONE CLIENT
# ============================================================================
print("\n" + "=" * 60)
print("             DETAILED RISK BREAKDOWN (Sample Client)")
print("=" * 60)

# Pick a medium-risk client for detailed analysis
sample_client = risk_df[risk_df['risk_level'] == 'medium'].iloc[0] if len(risk_df[risk_df['risk_level'] == 'medium']) > 0 else risk_df.iloc[0]

print(f"\n📋 Client: {sample_client['client_id']}")
print(f"   Overall Risk Score: {sample_client['risk_score']}/100")
print(f"   Risk Level: {sample_client['risk_level'].upper()}")

print(f"\n   Risk Factor Breakdown:")
breakdown = sample_client['risk_breakdown']
print(f"   ├─ Late Payment Risk: {breakdown['late_payment_risk']:.1f}/100")
print(f"   ├─ Speed Risk: {breakdown['speed_risk']:.1f}/100")
print(f"   ├─ Consistency Risk: {breakdown['consistency_risk']:.1f}/100")
print(f"   ├─ Trend Risk: {breakdown['trend_risk']:.1f}/100")
print(f"   └─ Experience Risk: {breakdown['experience_risk']:.1f}/100")

print(f"\n   Recommendation: {sample_client['recommendation']}")


# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================
print("\n💾 Saving risk scores...")

# Create directory
output_dir = Path(__file__).parent.parent.parent / 'models' / 'saved'
output_dir.mkdir(parents=True, exist_ok=True)

# Save risk scores
risk_scores_path = output_dir / 'client_risk_scores.csv'
risk_df.to_csv(risk_scores_path, index=False)

# Save risk scorer model
risk_scorer_path = output_dir / 'risk_scorer.joblib'
joblib.dump(risk_scorer, risk_scorer_path)

# Save metadata
metadata = {
    'total_clients': len(risk_df),
    'risk_distribution': risk_counts.to_dict(),
    'mean_risk_score': float(risk_df['risk_score'].mean()),
    'scoring_date': pd.Timestamp.now().isoformat()
}
metadata_path = output_dir / 'risk_scorer_metadata.joblib'
joblib.dump(metadata, metadata_path)

print(f"   ✅ Risk scores saved to: {risk_scores_path}")
print(f"   ✅ Risk scorer saved to: {risk_scorer_path}")
print(f"   ✅ Metadata saved to: {metadata_path}")


# ============================================================================
# STEP 9: ACTIONABLE INSIGHTS
# ============================================================================
print("\n" + "=" * 60)
print("             ACTIONABLE INSIGHTS")
print("=" * 60)

# High risk clients needing attention
high_risk = risk_df[risk_df['risk_level'] == 'high']
if len(high_risk) > 0:
    print(f"\n🚨 {len(high_risk)} HIGH RISK clients need attention:")
    for idx, client in high_risk.iterrows():
        print(f"   • {client['client_id']}: {client['recommendation']}")
else:
    print(f"\n✅ No high-risk clients!")

# Medium risk clients to monitor
medium_risk = risk_df[risk_df['risk_level'] == 'medium']
if len(medium_risk) > 0:
    print(f"\n⚠️  {len(medium_risk)} MEDIUM RISK clients to monitor:")
    for idx, client in medium_risk.head(3).iterrows():
        print(f"   • {client['client_id']}: {client['recommendation']}")
    if len(medium_risk) > 3:
        print(f"   ... and {len(medium_risk) - 3} more")

# Low risk clients (celebrate!)
low_risk = risk_df[risk_df['risk_level'] == 'low']
print(f"\n✅ {len(low_risk)} LOW RISK clients - your reliable partners!")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("             SUMMARY")
print("=" * 60)

print(f"\n📊 Analyzed {len(risk_df)} clients")
print(f"   Low Risk: {len(low_risk)} clients ({len(low_risk)/len(risk_df)*100:.1f}%)")
print(f"   Medium Risk: {len(medium_risk)} clients ({len(medium_risk)/len(risk_df)*100:.1f}%)")
print(f"   High Risk: {len(high_risk)} clients ({len(high_risk)/len(risk_df)*100:.1f}%)")

print("\n💡 Next Steps:")
print("   1. Review high-risk clients")
print("   2. Consider requiring deposits from risky clients")
print("   3. Monitor medium-risk clients closely")
print("   4. Reward low-risk clients with better terms")

print("\n" + "=" * 60)
print("✅ RISK SCORING COMPLETE!")
print("=" * 60)