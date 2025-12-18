"""
    Client segmentation training
    Automatically groups similar clients into segments using K-Means clustering
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import joblib

# setu paths
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

# import modules
from data.loader import load_invoices_from_database
from features.feature_engineering import calculate_client_history
from models.risk_scorer import ClientRiskScorer
from models.client_segmenter import ClientSegmenter

print("=" * 60)
print("      CLIENT SEGMENTATION ANALYSIS")
print("=" * 60)

# ============================================================================
# LOAD DATA AND CALCULATE FEATURES
# ============================================================================
print("\n📊 Loading invoice data and calculating features...")

# load invoices
invoices_df = load_invoices_from_database()

print(f"✅ Loaded {len(invoices_df)} invoices from {invoices_df['client_id'].nunique()} clients")

# ensure datetime
if not pd.api.types.is_datetime64_any_dtype(invoices_df['issue_date']):
    invoices_df['issue_date'] = pd.to_datetime(invoices_df['issue_date'])

# sort by date
invoices_df = invoices_df.sort_values('issue_date')

# client history features
print("\n🔧 Calculating client features...")
client_features_all = calculate_client_history(invoices_df)

# combine with original data
df_with_features = pd.concat([invoices_df, client_features_all], axis=1)

# get the LATEST features for each client (most recent invoice)
clients_df = df_with_features.groupby('client_id').last().reset_index()

print(f"✅ Calculated features for {len(clients_df)} clients")

# ============================================================================
# CALCULATE RISK SCORES
# ============================================================================
print("\n🎯 Calculating risk scores for segmentation...")

# risk scorer
risk_scorer = ClientRiskScorer()

# calculate risk for each client
risk_scores = []

for idx, client in clients_df.iterrows():
    client_features = {
        'client_avg_payment_days': client['client_avg_payment_days'],
        'client_payment_std': client['client_payment_std'],
        'client_late_payment_rate': client['client_late_payment_rate'],
        'client_total_invoices': client['client_total_invoices'],
        'client_payment_trend': client['client_payment_trend'],
        'days_since_last_invoice': client['days_since_last_invoice']
    }

    risk_data = risk_scorer.calculate_risk_score(client_features)
    risk_scores.append(risk_data['risk_score'])

clients_df['risk_score'] = risk_scores

print(f"✅ Risk scores calculated")

# ============================================================================
# TRAIN SEGMENTATION MODEL
# ============================================================================
print("\n" + "=" * 60)
print("      TRAINING SEGMENTATION MODEL")
print("=" * 60)

# segmenter
segmenter = ClientSegmenter(n_segments=4)

# train on client data
segmenter.fit(clients_df)

# ============================================================================
# ANALYZE SEGMENTS
# ============================================================================
print("\n" + "=" * 60)
print("      ANALYZE SEGMENTS")
print("=" * 60)

# get segment summary
summary_df = segmenter.get_segment_summary()

print("\n📊 Segment Overview:")
print("─" * 60)
print(summary_df.to_string(index=False))

# ============================================================================
# DETAILED SEGMENT PROFILES
# ============================================================================
print("\n" + "=" * 60)
print("      DETAILED SEGMENT PROFILES")
print("=" * 60)

for segment_id, profile in segmenter.segment_profiles.items():
    print(f"\n{'=' * 60}")
    print(f"  SEGMENT {segment_id}: {profile['name'].upper()}")
    print(f"{'=' * 60}")
    
    print(f"\n📊 Characteristics:")
    print(f"   Clients in segment: {profile['count']}")
    print(f"   Avg payment time: {profile['avg_payment_days']} days")
    print(f"   Payment consistency: {profile['avg_consistency']} days (std)")
    print(f"   Avg risk score: {profile['avg_risk_score']}/100")
    print(f"   Late payment rate: {profile['late_payment_rate']}%")
    
    print(f"\n📝 Description:")
    print(f"   {profile['description']}")
    
    print(f"\n💡 Recommendation:")
    print(f"   {profile['recommendation']}")
    
    # Show sample clients from this segment
    segment_clients = clients_df[clients_df['segment'] == segment_id].head(3)
    
    print(f"\n👥 Sample Clients:")
    for idx, client in segment_clients.iterrows():
        print(f"   • {client['client_id']}: "
              f"{client['client_avg_payment_days']:.1f} days avg, "
              f"risk {client['risk_score']}/100")
        
# ============================================================================
# CLIENT-BY-CLIENT BREAKDOWN
# ============================================================================
print("\n" + "=" * 60)
print("      ALL CLIENTS SEGMENTED")
print("=" * 60)

# sort by segment and risk score
clients_sorted = clients_df.sort_values(['segment', 'risk_score'])

print("\n")
for segment_id in range(4):
    segment_clients = clients_sorted[clients_sorted['segment'] == segment_id]
    
    if len(segment_clients) == 0:
        continue
    
    segment_name = segmenter.segment_profiles[segment_id]['name']
    
    print(f"\n{'─' * 60}")
    print(f"Segment {segment_id}: {segment_name} ({len(segment_clients)} clients)")
    print(f"{'─' * 60}")
    
    for idx, client in segment_clients.iterrows():
        print(f"  {client['client_id']:12} | "
              f"Avg: {client['client_avg_payment_days']:5.1f} days | "
              f"Risk: {client['risk_score']:2}/100 | "
              f"Invoices: {int(client['client_total_invoices']):2}")

# ============================================================================
# ACTIONABLE INSIGHTS
# ============================================================================
print("\n" + "=" * 60)
print("      ACTIONABLE INSIGHTS")
print("=" * 60)

# Identify best and worst segments
segment_by_risk = summary_df.sort_values('Risk Score')
best_segment = segment_by_risk.iloc[0]
worst_segment = segment_by_risk.iloc[-1]

print(f"\n✅ BEST SEGMENT: {best_segment['Name']}")
print(f"   {best_segment['Clients']} clients | "
      f"Avg payment: {best_segment['Avg Payment (days)']} days | "
      f"Risk: {best_segment['Risk Score']}/100")
print(f"   → Focus on retaining and rewarding these clients!")

print(f"\n🚨 WORST SEGMENT: {worst_segment['Name']}")
print(f"   {worst_segment['Clients']} clients | "
      f"Avg payment: {worst_segment['Avg Payment (days)']} days | "
      f"Risk: {worst_segment['Risk Score']}/100")
print(f"   → Require deposits or reconsider working with these clients!")

# Portfolio balance
print(f"\n📊 PORTFOLIO BALANCE:")
for _, row in summary_df.iterrows():
    percentage = (row['Clients'] / len(clients_df)) * 100
    bar_length = int(percentage / 2)
    bar = '█' * bar_length
    print(f"   {row['Name']:20} | {bar} {percentage:.1f}%")


# ============================================================================
# SAVE MODEL AND RESULTS
# ============================================================================
print("\n💾 Saving segmentation model and results...")

# Create directory
output_dir = Path(__file__).parent.parent.parent / 'models' / 'saved'
output_dir.mkdir(parents=True, exist_ok=True)

# Save segmentation model
segmenter_path = output_dir / 'client_segmenter.joblib'
joblib.dump(segmenter, segmenter_path)

# Save segmented clients
clients_segmented_path = output_dir / 'clients_segmented.csv'
clients_df[['client_id', 'segment', 'client_avg_payment_days', 
            'client_payment_std', 'risk_score']].to_csv(clients_segmented_path, index=False)

# Save segment profiles
segment_summary_path = output_dir / 'segment_profiles.csv'
summary_df.to_csv(segment_summary_path, index=False)

# Save metadata
metadata = {
    'n_segments': segmenter.n_segments,
    'total_clients': len(clients_df),
    'segment_distribution': summary_df['Clients'].to_dict(),
    'training_date': pd.Timestamp.now().isoformat()
}
metadata_path = output_dir / 'segmenter_metadata.joblib'
joblib.dump(metadata, metadata_path)

print(f"   ✅ Segmenter saved to: {segmenter_path}")
print(f"   ✅ Segmented clients saved to: {clients_segmented_path}")
print(f"   ✅ Segment profiles saved to: {segment_summary_path}")
print(f"   ✅ Metadata saved to: {metadata_path}")


# ============================================================================
# BUSINESS RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 60)
print("      STRATEGIC RECOMMENDATIONS")
print("=" * 60)

print("\n💡 Key Takeaways:")

# Calculate portfolio metrics
low_risk_count = len(clients_df[clients_df['risk_score'] <= 30])
high_risk_count = len(clients_df[clients_df['risk_score'] >= 60])

print(f"\n1. Portfolio Health:")
print(f"   • {low_risk_count} clients are low-risk ({low_risk_count/len(clients_df)*100:.1f}%)")
print(f"   • {high_risk_count} clients are high-risk ({high_risk_count/len(clients_df)*100:.1f}%)")

print(f"\n2. Segment-Based Strategies:")
for segment_id, profile in segmenter.segment_profiles.items():
    print(f"   • {profile['name']}: {profile['recommendation']}")

print(f"\n3. Next Steps:")
print(f"   ✓ Review each segment's characteristics")
print(f"   ✓ Adjust payment terms per segment")
print(f"   ✓ Focus growth on best-performing segments")
print(f"   ✓ Consider exiting relationships with worst segment")


# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 60)
print("             SUMMARY")
print("=" * 60)

print(f"\n📊 Segmented {len(clients_df)} clients into {segmenter.n_segments} groups:")
for _, row in summary_df.iterrows():
    print(f"   • {row['Name']}: {row['Clients']} clients")

print(f"\n💾 Model and results saved to: {output_dir}")

print("\n" + "=" * 60)
print("✅ SEGMENTATION COMPLETE!")
print("=" * 60)
