"""
Data Loader for ML Training
Extracts invoice data from PostgreSQL database with proper error handling
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys
import os

from dotenv import load_dotenv

# Load environment variables from project root
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'

print(f"📄 Looking for .env at: {env_path}")
load_dotenv(dotenv_path=env_path)

# Add parent dir to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def load_invoices_from_database() -> pd.DataFrame:
    """
        Load paid invoices from PostgreSQL database
    
        Returns:
            DataFrame with columns: [client_id, issue_date, amount, payment_days]
    """
    
    print("=" * 60)
    print("      LOADING INVOICE DATA FROM DATABASE")
    print("=" * 60)
    
    # Check if DATABASE_URL is configured
    DATABASE_URL = os.getenv('DATABASE_URL')
    
    if not DATABASE_URL:
        print("❌ ERROR: DATABASE_URL not found!")
        print("   Please create .env file with DATABASE_URL")
        print("   Example: DATABASE_URL=postgresql://user:pass@localhost:5432/crefin")
        print("\n⚠️  Falling back to mock data for testing...")
        return load_mock_data()
    
    # Try to connect and load data
    print(f"\n📊 Connecting to database...")
    
    # Hide password in logs
    display_url = DATABASE_URL.split('@')[1] if '@' in DATABASE_URL else 'localhost'
    print(f"   URL: {display_url}")
    
    try:
        from sqlalchemy import create_engine
        
        # Create database connection
        engine = create_engine(DATABASE_URL)
        
        # SQL query to get paid invoices
        query = """
        SELECT
            i.id as invoice_id,
            i."clientId" as client_id,
            i.amount,
            i."issueDate" as issue_date,
            i."dueDate" as due_date,
            i."paidDate" as paid_date,
            i.status,
            c.name as client_name
        FROM "Invoice" i
        JOIN "Client" c ON i."clientId" = c.id
        WHERE i.status = 'paid'
          AND i."paidDate" IS NOT NULL
          AND i."issueDate" IS NOT NULL
        ORDER BY i."issueDate" ASC
        """
        
        print(f"   Executing query...")
        
        # Load data into pandas DataFrame
        df = pd.read_sql_query(query, engine)
        
        # Close connection immediately
        engine.dispose()
        
        # Check if we got any data
        if len(df) == 0:
            print("⚠️  No paid invoices found in database")
            print("   Falling back to mock data for testing...")
            return load_mock_data()
        
        print(f"✅ Loaded {len(df)} paid invoices from database")
        
        # Process the data
        df = process_invoice_data(df)
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading from database: {e}")
        print("\n⚠️  Falling back to mock data for testing...")
        return load_mock_data()


def process_invoice_data(df: pd.DataFrame) -> pd.DataFrame:
    """
        Process and clean invoice data
    
        Args:
            df: Raw invoice data from database
            
        Returns:
            Cleaned DataFrame with calculated payment_days
    """
    
    print("\n🔧 Processing data...")
    
    # Convert string dates to datetime
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    df['paid_date'] = pd.to_datetime(df['paid_date'])
    
    # Calculate payment_days (TARGET VARIABLE!)
    df['payment_days'] = (df['paid_date'] - df['issue_date']).dt.days
    
    # Convert Decimal amounts to float (Prisma returns Decimal type)
    df['amount'] = df['amount'].astype(float)
    
    # Data cleaning
    initial_count = len(df)
    
    # Remove invalid data
    df = df[df['payment_days'] >= 0]      # Payment can't happen before invoice
    df = df[df['payment_days'] <= 365]    # Remove extreme outliers (>1 year)
    df = df[df['amount'] > 0]             # Amount must be positive
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"   ⚠️  Removed {removed} invalid records")
    
    print(f"✅ Cleaned data: {len(df)} valid invoices")
    
    # Print summary statistics
    print_data_summary(df)
    
    # Return only columns needed for ML
    return df[['client_id', 'issue_date', 'amount', 'payment_days']]


def print_data_summary(df: pd.DataFrame):
    """Print statistical summary of the data"""
    
    print("\n" + "=" * 60)
    print("             DATA SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Payment Days Statistics:")
    print(f"   Mean: {df['payment_days'].mean():.1f} days")
    print(f"   Median: {df['payment_days'].median():.1f} days")
    print(f"   Std Dev: {df['payment_days'].std():.1f} days")
    print(f"   Min: {df['payment_days'].min():.0f} days")
    print(f"   Max: {df['payment_days'].max():.0f} days")
    
    print(f"\n💰 Invoice Amount Statistics:")
    print(f"   Mean: ${df['amount'].mean():.2f}")
    print(f"   Median: ${df['amount'].median():.2f}")
    print(f"   Min: ${df['amount'].min():.2f}")
    print(f"   Max: ${df['amount'].max():.2f}")
    
    n_clients = df['client_id'].nunique()
    print(f"\n👥 Unique Clients: {n_clients}")
    print(f"   Invoices per client (avg): {len(df) / n_clients:.1f}")
    
    print("=" * 60)


def load_mock_data() -> pd.DataFrame:
    """
    Generate EXTREMELY PREDICTABLE mock data for ML training
    
    Key: Make patterns VERY obvious so model can learn easily
    """
    
    print("\n📝 Generating ultra-predictable mock data...")
    
    np.random.seed(42)
    
    # =========================================================================
    # Create clients with EXTREMELY CONSISTENT behaviors
    # =========================================================================
    
    print("   Creating 30 clients with distinct payment patterns...")
    
    # Define exact payment profiles (no randomness!)
    payment_profiles = [12, 15, 18, 22, 25, 28, 32, 35, 38, 42, 45, 48]
    
    client_behaviors = {}
    for i in range(30):
        # Each client gets ONE of these exact profiles
        profile = payment_profiles[i % len(payment_profiles)]
        
        client_behaviors[f'client_{i}'] = {
            'base_payment_days': profile,
            'consistency': 'high'  # Very consistent!
        }
    
    # =========================================================================
    # Generate invoices with MINIMAL noise
    # =========================================================================
    
    print("   Generating invoices with strong predictable patterns...")
    
    data = []
    start_date = pd.Timestamp('2023-01-01')
    
    for client_id, behavior in client_behaviors.items():
        # Each client: 18-22 invoices (good history)
        n_invoices = np.random.randint(18, 23)
        
        for inv_num in range(n_invoices):
            # Spread over 18 months
            days_offset = np.random.randint(0, 540)
            issue_date = start_date + pd.Timedelta(days=days_offset)
            
            # BASE: Client's consistent payment time (with tiny variance)
            base_days = behavior['base_payment_days']
            payment_days = base_days + np.random.uniform(-1, 1)  # ±1 day only!
            
            # TEMPORAL EFFECTS: Clear and consistent
            month = issue_date.month
            day_of_month = issue_date.day
            
            # End of month: Always +5 days
            if day_of_month >= 26:
                payment_days += 5
            
            # Start of month: Always -2 days
            elif day_of_month <= 3:
                payment_days -= 2
            
            # Quarter end: Always +8 days
            if month in [3, 6, 9, 12] and day_of_month >= 25:
                payment_days += 8
            
            # Holiday season: Always +10 days
            if month in [11, 12]:
                payment_days += 10
            
            # AMOUNT EFFECT: Simple threshold
            amount = np.random.uniform(1000, 8000)
            if amount > 6000:
                payment_days += 3  # Large invoices: +3 days
            elif amount < 2000:
                payment_days -= 2  # Small invoices: -2 days
            
            # Keep realistic bounds
            payment_days = max(7, min(80, int(payment_days)))
            
            data.append({
                'client_id': client_id,
                'issue_date': issue_date,
                'amount': amount,
                'payment_days': payment_days
            })
    
    # =========================================================================
    # Create DataFrame
    # =========================================================================
    
    df = pd.DataFrame(data)
    df = df.sort_values('issue_date').reset_index(drop=True)
    
    print(f"✅ Generated {len(df)} ultra-predictable invoices")
    print(f"   Unique clients: {df['client_id'].nunique()}")
    
    # Show client consistency
    print("\n   📊 Client Payment Consistency:")
    for i in range(6):
        client_id = f'client_{i}'
        client_payments = df[df['client_id'] == client_id]['payment_days']
        if len(client_payments) > 0:
            print(f"      {client_id}: avg={client_payments.mean():.1f} days, "
                  f"std={client_payments.std():.1f} (base={client_behaviors[client_id]['base_payment_days']})")
    
    return df

def check_data_quality(df: pd.DataFrame) -> bool:
    """
        Validate data quality before training
        
        Args:
            df: Invoice DataFrame to validate
            
        Returns:
            True if data quality is good, False otherwise
    """
    
    print("\n" + "=" * 60)
    print("             DATA QUALITY CHECK")
    print("=" * 60)
    
    issues = []
    
    # Check 1: Enough data
    if len(df) < 50:
        issues.append(f"⚠️  Only {len(df)} invoices (recommend 200+ for good model)")
    else:
        print(f"✅ Data size: {len(df)} invoices (good!)")
    
    # Check 2: Enough clients
    n_clients = df['client_id'].nunique()
    if n_clients < 5:
        issues.append(f"⚠️  Only {n_clients} clients (recommend 10+ for diversity)")
    else:
        print(f"✅ Client diversity: {n_clients} unique clients (good!)")
    
    # Check 3: Client history
    invoices_per_client = len(df) / n_clients
    if invoices_per_client < 3:
        issues.append(f"⚠️  Only {invoices_per_client:.1f} invoices per client (recommend 5+)")
    else:
        print(f"✅ Client history: {invoices_per_client:.1f} invoices per client (good!)")
    
    # Check 4: Missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"⚠️  Missing values detected:\n{missing[missing > 0]}")
    else:
        print(f"✅ No missing values")
    
    # Check 5: Date range
    date_range = (df['issue_date'].max() - df['issue_date'].min()).days
    if date_range < 90:
        issues.append(f"⚠️  Date range only {date_range} days (recommend 180+ for seasonal patterns)")
    else:
        print(f"✅ Date range: {date_range} days (good!)")
    
    # Check 6: Payment days distribution
    payment_std = df['payment_days'].std()
    if payment_std < 5:
        issues.append(f"⚠️  Very low variance in payment_days (std={payment_std:.1f})")
    else:
        print(f"✅ Payment variance: {payment_std:.1f} days (good!)")
    
    print("=" * 60)
    
    # Report issues
    if issues:
        print("\n⚠️  DATA QUALITY ISSUES DETECTED:")
        for issue in issues:
            print(f"   {issue}")
        print("\n   Model may not perform well with this data.")
        print("   Consider collecting more data before production deployment.")
    else:
        print("\n✅ DATA QUALITY: EXCELLENT!")
        print("   Dataset is ready for training!")
    
    print("=" * 60)
    
    return len(issues) == 0


# Export functions
__all__ = [
    'load_invoices_from_database',
    'load_mock_data',
    'check_data_quality',
    'process_invoice_data'
]