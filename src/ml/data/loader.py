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
        Generate realistic mock data for testing
        
        Returns:
            DataFrame with mock invoice data
    """
    
    print("\n📝 Generating mock data for testing...")
    
    np.random.seed(42)  # Reproducible results
    
    # Configuration
    n_invoices = 200
    n_clients = 20
    
    # Create client behaviors (some fast payers, some slow)
    client_behaviors = {}
    for i in range(n_clients):
        client_behaviors[f'client_{i}'] = {
            'avg_days': np.random.randint(10, 50),
            'std_days': np.random.randint(2, 10),
        }
    
    # Generate invoices
    data = []
    start_date = pd.Timestamp('2024-01-01')
    
    for i in range(n_invoices):
        # Pick random client
        client_id = f'client_{np.random.randint(0, n_clients)}'
        behavior = client_behaviors[client_id]
        
        # Random date in 2024
        days_offset = np.random.randint(0, 300)
        issue_date = start_date + pd.Timedelta(days=days_offset)
        
        # Payment days based on client behavior (normal distribution)
        payment_days = max(5, int(np.random.normal(
            behavior['avg_days'],
            behavior['std_days']
        )))
        
        # Invoice amount (realistic range)
        amount = np.random.uniform(500, 10000)
        
        data.append({
            'client_id': client_id,
            'issue_date': issue_date,
            'amount': amount,
            'payment_days': payment_days
        })
    
    # Create DataFrame AFTER the loop (not inside!)
    df = pd.DataFrame(data)
    
    print(f"✅ Generated {len(df)} mock invoices")
    print(f"   Unique clients: {df['client_id'].nunique()}")
    
    # Print summary for mock data too
    print_data_summary(df)
    
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