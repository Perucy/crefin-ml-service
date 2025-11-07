"""
    Database Connection Management
    Handles connection pooling, retries, and error handling
"""

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError
import time
from typing import Optional
import pandas as pd

from config.settings import settings


class DatabaseManager:
    """
        Manages database connections with retry logic and error handling
        
        Industry practices:
        - Connection pooling for performance
        - Retry logic for transient failures
        - Proper error handling
        - Connection cleanup
    """
    
    def __init__(self):
        self.engine: Optional[object] = None
        self._connected = False
    
    def connect(self, max_retries: int = 3, retry_delay: int = 2) -> bool:
        """
            Connect to database with retry logic
            
            Args:
                max_retries: Maximum number of connection attempts
                retry_delay: Seconds to wait between retries
                
            Returns:
                True if connected successfully
        """
        
        if not settings.database_url:
            print("⚠️  No DATABASE_URL configured")
            return False
        
        print(f"📊 Connecting to database...")
        
        for attempt in range(1, max_retries + 1):
            try:
                # Create engine with connection pooling
                self.engine = create_engine(
                    settings.database_url,
                    poolclass=NullPool,  # No connection pooling for ML training
                    echo=False  # Set to True for SQL debugging
                )
                
                # Test connection
                with self.engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                
                self._connected = True
                print(f"   ✅ Connected successfully!")
                return True
                
            except OperationalError as e:
                print(f"   ⚠️  Connection attempt {attempt}/{max_retries} failed")
                
                if attempt < max_retries:
                    print(f"   Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"   ❌ Failed to connect after {max_retries} attempts")
                    print(f"   Error: {str(e)}")
                    return False
        
        return False
    
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """
        Execute SQL query and return DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with results, or None if failed
        """
        
        if not self._connected:
            print("❌ Not connected to database")
            return None
        
        try:
            df = pd.read_sql_query(query, self.engine)
            return df
            
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            self._connected = False
            print("✅ Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# ===================================================================
# GLOBAL DATABASE INSTANCE
# ===================================================================
db_manager = DatabaseManager()