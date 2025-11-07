"""
    Config file
"""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Literal

class Settings(BaseSettings):
    """
        Application settings with validation
    """
    # ===================================================================
    # DATABASE CONFIGURATION
    # ===================================================================
    database_url: str = Field(
        default="",
        description="PostgreSQL connection URL"
    )

    # ===================================================================
    # MODEL CONFIGURATION
    # ===================================================================
    model_version: str = Field(
        default="v2",
        description="Model verification identifier"
    )

    model_path: Path = Field(
        default=Path("models/saved"),
        description="Directory to save trained models"
    )

    # ===================================================================
    # TRAINING CONFIGURATION
    # ===================================================================
    test_size: float = Field(
        default=0.2,
        ge=0.1,  # Must be >= 0.1
        le=0.5,  # Must be <= 0.5
        description="Fraction of data to use for testing"
    )
    
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    
    min_samples_for_split: int = Field(
        default=10,
        description="Minimum samples required for train/test split"
    )
    
    # Random Forest parameters
    n_estimators: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of trees in Random Forest"
    )
    
    max_depth: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Maximum depth of trees"
    )
    
    min_samples_split: int = Field(
        default=5,
        ge=2,
        description="Minimum samples required to split node"
    )
    
    min_samples_leaf: int = Field(
        default=2,
        ge=1,
        description="Minimum samples required in leaf node"
    )
    
    # ===================================================================
    # DATA QUALITY THRESHOLDS
    # ===================================================================
    min_invoices_for_training: int = Field(
        default=50,
        description="Minimum invoices needed for good model"
    )
    
    recommended_invoices: int = Field(
        default=200,
        description="Recommended number of invoices"
    )
    
    min_clients: int = Field(
        default=5,
        description="Minimum unique clients needed"
    )
    
    recommended_clients: int = Field(
        default=10,
        description="Recommended number of clients"
    )
    
    # ===================================================================
    # MOCK DATA CONFIGURATION
    # ===================================================================
    mock_n_invoices: int = Field(
        default=200,
        description="Number of mock invoices to generate"
    )
    
    mock_n_clients: int = Field(
        default=20,
        description="Number of mock clients to generate"
    )
    
    # ===================================================================
    # LOGGING CONFIGURATION
    # ===================================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    # ===================================================================
    # ENVIRONMENT CONFIGURATION
    # ===================================================================
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Application environment"
    )
    
    # ===================================================================
    # PYDANTIC CONFIGURATION
    # ===================================================================
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in .env
    )
    
    # ===================================================================
    # COMPUTED PROPERTIES
    # ===================================================================
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"
    
    @property
    def use_real_database(self) -> bool:
        """Check if we should use real database"""
        return bool(self.database_url)
    
    @property
    def model_file_path(self) -> Path:
        """Get full path to model file"""
        return self.model_path / f"payment_predictor_{self.model_version}.joblib"
    
    @property
    def metadata_file_path(self) -> Path:
        """Get full path to metadata file"""
        return self.model_path / f"payment_predictor_{self.model_version}_metadata.joblib"


# ===================================================================
# GLOBAL SETTINGS INSTANCE
# ===================================================================
settings = Settings()


# ===================================================================
# VALIDATION ON IMPORT
# ===================================================================
def validate_settings():
    """Validate settings on import"""
    
    # Create directories if they don't exist
    settings.model_path.mkdir(parents=True, exist_ok=True)
    
    # Log configuration
    print("=" * 60)
    print("      CONFIGURATION LOADED")
    print("=" * 60)
    print(f"Environment: {settings.environment}")
    print(f"Database: {'✅ Configured' if settings.use_real_database else '⚠️  Using mock data'}")
    print(f"Model Version: {settings.model_version}")
    print(f"Log Level: {settings.log_level}")
    print("=" * 60)


# Validate on import
validate_settings()

