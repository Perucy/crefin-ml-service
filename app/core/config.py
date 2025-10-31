"""
    Config file for the ML Service
    file loads settings from .env file
"""

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Basic service info
    service_name: str = "crefin-ml-service"
    service_version: str = "1.0.0"
    environment: str = "development"

    #server config
    host: str = "0.0.0.0"
    port: int = 8000

    #logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()