"""
Centralized Configuration for JARVISv3
Uses Pydantic Settings for robust environment management and validation.
"""
import secrets
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings and environment variables
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project Info
    PROJECT_NAME: str = "JARVISv3"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    JWT_SECRET_KEY: Optional[str] = None
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Data Directory
    JARVIS_DATA_DIR: str = "./data"

    # Database
    _DATABASE_URL: str = "sqlite:///./JARVISv3.db"  # Fallback for backward compatibility

    # AI Models
    MODEL_PATH: str = "./models"
    DEFAULT_LLM_TIER: str = "medium"

    # Redis (Caching)
    ENABLE_CACHE: bool = True
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Voice Service
    WAKE_WORD: str = "Jarvis"
    WAKE_WORD_SENSITIVITY: float = 0.5
    STT_MODEL_TIER: str = "base"
    TTS_VOICE: str = "en_US-lessac-medium"
    TTS_PREFERRED_VOICE: Optional[str] = None

    # Search Services
    SEARCH_ENABLED: bool = True
    SEARCH_PROVIDERS: str = "duckduckgo"  # Comma-separated: duckduckgo,bing,google,tavily
    SEARCH_MAX_RESULTS: int = 5
    SEARCH_BING_API_KEY: Optional[str] = None
    SEARCH_BING_ENDPOINT: str = "https://api.bing.microsoft.com/v7.0/search"
    SEARCH_GOOGLE_API_KEY: Optional[str] = None
    SEARCH_GOOGLE_CX: Optional[str] = None
    SEARCH_TAVILY_API_KEY: Optional[str] = None

    # Distributed
    IS_DISTRIBUTED: bool = False
    LOCAL_NODE_ID: str = secrets.token_hex(4)

    @property
    def effective_secret_key(self) -> str:
        """Return the user-defined secret key or the auto-generated one"""
        return self.JWT_SECRET_KEY or self.SECRET_KEY

    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL using the configured data directory"""
        db_path = Path(self.JARVIS_DATA_DIR) / "JARVISv3.db"
        return f"sqlite:///{db_path}"

    # Keep backward compatibility - if someone accesses DATABASE_URL directly
    # it will use the constructed path, but we keep _DATABASE_URL as fallback


settings = Settings()
