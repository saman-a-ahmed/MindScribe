"""
Configuration management for MindScribe
"""

import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


class Settings:
    """
    Application settings loaded from environment variables
    """
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./mindscribe.db"
    )
    
    # Model paths
    EMOTION_MODEL_PATH: str = os.getenv(
        "EMOTION_MODEL_PATH",
        "models/emotion_classifier"
    )
    
    DISTORTION_MODEL_PATH: str = os.getenv(
        "DISTORTION_MODEL_PATH",
        "models/distortion_classifier"
    )
    
    # Analysis thresholds
    EMOTION_THRESHOLD: float = float(os.getenv("EMOTION_THRESHOLD", "0.3"))
    DISTORTION_THRESHOLD: float = float(os.getenv("DISTORTION_THRESHOLD", "0.3"))
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # CORS settings (comma-separated origins in the env var)
    CORS_ORIGINS: List[str] = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://localhost:3000"
    ).split(",")
    
    # Application
    APP_NAME: str = os.getenv("APP_NAME", "MindScribe")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # User settings (for future multi-user support)
    DEFAULT_USER_ID: str = os.getenv("DEFAULT_USER_ID", "default")
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate that required settings are present
        
        Returns:
            True if valid, raises exception if not
        """
        # Check if model paths exist (warn if not, but don't fail)
        emotion_path = Path(cls.EMOTION_MODEL_PATH)
        distortion_path = Path(cls.DISTORTION_MODEL_PATH)
        
        if not emotion_path.exists():
            print(f"⚠️  Warning: Emotion model not found at {cls.EMOTION_MODEL_PATH}")
        
        if not distortion_path.exists():
            print(f"⚠️  Warning: Distortion model not found at {cls.DISTORTION_MODEL_PATH}")
        
        return True


# Global settings instance
settings = Settings()
