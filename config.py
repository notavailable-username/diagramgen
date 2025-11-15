"""Configuration management for the bar model diagram generator.

This module handles environment variable loading and application configuration
using pydantic-settings for type-safe configuration management.
"""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        GEMINI_API_KEY: Google Gemini API key for LLM access.
        GEMINI_MODEL: The Gemini model to use (default: gemini-2.5-flash).
        MAX_RETRIES: Maximum number of retry attempts for API calls.
        RETRY_DELAY: Delay in seconds between retry attempts.
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=True
    )

    # Required settings
    GEMINI_API_KEY: str = Field(
        ...,
        description="Google Gemini API key"
    )

    # Optional settings with defaults
    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model identifier"
    )

    MAX_RETRIES: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Maximum retry attempts for API calls"
    )

    RETRY_DELAY: float = Field(
        default=0.3,
        ge=0.1,
        le=10.0,
        description="Delay between retries in seconds"
    )

    LOG_LEVEL: str = Field(
        default="ERROR",
        description="Logging level"
    )

    @field_validator('GEMINI_API_KEY')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is not empty."""
        if not v or not v.strip():
            raise ValueError("GEMINI_API_KEY cannot be empty")
        return v.strip()

    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the standard levels."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {valid_levels}, got '{v}'"
            )
        return v_upper


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path object pointing to the project root.
    """
    return Path(__file__).parent


def get_images_dir() -> Path:
    """Get the images output directory.

    Returns:
        Path object pointing to the images directory.
    """
    images_dir = get_project_root() / "images"
    images_dir.mkdir(exist_ok=True)
    return images_dir


def get_mcp_servers_dir() -> Path:
    """Get the MCP servers directory.

    Returns:
        Path object pointing to the mcp_servers directory.
    """
    return get_project_root() / "mcp_servers"


# Global settings instance
settings = Settings()
