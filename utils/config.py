"""
Configuration settings for the Vehicle Registration Card OCR System.
Loads settings from environment variables or defaults to sensible values.
"""

import os
from typing import Dict, Any, Optional, Union
import logging

# Set up logger
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for the OCR system."""
    
    # Default configuration
    _defaults = {
        # Model settings
        "MODEL_NAME": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        "USE_FLASH_ATTENTION": False,
        "DEVICE_MAP": "auto",
        "MIN_PIXELS": 256 * 28 * 28,  # Minimum image resolution for tokens
        "MAX_PIXELS": 1280 * 28 * 28, # Maximum image resolution for tokens
        "YOLO_MODEL": "yolo11l",  # YOLO model for face detection
        
        # Processing settings
        "VALIDATE_RESULTS": True,
        "MIN_CONFIDENCE": 0.7,
        "MAX_TOKENS": 512,
        
        # Output settings
        "DEFAULT_OUTPUT_DIR": "output",
        "DEFAULT_FORMAT": "json",
        
        # API settings
        "API_HOST": "0.0.0.0",
        "API_PORT": 8000,
        "API_WORKERS": 4,
        "API_TIMEOUT": 60,
        
        # Storage settings
        "STORAGE_TYPE": "local",  # local, s3, azure, etc.
        "S3_BUCKET": "vehicle-registration-ocr",
        "S3_PREFIX": "images/",
        "LOCAL_STORAGE_PATH": "storage/",
        
        # Database settings
        "DB_TYPE": "postgres",  # postgres, mongodb, etc.
        "DB_HOST": "localhost",
        "DB_PORT": 5432,
        "DB_NAME": "ocr_results",
        "DB_USER": "ocr_user",
        "DB_PASSWORD": "",
        
        # Monitoring settings
        "ENABLE_MONITORING": True,
        "LOG_LEVEL": "INFO",
        "SENTRY_DSN": "",
        "PROMETHEUS_PORT": 9090,
    }
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            
        Returns:
            Configuration value
        """
        # Check for environment variable first
        env_key = f"OCR_{key}"
        env_value = os.environ.get(env_key)
        
        if env_value is not None:
            return cls._convert_value(env_value)
        
        # Check defaults
        if key in cls._defaults:
            return cls._defaults[key]
            
        # Return provided default
        return default
    
    @classmethod
    def _convert_value(cls, value: str) -> Any:
        """Convert string value to appropriate type."""
        # Convert true/false strings to booleans
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
            
        # Convert numeric strings to int or float
        try:
            if value.isdigit():
                return int(value)
            if value.replace('.', '', 1).isdigit():
                return float(value)
        except (ValueError, AttributeError):
            pass
            
        # Return as string
        return value
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary."""
        config_dict = {}
        
        # Start with defaults
        for key in cls._defaults:
            config_dict[key] = cls.get(key)
            
        # Add any additional environment variables
        for key, value in os.environ.items():
            if key.startswith("OCR_"):
                config_key = key[4:]  # Remove "OCR_" prefix
                if config_key not in config_dict:
                    config_dict[config_key] = cls._convert_value(value)
        
        return config_dict

    @classmethod
    def model_config(cls) -> Dict[str, Any]:
        """Get model-specific configuration."""
        return {
            "model_name": cls.get("MODEL_NAME"),
            "device_map": cls.get("DEVICE_MAP"),
            "use_flash_attention": cls.get("USE_FLASH_ATTENTION"),
            "min_pixels": cls.get("MIN_PIXELS"),
            "max_pixels": cls.get("MAX_PIXELS"),
        }
    
    @classmethod
    def validation_config(cls) -> Dict[str, Any]:
        """Get validation-specific configuration."""
        return {
            "validate_results": cls.get("VALIDATE_RESULTS"),
            "min_confidence": cls.get("MIN_CONFIDENCE"),
        }
    
    @classmethod
    def api_config(cls) -> Dict[str, Any]:
        """Get API-specific configuration."""
        return {
            "host": cls.get("API_HOST"),
            "port": cls.get("API_PORT"),
            "workers": cls.get("API_WORKERS"),
            "timeout": cls.get("API_TIMEOUT"),
        }
    
    @classmethod
    def storage_config(cls) -> Dict[str, Any]:
        """Get storage-specific configuration."""
        storage_type = cls.get("STORAGE_TYPE")
        
        config = {
            "type": storage_type,
        }
        
        if storage_type == "s3":
            config.update({
                "bucket": cls.get("S3_BUCKET"),
                "prefix": cls.get("S3_PREFIX"),
            })
        elif storage_type == "local":
            config.update({
                "path": cls.get("LOCAL_STORAGE_PATH"),
            })
            
        return config

# Load configuration when module is imported
config = Config()

# Log configuration at startup in debug mode
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Loaded configuration: {config.as_dict()}")
