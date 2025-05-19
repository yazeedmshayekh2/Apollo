"""
Model Registry for the Vehicle Registration Card OCR System.

This module provides a registry for managing different models and their configurations.
It enables version control, model switching, and performance tracking.
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Registry for managing vision language models for OCR processing.
    
    This class provides functionality to:
    - Register and track different model versions
    - Store model configurations and performance metrics
    - Select the appropriate model based on requirements
    - Manage model lifecycle (staging, production, deprecated)
    """
    
    def __init__(self, registry_path: Optional[str] = None):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to the registry file (JSON)
        """
        self.registry_path = registry_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "model_registry.json"
        )
        self.registry = self._load_registry()
        
        # Ensure required keys exist
        if "models" not in self.registry:
            self.registry["models"] = {}
        if "default_model" not in self.registry:
            self.registry["default_model"] = None
        if "last_updated" not in self.registry:
            self.registry["last_updated"] = datetime.now().isoformat()
            
        # Current active model
        self.active_model_id = self.get_default_model_id()
        
        logger.info(f"Model registry initialized with {len(self.registry['models'])} models")
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the registry from file or create if it doesn't exist."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading registry: {e}")
                # Return empty registry in case of error
                return {"models": {}, "default_model": None, "last_updated": datetime.now().isoformat()}
        else:
            # Create default registry
            registry = {
                "models": {
                    "qwen2.5-vl-7b-instruct-awq": {
                        "model_id": "qwen2.5-vl-7b-instruct-awq",
                        "model_name": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                        "description": "Default Qwen 2.5 Vision model with AWQ quantization",
                        "version": "2.5",
                        "status": "production",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "parameters": "7B",
                        "quantization": "AWQ",
                        "configuration": {
                            "use_flash_attention": True,
                            "device_map": "auto",
                            "min_pixels": 256 * 28 * 28,
                            "max_pixels": 1280 * 28 * 28,
                        },
                        "metrics": {
                            "avg_processing_time": None,
                            "accuracy": None,
                            "success_rate": None,
                        }
                    }
                },
                "default_model": "qwen2.5-vl-7b-instruct-awq",
                "last_updated": datetime.now().isoformat()
            }
            
            # Save the default registry
            self._save_registry(registry)
            return registry
    
    def _save_registry(self, registry: Optional[Dict[str, Any]] = None) -> None:
        """Save the registry to file."""
        if registry is None:
            registry = self.registry
            
        registry["last_updated"] = datetime.now().isoformat()
        
        try:
            os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
            with open(self.registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
        except IOError as e:
            logger.error(f"Error saving registry: {e}")
    
    def register_model(
        self,
        model_name: str,
        description: str,
        version: str,
        status: str = "production",
        parameters: Optional[str] = None,
        quantization: Optional[str] = None,
        configuration: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        set_as_default: bool = False
    ) -> str:
        """
        Register a new model or update an existing one.
        
        Args:
            model_name: The full name/path of the model
            description: Description of the model
            version: Version identifier
            status: Model status ('development', 'staging', 'production', 'deprecated')
            parameters: Size of the model (e.g., '7B', '14B')
            quantization: Quantization method if any
            configuration: Model configuration parameters
            metrics: Performance metrics for the model
            set_as_default: Whether to set this model as the default
            
        Returns:
            Model ID (hash of model name and version)
        """
        # Generate a unique ID based on the model name and version
        model_id = self._generate_model_id(model_name, version)
        
        # Check if the model already exists
        is_update = model_id in self.registry["models"]
        
        # Prepare the model entry
        model_entry = {
            "model_id": model_id,
            "model_name": model_name,
            "description": description,
            "version": version,
            "status": status,
            "created_at": self.registry["models"].get(model_id, {}).get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat(),
            "parameters": parameters,
            "quantization": quantization,
            "configuration": configuration or {},
            "metrics": metrics or {}
        }
        
        # Update or add the model
        self.registry["models"][model_id] = model_entry
        
        # Set as default if requested
        if set_as_default:
            self.set_default_model(model_id)
        
        # Save the registry
        self._save_registry()
        
        action = "Updated" if is_update else "Registered"
        logger.info(f"{action} model {model_name} (ID: {model_id})")
        
        return model_id
    
    def update_model_metrics(
        self, 
        model_id: str, 
        metrics: Dict[str, Any]
    ) -> bool:
        """
        Update the metrics for a model.
        
        Args:
            model_id: The model ID
            metrics: The metrics to update
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.registry["models"]:
            logger.error(f"Model ID {model_id} not found in registry")
            return False
        
        # Get current metrics and update with new ones
        current_metrics = self.registry["models"][model_id].get("metrics", {})
        updated_metrics = {**current_metrics, **metrics}
        
        # Update the registry
        self.registry["models"][model_id]["metrics"] = updated_metrics
        self.registry["models"][model_id]["updated_at"] = datetime.now().isoformat()
        
        # Save the registry
        self._save_registry()
        
        logger.info(f"Updated metrics for model {model_id}")
        return True
    
    def update_model_status(self, model_id: str, status: str) -> bool:
        """
        Update the status of a model.
        
        Args:
            model_id: The model ID
            status: The new status ('development', 'staging', 'production', 'deprecated')
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.registry["models"]:
            logger.error(f"Model ID {model_id} not found in registry")
            return False
        
        # Update the registry
        self.registry["models"][model_id]["status"] = status
        self.registry["models"][model_id]["updated_at"] = datetime.now().isoformat()
        
        # Save the registry
        self._save_registry()
        
        logger.info(f"Updated status for model {model_id} to {status}")
        return True
    
    def get_model_by_id(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a model by its ID.
        
        Args:
            model_id: The model ID
            
        Returns:
            The model entry or None if not found
        """
        return self.registry["models"].get(model_id)
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest model with the given name.
        
        Args:
            model_name: The model name
            
        Returns:
            The model entry or None if not found
        """
        # Find all models with the given name
        matching_models = [
            model for model in self.registry["models"].values()
            if model["model_name"] == model_name
        ]
        
        # If no matching models, return None
        if not matching_models:
            return None
        
        # Return the most recently updated one
        return max(matching_models, key=lambda m: m["updated_at"])
    
    def get_default_model_id(self) -> Optional[str]:
        """
        Get the ID of the default model.
        
        Returns:
            The default model ID or None if not set
        """
        default_id = self.registry["default_model"]
        
        # Verify the default model still exists
        if default_id and default_id not in self.registry["models"]:
            logger.warning(f"Default model {default_id} not found in registry, resetting default")
            self.registry["default_model"] = None
            self._save_registry()
            return None
            
        return default_id
    
    def get_default_model(self) -> Optional[Dict[str, Any]]:
        """
        Get the default model.
        
        Returns:
            The default model entry or None if not set
        """
        default_id = self.get_default_model_id()
        if default_id:
            return self.get_model_by_id(default_id)
        return None
    
    def set_default_model(self, model_id: str) -> bool:
        """
        Set the default model.
        
        Args:
            model_id: The model ID to set as default
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.registry["models"]:
            logger.error(f"Model ID {model_id} not found in registry")
            return False
        
        # Update the registry
        self.registry["default_model"] = model_id
        
        # Save the registry
        self._save_registry()
        
        logger.info(f"Set default model to {model_id}")
        return True
    
    def get_active_model_config(self) -> Dict[str, Any]:
        """
        Get the configuration for the active model.
        
        Returns:
            The model configuration
        """
        model_id = self.active_model_id or self.get_default_model_id()
        
        if not model_id or model_id not in self.registry["models"]:
            logger.warning("No active or default model found, using fallback configuration")
            return {
                "model_name": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                "use_flash_attention": True,
                "device_map": "auto",
            }
        
        model = self.registry["models"][model_id]
        config = {
            "model_name": model["model_name"],
            **model.get("configuration", {})
        }
        
        return config
    
    def set_active_model(self, model_id: str) -> bool:
        """
        Set the active model for the current session.
        
        Args:
            model_id: The model ID to activate
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.registry["models"]:
            logger.error(f"Model ID {model_id} not found in registry")
            return False
        
        self.active_model_id = model_id
        logger.info(f"Set active model to {model_id}")
        return True
    
    def list_models(
        self, 
        status: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List models in the registry.
        
        Args:
            status: Filter by status
            limit: Maximum number of models to return
            
        Returns:
            List of model entries
        """
        models = list(self.registry["models"].values())
        
        # Filter by status if provided
        if status:
            models = [m for m in models if m["status"] == status]
        
        # Sort by updated_at (newest first)
        models.sort(key=lambda m: m["updated_at"], reverse=True)
        
        # Limit the number of results
        return models[:limit]
    
    def list_production_models(self) -> List[Dict[str, Any]]:
        """
        List all production models.
        
        Returns:
            List of production model entries
        """
        return self.list_models(status="production", limit=100)
    
    def _generate_model_id(self, model_name: str, version: str) -> str:
        """
        Generate a unique ID for a model based on its name and version.
        
        Args:
            model_name: The model name
            version: The model version
            
        Returns:
            A unique ID
        """
        # Remove the organization prefix if present
        name_parts = model_name.split('/')
        if len(name_parts) > 1:
            model_name = name_parts[-1]
            
        # Normalize the name
        model_name = model_name.lower().replace(' ', '-')
        version = version.lower().replace(' ', '-')
        
        # Clean up special characters
        model_id = f"{model_name}-{version}"
        model_id = ''.join(c if c.isalnum() or c == '-' else '-' for c in model_id)
        
        # Remove consecutive hyphens
        while '--' in model_id:
            model_id = model_id.replace('--', '-')
        
        # Remove leading/trailing hyphens
        model_id = model_id.strip('-')
        
        return model_id
    
    def get_model_performance_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get the performance history of a model from the performance log.
        
        Args:
            model_id: The model ID
            
        Returns:
            List of performance entries
        """
        performance_log_path = os.path.join(
            os.path.dirname(self.registry_path), 
            "performance_log.json"
        )
        
        if not os.path.exists(performance_log_path):
            return []
            
        try:
            with open(performance_log_path, 'r') as f:
                log = json.load(f)
                
            # Filter entries for the specified model
            model_entries = [
                entry for entry in log.get("entries", [])
                if entry.get("model_id") == model_id
            ]
            
            # Sort by timestamp (newest first)
            model_entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
            
            return model_entries
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading performance log: {e}")
            return []
    
    def log_model_performance(
        self, 
        model_id: str, 
        metrics: Dict[str, Any], 
        sample_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Log model performance for a specific run.
        
        Args:
            model_id: The model ID
            metrics: Performance metrics
            sample_id: Identifier for the processed sample (optional)
            details: Additional details about the run (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.registry["models"]:
            logger.error(f"Model ID {model_id} not found in registry")
            return False
            
        # Get model info
        model = self.registry["models"][model_id]
        
        # Prepare log entry
        entry = {
            "model_id": model_id,
            "model_name": model["model_name"],
            "version": model["version"],
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
            "sample_id": sample_id,
            "details": details or {}
        }
        
        # Load existing log
        performance_log_path = os.path.join(
            os.path.dirname(self.registry_path), 
            "performance_log.json"
        )
        
        if os.path.exists(performance_log_path):
            try:
                with open(performance_log_path, 'r') as f:
                    log = json.load(f)
            except (json.JSONDecodeError, IOError):
                log = {"entries": []}
        else:
            log = {"entries": []}
        
        # Add entry
        log["entries"].append(entry)
        
        # Save log
        try:
            os.makedirs(os.path.dirname(performance_log_path), exist_ok=True)
            with open(performance_log_path, 'w') as f:
                json.dump(log, f, indent=2)
                
            # Update model metrics with averages
            self._update_model_average_metrics(model_id)
                
            return True
        except IOError as e:
            logger.error(f"Error saving performance log: {e}")
            return False
    
    def _update_model_average_metrics(self, model_id: str) -> None:
        """
        Update model metrics with averages from the performance log.
        
        Args:
            model_id: The model ID
        """
        # Get performance history
        history = self.get_model_performance_history(model_id)
        
        if not history:
            return
            
        # Calculate average metrics
        avg_metrics = {}
        
        # Find common metric keys
        metric_keys = set()
        for entry in history:
            metric_keys.update(entry.get("metrics", {}).keys())
        
        # Calculate averages
        for key in metric_keys:
            values = [
                entry["metrics"][key] 
                for entry in history 
                if key in entry.get("metrics", {}) and entry["metrics"][key] is not None
            ]
            
            if values:
                avg_metrics[f"avg_{key}"] = sum(values) / len(values)
        
        # Update model metrics
        if avg_metrics:
            self.update_model_metrics(model_id, avg_metrics)

# Initialize global registry
registry = ModelRegistry()

def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return registry
