import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
import sys
from datetime import datetime

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OCR.core.registry import ModelRegistry


class TestModelRegistry(unittest.TestCase):
    """Test cases for the ModelRegistry class."""

    def setUp(self):
        """Set up test cases with a temporary directory for registry files."""
        # Create a temporary directory for the registry files
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "test_registry.json")
        
        # Create the registry instance
        self.registry = ModelRegistry(registry_path=self.registry_path)
        
        # Sample model data for testing
        self.test_model = {
            "model_name": "TestModel/TestVersion",
            "description": "Test model for unit testing",
            "version": "1.0",
            "status": "development",
            "parameters": "7B",
            "quantization": "AWQ",
            "configuration": {
                "use_flash_attention": True,
                "device_map": "cuda"
            }
        }

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_init_creates_default_registry(self):
        """Test that initialization creates a default registry file if none exists."""
        # Check that the registry file was created
        self.assertTrue(os.path.exists(self.registry_path))
        
        # Check that the registry has the expected structure
        with open(self.registry_path, 'r') as f:
            registry_data = json.load(f)
            
        self.assertIn("models", registry_data)
        self.assertIn("default_model", registry_data)
        self.assertIn("last_updated", registry_data)
        
        # Check that a default model exists
        self.assertIn("qwen2.5-vl-7b-instruct-awq", registry_data["models"])

    def test_register_model(self):
        """Test registering a new model."""
        # Register a new model
        model_id = self.registry.register_model(
            model_name=self.test_model["model_name"],
            description=self.test_model["description"],
            version=self.test_model["version"],
            status=self.test_model["status"],
            parameters=self.test_model["parameters"],
            quantization=self.test_model["quantization"],
            configuration=self.test_model["configuration"]
        )
        
        # Check that the model ID was generated and returned
        self.assertIsNotNone(model_id)
        
        # Check that the model was added to the registry
        self.assertIn(model_id, self.registry.registry["models"])
        
        # Check that the model details were stored correctly
        registered_model = self.registry.registry["models"][model_id]
        self.assertEqual(registered_model["model_name"], self.test_model["model_name"])
        self.assertEqual(registered_model["description"], self.test_model["description"])
        self.assertEqual(registered_model["version"], self.test_model["version"])
        self.assertEqual(registered_model["configuration"], self.test_model["configuration"])

    def test_register_model_with_defaults(self):
        """Test registering a model with minimal required parameters."""
        # Register a model with only required parameters
        model_id = self.registry.register_model(
            model_name="MinimalModel",
            description="Minimal model",
            version="0.1"
        )
        
        # Check that the model has default values for optional parameters
        registered_model = self.registry.registry["models"][model_id]
        self.assertEqual(registered_model["status"], "development")
        self.assertEqual(registered_model["configuration"], {})
        self.assertEqual(registered_model["metrics"], {})

    def test_update_model_metrics(self):
        """Test updating model metrics."""
        # Register a model
        model_id = self.registry.register_model(
            model_name=self.test_model["model_name"],
            description=self.test_model["description"],
            version=self.test_model["version"]
        )
        
        # Update metrics
        metrics = {
            "accuracy": 0.95,
            "avg_processing_time": 1.23
        }
        
        success = self.registry.update_model_metrics(model_id, metrics)
        
        # Check that the update was successful
        self.assertTrue(success)
        
        # Check that the metrics were updated
        updated_model = self.registry.registry["models"][model_id]
        self.assertEqual(updated_model["metrics"]["accuracy"], 0.95)
        self.assertEqual(updated_model["metrics"]["avg_processing_time"], 1.23)

    def test_update_model_status(self):
        """Test updating model status."""
        # Register a model
        model_id = self.registry.register_model(
            model_name=self.test_model["model_name"],
            description=self.test_model["description"],
            version=self.test_model["version"],
            status="development"
        )
        
        # Update status to production
        success = self.registry.update_model_status(model_id, "production")
        
        # Check that the update was successful
        self.assertTrue(success)
        
        # Check that the status was updated
        updated_model = self.registry.registry["models"][model_id]
        self.assertEqual(updated_model["status"], "production")

    def test_set_default_model(self):
        """Test setting a model as the default."""
        # Register a model
        model_id = self.registry.register_model(
            model_name=self.test_model["model_name"],
            description=self.test_model["description"],
            version=self.test_model["version"]
        )
        
        # Set as default
        success = self.registry.set_default_model(model_id)
        
        # Check that the operation was successful
        self.assertTrue(success)
        
        # Check that the model is now the default
        self.assertEqual(self.registry.registry["default_model"], model_id)
        
        # Check that get_default_model_id returns the correct ID
        self.assertEqual(self.registry.get_default_model_id(), model_id)

    def test_get_model_by_id(self):
        """Test retrieving a model by ID."""
        # Register a model
        model_id = self.registry.register_model(
            model_name=self.test_model["model_name"],
            description=self.test_model["description"],
            version=self.test_model["version"]
        )
        
        # Retrieve the model
        model = self.registry.get_model_by_id(model_id)
        
        # Check that the correct model was retrieved
        self.assertEqual(model["model_name"], self.test_model["model_name"])
        self.assertEqual(model["description"], self.test_model["description"])
        
        # Test retrieving a non-existent model
        non_existent_model = self.registry.get_model_by_id("non_existent_id")
        self.assertIsNone(non_existent_model)

    def test_get_model_by_name(self):
        """Test retrieving the latest model with a given name."""
        # Register two versions of the same model
        self.registry.register_model(
            model_name="TestModel",
            description="Test model v1",
            version="1.0"
        )
        
        # Register a newer version
        self.registry.register_model(
            model_name="TestModel",
            description="Test model v2",
            version="2.0"
        )
        
        # Retrieve the model by name
        model = self.registry.get_model_by_name("TestModel")
        
        # Check that the latest version was retrieved
        self.assertEqual(model["version"], "2.0")
        self.assertEqual(model["description"], "Test model v2")

    def test_list_models(self):
        """Test listing models with different filters."""
        # Register models with different statuses
        self.registry.register_model(
            model_name="DevModel1",
            description="Development model 1",
            version="1.0",
            status="development"
        )
        
        self.registry.register_model(
            model_name="DevModel2",
            description="Development model 2",
            version="1.0",
            status="development"
        )
        
        self.registry.register_model(
            model_name="ProdModel",
            description="Production model",
            version="1.0",
            status="production"
        )
        
        # List all models
        all_models = self.registry.list_models()
        self.assertEqual(len(all_models), 3)
        
        # List only development models
        dev_models = self.registry.list_models(status="development")
        self.assertEqual(len(dev_models), 2)
        self.assertTrue(all(m["status"] == "development" for m in dev_models))
        
        # List only production models
        prod_models = self.registry.list_models(status="production")
        self.assertEqual(len(prod_models), 1)
        self.assertEqual(prod_models[0]["model_name"], "ProdModel")

    def test_get_active_model_config(self):
        """Test getting the configuration for the active model."""
        # Register a model and set as default
        model_id = self.registry.register_model(
            model_name=self.test_model["model_name"],
            description=self.test_model["description"],
            version=self.test_model["version"],
            configuration=self.test_model["configuration"],
            set_as_default=True
        )
        
        # Get active model config
        config = self.registry.get_active_model_config()
        
        # Check that the config matches the model
        self.assertEqual(config["model_name"], self.test_model["model_name"])
        self.assertEqual(config["use_flash_attention"], self.test_model["configuration"]["use_flash_attention"])
        self.assertEqual(config["device_map"], self.test_model["configuration"]["device_map"])

    def test_set_active_model(self):
        """Test setting the active model."""
        # Register two models
        model_id1 = self.registry.register_model(
            model_name="Model1",
            description="Model 1",
            version="1.0"
        )
        
        model_id2 = self.registry.register_model(
            model_name="Model2",
            description="Model 2",
            version="1.0"
        )
        
        # Set first model as active
        success = self.registry.set_active_model(model_id1)
        self.assertTrue(success)
        self.assertEqual(self.registry.active_model_id, model_id1)
        
        # Set second model as active
        success = self.registry.set_active_model(model_id2)
        self.assertTrue(success)
        self.assertEqual(self.registry.active_model_id, model_id2)
        
        # Try to set a non-existent model as active
        success = self.registry.set_active_model("non_existent_id")
        self.assertFalse(success)
        self.assertEqual(self.registry.active_model_id, model_id2)  # Should not change

    @patch('OCR.core.registry.ModelRegistry.get_model_performance_history')
    def test_log_model_performance(self, mock_get_history):
        """Test logging model performance."""
        # Mock the get_performance_history method
        mock_get_history.return_value = []
        
        # Register a model
        model_id = self.registry.register_model(
            model_name=self.test_model["model_name"],
            description=self.test_model["description"],
            version=self.test_model["version"]
        )
        
        # Log performance
        metrics = {"accuracy": 0.98, "processing_time": 0.75}
        success = self.registry.log_model_performance(
            model_id=model_id,
            metrics=metrics,
            sample_id="test_sample_001"
        )
        
        # Check that the operation was successful
        self.assertTrue(success)
        
        # Check that the performance log file was created
        performance_log_path = os.path.join(
            os.path.dirname(self.registry_path), 
            "performance_log.json"
        )
        self.assertTrue(os.path.exists(performance_log_path))
        
        # Check the content of the performance log
        with open(performance_log_path, 'r') as f:
            log_data = json.load(f)
            
        self.assertIn("entries", log_data)
        self.assertEqual(len(log_data["entries"]), 1)
        self.assertEqual(log_data["entries"][0]["model_id"], model_id)
        self.assertEqual(log_data["entries"][0]["metrics"]["accuracy"], 0.98)


if __name__ == '__main__':
    unittest.main()
