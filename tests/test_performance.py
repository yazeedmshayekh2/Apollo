import unittest
import os
import json
import sys
import time
import tempfile
from unittest.mock import patch, MagicMock

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OCR.core.extractor import VehicleDataExtractor
from OCR.core.registry import ModelRegistry


class TestPerformance(unittest.TestCase):
    """Performance tests for the OCR system components."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test registry path
        self.registry_path = os.path.join(self.temp_dir, "test_registry.json")
        
        # Create instances
        self.extractor = VehicleDataExtractor()
        
        # Test data
        self.test_json_small = json.dumps({
            "vehicle": {"make": "Toyota", "model": "Camry"},
            "owner": {"name": "John Doe"}
        })
        
        # Generate larger test data
        vehicle_data = {f"field_{i}": f"value_{i}" for i in range(50)}
        owner_data = {f"field_{i}": f"value_{i}" for i in range(50)}
        registration_data = {f"field_{i}": f"value_{i}" for i in range(50)}
        
        self.test_json_large = json.dumps({
            "vehicle": vehicle_data,
            "owner": owner_data,
            "registration": registration_data
        })
        
        # Create text with embedded JSON
        self.text_with_json = f"Some text before {self.test_json_small} and more text after"
        self.text_with_large_json = f"Some text before {self.test_json_large} and more text after"

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_json_extraction_performance(self):
        """Test JSON extraction performance with different input sizes."""
        # Small JSON performance
        start_time = time.time()
        for _ in range(100):  # Run 100 times
            self.extractor.extract_json(self.test_json_small)
        small_time = time.time() - start_time
        
        # JSON embedded in text performance
        start_time = time.time()
        for _ in range(100):  # Run 100 times
            self.extractor.extract_json(self.text_with_json)
        embedded_time = time.time() - start_time
        
        # Large JSON performance
        start_time = time.time()
        for _ in range(10):  # Run 10 times for larger input
            self.extractor.extract_json(self.test_json_large)
        large_time = time.time() - start_time
        
        # Large JSON embedded in text performance
        start_time = time.time()
        for _ in range(10):  # Run 10 times for larger input
            self.extractor.extract_json(self.text_with_large_json)
        embedded_large_time = time.time() - start_time
        
        # Print performance metrics
        print(f"\nJSON Extraction Performance:")
        print(f"Small JSON (100 runs): {small_time:.4f}s ({small_time/100:.4f}s per run)")
        print(f"Embedded JSON (100 runs): {embedded_time:.4f}s ({embedded_time/100:.4f}s per run)")
        print(f"Large JSON (10 runs): {large_time:.4f}s ({large_time/10:.4f}s per run)")
        print(f"Embedded Large JSON (10 runs): {embedded_large_time:.4f}s ({embedded_large_time/10:.4f}s per run)")
        
        # Verify performance is within acceptable ranges
        # These assertions check that extracting JSON from text is not significantly slower
        self.assertLess(embedded_time / small_time, 5.0, "Embedded JSON extraction is too slow compared to direct JSON")
        self.assertLess(embedded_large_time / large_time, 5.0, "Embedded large JSON extraction is too slow")

    def test_merge_card_data_performance(self):
        """Test performance of merging data from front and back cards."""
        # Parse test data
        front_data = json.loads(self.test_json_small)
        back_data = {"registration": {"expiry_date": "2025-01-01"}}
        
        # Small merge performance
        start_time = time.time()
        for _ in range(1000):  # Run 1000 times
            self.extractor.merge_card_data(front_data, back_data)
        small_merge_time = time.time() - start_time
        
        # Large merge performance
        large_front_data = json.loads(self.test_json_large)
        large_back_data = {
            "additional_info": {f"field_{i}": f"value_{i}" for i in range(50)},
            "notes": {f"note_{i}": f"content_{i}" for i in range(50)}
        }
        
        start_time = time.time()
        for _ in range(100):  # Run 100 times for larger data
            self.extractor.merge_card_data(large_front_data, large_back_data)
        large_merge_time = time.time() - start_time
        
        # Print performance metrics
        print(f"\nCard Data Merge Performance:")
        print(f"Small Merge (1000 runs): {small_merge_time:.4f}s ({small_merge_time/1000:.6f}s per run)")
        print(f"Large Merge (100 runs): {large_merge_time:.4f}s ({large_merge_time/100:.6f}s per run)")
        
        # Verify performance is acceptable
        # These are basic checks to ensure performance doesn't degrade significantly
        self.assertLess(small_merge_time / 1000, 0.001, "Small merge is too slow (>1ms per operation)")
        self.assertLess(large_merge_time / 100, 0.01, "Large merge is too slow (>10ms per operation)")

    @patch('OCR.core.registry.ModelRegistry._load_registry')
    @patch('OCR.core.registry.ModelRegistry._save_registry')
    def test_registry_operations_performance(self, mock_save, mock_load):
        """Test performance of registry operations."""
        # Set up mock registry
        mock_registry = {"models": {}, "default_model": None, "last_updated": ""}
        mock_load.return_value = mock_registry
        
        # Create registry
        registry = ModelRegistry(registry_path=self.registry_path)
        
        # Register model performance
        start_time = time.time()
        for i in range(100):
            registry.register_model(
                model_name=f"TestModel_{i}",
                description=f"Test model {i}",
                version=f"1.{i}"
            )
        register_time = time.time() - start_time
        
        # List models performance
        start_time = time.time()
        for _ in range(1000):
            registry.list_models()
        list_time = time.time() - start_time
        
        # Get model by name performance (worst case - scan all models)
        start_time = time.time()
        for i in range(100):
            registry.get_model_by_name(f"TestModel_{i % 100}")
        get_by_name_time = time.time() - start_time
        
        # Print performance metrics
        print(f"\nRegistry Operations Performance:")
        print(f"Register Models (100 runs): {register_time:.4f}s ({register_time/100:.4f}s per run)")
        print(f"List Models (1000 runs): {list_time:.4f}s ({list_time/1000:.6f}s per run)")
        print(f"Get Model by Name (100 runs): {get_by_name_time:.4f}s ({get_by_name_time/100:.6f}s per run)")
        
        # Verify performance is acceptable
        self.assertLess(list_time / 1000, 0.001, "Listing models is too slow (>1ms per operation)")
        self.assertLess(get_by_name_time / 100, 0.001, "Getting model by name is too slow (>1ms per operation)")


class TestScalability(unittest.TestCase):
    """Scalability tests for the OCR system components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test registry path
        self.registry_path = os.path.join(self.temp_dir, "test_registry.json")

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('OCR.core.registry.ModelRegistry._load_registry')
    @patch('OCR.core.registry.ModelRegistry._save_registry')
    def test_registry_scalability(self, mock_save, mock_load):
        """Test registry scalability with a large number of models."""
        # Set up mock registry
        mock_registry = {"models": {}, "default_model": None, "last_updated": ""}
        mock_load.return_value = mock_registry
        
        # Create registry
        registry = ModelRegistry(registry_path=self.registry_path)
        
        # Register many models
        model_count = 1000
        print(f"\nRegistering {model_count} models to test scalability...")
        
        start_time = time.time()
        for i in range(model_count):
            registry.register_model(
                model_name=f"ScalabilityTest_{i}",
                description=f"Scalability test model {i}",
                version=f"1.{i}"
            )
        register_time = time.time() - start_time
        
        # Perform operations on the large registry
        start_time = time.time()
        registry.list_models(limit=10)  # Get first 10 models
        registry.list_models(status="development", limit=10)  # With filter
        registry.get_model_by_name("ScalabilityTest_500")  # Get a specific model
        operations_time = time.time() - start_time
        
        # Print scalability metrics
        print(f"Registry Scalability with {model_count} models:")
        print(f"Registration Time: {register_time:.4f}s ({register_time/model_count:.6f}s per model)")
        print(f"Operations Time: {operations_time:.4f}s")
        
        # Verify scalability is acceptable
        self.assertLess(register_time / model_count, 0.01, "Model registration does not scale well")
        self.assertLess(operations_time, 0.1, "Registry operations are too slow with many models")


if __name__ == '__main__':
    unittest.main()
