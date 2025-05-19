import unittest
import os
import json
import sys
import tempfile
from unittest.mock import patch, MagicMock, mock_open

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OCR.core.qwen_service import QwenVisionService
from OCR.core.extractor import VehicleDataExtractor
from OCR.core.registry import ModelRegistry


class TestIntegration(unittest.TestCase):
    """Integration tests for the OCR system components."""

    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Use a temporary registry file
        self.registry_path = os.path.join(self.temp_dir, "test_registry.json")
        
        # Set up mock paths for test images
        self.front_image_path = os.path.join(self.temp_dir, "test_front.jpg")
        self.back_image_path = os.path.join(self.temp_dir, "test_back.jpg")
        
        # Create empty files for testing
        with open(self.front_image_path, 'w') as f:
            f.write("")
        with open(self.back_image_path, 'w') as f:
            f.write("")
        
        # Sample response data
        self.front_response = """
        Here is the extracted information from the front side:
        {
            "vehicle": {
                "make": "Toyota",
                "model": "Camry",
                "year": "2020",
                "color": "Blue"
            },
            "owner": {
                "name": "John Doe",
                "address": "123 Main St"
            }
        }
        """
        
        self.back_response = """
        The back side contains the following information:
        {
            "vehicle": {
                "vin": "1HGCM82633A123456"
            },
            "registration": {
                "expiry_date": "2025-01-01",
                "issued_date": "2023-01-01"
            }
        }
        """

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    @patch('OCR.core.registry.ModelRegistry._load_registry')
    @patch('transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    @patch('torch.cuda.is_available', return_value=False)
    @patch('OCR.core.qwen_service.QwenVisionService.process_image')
    def test_end_to_end_extraction(self, mock_process_image, mock_cuda, 
                                   mock_processor, mock_model, mock_load_registry):
        """Test the end-to-end extraction process from images to structured data."""
        # Set up registry mock
        mock_load_registry.return_value = {
            "models": {
                "test-model": {
                    "model_id": "test-model",
                    "model_name": "test_model",
                    "description": "Test model",
                    "version": "1.0",
                    "status": "production",
                    "configuration": {
                        "use_flash_attention": False,
                        "device_map": "cpu"
                    }
                }
            },
            "default_model": "test-model",
            "last_updated": "2023-01-01T00:00:00"
        }
        
        # Set up process_image mock
        mock_process_image.side_effect = [
            self.front_response,
            self.back_response
        ]
        
        # Create the components
        registry = ModelRegistry(registry_path=self.registry_path)
        service = QwenVisionService(device_map="cpu")
        extractor = VehicleDataExtractor()
        
        # Process the front image
        front_text = service.process_image(self.front_image_path, "Extract vehicle information")
        front_json = extractor.extract_json(front_text)
        
        # Process the back image
        back_text = service.process_image(self.back_image_path, "Extract vehicle information")
        back_json = extractor.extract_json(back_text)
        
        # Merge the data
        merged_data = extractor.merge_card_data(front_json, back_json)
        
        # Verify the results
        self.assertEqual(merged_data["vehicle"]["make"], "Toyota")
        self.assertEqual(merged_data["vehicle"]["vin"], "1HGCM82633A123456")
        self.assertEqual(merged_data["registration"]["expiry_date"], "2025-01-01")
        self.assertEqual(merged_data["owner"]["name"], "John Doe")
        
        # Verify process_image was called correctly
        self.assertEqual(mock_process_image.call_count, 2)
        mock_process_image.assert_any_call(self.front_image_path, "Extract vehicle information")
        mock_process_image.assert_any_call(self.back_image_path, "Extract vehicle information")

    @patch('OCR.core.registry.ModelRegistry._load_registry')
    @patch('transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    @patch('OCR.core.qwen_service.QwenVisionService.process_image')
    def test_qwen_service_with_vehicle_extractor(self, mock_process_image, 
                                                mock_processor, mock_model, mock_load_registry):
        """Test integration between QwenVisionService and VehicleDataExtractor."""
        # Set up registry mock
        mock_load_registry.return_value = {
            "models": {"test-model": {}},
            "default_model": "test-model"
        }
        
        # Configure process_image to return our test responses
        mock_process_image.return_value = self.front_response
        
        # Create service and extractor
        service = QwenVisionService()
        extractor = VehicleDataExtractor()
        
        # Get the model output
        output = service.process_image(self.front_image_path, "Extract information")
        
        # Extract JSON from the output
        extracted_json = extractor.extract_json(output)
        
        # Validate the extracted data
        self.assertEqual(extracted_json["vehicle"]["make"], "Toyota")
        self.assertEqual(extracted_json["vehicle"]["model"], "Camry")
        self.assertEqual(extracted_json["owner"]["name"], "John Doe")
        
        # Process the extracted JSON
        processed_data = extractor.process_extracted_json(extracted_json)
        
        # Verify validation fields were added
        self.assertIn("validation", processed_data)
        self.assertTrue(processed_data["validation"]["is_valid_format"])

    @patch('OCR.core.registry.ModelRegistry._load_registry')
    @patch('OCR.core.registry.ModelRegistry._save_registry')
    def test_registry_integration_with_model_service(self, mock_save, mock_load):
        """Test integration between ModelRegistry and QwenVisionService."""
        # Set up registry mock
        mock_load.return_value = {
            "models": {
                "model1": {
                    "model_id": "model1",
                    "model_name": "TestModel/Model1",
                    "description": "Test model 1",
                    "version": "1.0",
                    "status": "production",
                    "configuration": {
                        "use_flash_attention": False,
                        "device_map": "cpu"
                    }
                },
                "model2": {
                    "model_id": "model2",
                    "model_name": "TestModel/Model2",
                    "description": "Test model 2",
                    "version": "2.0",
                    "status": "development",
                    "configuration": {
                        "use_flash_attention": True,
                        "device_map": "cuda"
                    }
                }
            },
            "default_model": "model1",
            "last_updated": "2023-01-01T00:00:00"
        }
        
        # Create a registry instance
        registry = ModelRegistry(registry_path=self.registry_path)
        
        # Test getting default model config
        default_config = registry.get_active_model_config()
        self.assertEqual(default_config["model_name"], "TestModel/Model1")
        self.assertEqual(default_config["device_map"], "cpu")
        self.assertEqual(default_config["use_flash_attention"], False)
        
        # Change active model
        registry.set_active_model("model2")
        
        # Get updated config
        new_config = registry.get_active_model_config()
        self.assertEqual(new_config["model_name"], "TestModel/Model2")
        self.assertEqual(new_config["device_map"], "cuda")


if __name__ == '__main__':
    unittest.main()
