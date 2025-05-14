import unittest
import os
import sys
import tempfile
import json
import shutil
import uuid
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OCR.utils.config import Config
from OCR.utils.helpers import (
    generate_unique_id, 
    enhance_image, 
    save_json_result, 
    calculate_hash,
    validate_field,
    get_file_info
)


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""

    def setUp(self):
        """Set up test environment."""
        # Save original environment
        self.original_env = os.environ.copy()
        
        # Clear any existing OCR_ environment variables
        for key in list(os.environ.keys()):
            if key.startswith("OCR_"):
                del os.environ[key]

    def tearDown(self):
        """Clean up test environment."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_get_default_value(self):
        """Test getting default configuration values."""
        # Test getting a default value
        value = Config.get("MODEL_NAME")
        self.assertEqual(value, "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        
        # Test getting another default value
        value = Config.get("API_PORT")
        self.assertEqual(value, 8000)
        
        # Test getting a boolean default value
        value = Config.get("USE_FLASH_ATTENTION")
        self.assertTrue(value)

    def test_get_environment_value(self):
        """Test getting values from environment variables."""
        # Set environment variables
        os.environ["OCR_MODEL_NAME"] = "TestModel"
        os.environ["OCR_API_PORT"] = "9000"
        os.environ["OCR_USE_FLASH_ATTENTION"] = "false"
        
        # Test that environment values are used
        value = Config.get("MODEL_NAME")
        self.assertEqual(value, "TestModel")
        
        value = Config.get("API_PORT")
        self.assertEqual(value, 9000)  # Should be converted to int
        
        value = Config.get("USE_FLASH_ATTENTION")
        self.assertFalse(value)  # Should be converted to boolean

    def test_get_nonexistent_value(self):
        """Test getting a value that does not exist."""
        # Test with default argument
        value = Config.get("NONEXISTENT_KEY", "default_value")
        self.assertEqual(value, "default_value")
        
        # Test without default argument
        value = Config.get("NONEXISTENT_KEY")
        self.assertIsNone(value)

    def test_convert_value(self):
        """Test converting string values to appropriate types."""
        # Test boolean conversions
        self.assertTrue(Config._convert_value("true"))
        self.assertTrue(Config._convert_value("True"))
        self.assertTrue(Config._convert_value("YES"))
        self.assertTrue(Config._convert_value("1"))
        
        self.assertFalse(Config._convert_value("false"))
        self.assertFalse(Config._convert_value("False"))
        self.assertFalse(Config._convert_value("NO"))
        self.assertFalse(Config._convert_value("0"))
        
        # Test numeric conversions
        self.assertEqual(Config._convert_value("123"), 123)
        self.assertEqual(Config._convert_value("123.45"), 123.45)
        
        # Test string values
        self.assertEqual(Config._convert_value("hello"), "hello")
        self.assertEqual(Config._convert_value("123abc"), "123abc")

    def test_as_dict(self):
        """Test getting all configuration as a dictionary."""
        # Set a custom environment variable
        os.environ["OCR_CUSTOM_SETTING"] = "custom_value"
        
        # Get the configuration dictionary
        config_dict = Config.as_dict()
        
        # Check that it includes defaults
        self.assertEqual(config_dict["MODEL_NAME"], "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        self.assertEqual(config_dict["API_PORT"], 8000)
        
        # Check that it includes the custom setting
        self.assertEqual(config_dict["CUSTOM_SETTING"], "custom_value")

    def test_model_config(self):
        """Test getting model-specific configuration."""
        # Set environment variables
        os.environ["OCR_MODEL_NAME"] = "TestModel"
        os.environ["OCR_USE_FLASH_ATTENTION"] = "false"
        
        # Get model config
        model_config = Config.model_config()
        
        # Check values
        self.assertEqual(model_config["model_name"], "TestModel")
        self.assertFalse(model_config["use_flash_attention"])
        self.assertEqual(model_config["device_map"], "auto")  # Default value
        self.assertIn("min_pixels", model_config)
        self.assertIn("max_pixels", model_config)

    def test_processor_config(self):
        """Test getting processor-specific configuration."""
        # Set environment variables
        os.environ["OCR_ENHANCE_IMAGES"] = "false"
        os.environ["OCR_MIN_CONFIDENCE"] = "0.9"
        
        # Get processor config
        processor_config = Config.processor_config()
        
        # Check values
        self.assertFalse(processor_config["enhance_images"])
        self.assertEqual(processor_config["min_confidence"], 0.9)
        self.assertTrue(processor_config["validate_results"])  # Default value

    def test_api_config(self):
        """Test getting API-specific configuration."""
        # Set environment variables
        os.environ["OCR_API_HOST"] = "127.0.0.1"
        os.environ["OCR_API_PORT"] = "9000"
        
        # Get API config
        api_config = Config.api_config()
        
        # Check values
        self.assertEqual(api_config["host"], "127.0.0.1")
        self.assertEqual(api_config["port"], 9000)
        self.assertEqual(api_config["workers"], 4)  # Default value
        self.assertEqual(api_config["timeout"], 60)  # Default value

    def test_storage_config_local(self):
        """Test getting storage configuration for local storage."""
        # Set environment variables
        os.environ["OCR_STORAGE_TYPE"] = "local"
        os.environ["OCR_LOCAL_STORAGE_PATH"] = "/custom/path"
        
        # Get storage config
        storage_config = Config.storage_config()
        
        # Check values
        self.assertEqual(storage_config["type"], "local")
        self.assertEqual(storage_config["path"], "/custom/path")

    def test_storage_config_s3(self):
        """Test getting storage configuration for S3."""
        # Set environment variables
        os.environ["OCR_STORAGE_TYPE"] = "s3"
        os.environ["OCR_S3_BUCKET"] = "custom-bucket"
        
        # Get storage config
        storage_config = Config.storage_config()
        
        # Check values
        self.assertEqual(storage_config["type"], "s3")
        self.assertEqual(storage_config["bucket"], "custom-bucket")
        self.assertEqual(storage_config["prefix"], "images/")  # Default value


class TestHelpers(unittest.TestCase):
    """Test cases for the helper functions."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a sample test file
        self.test_file_path = os.path.join(self.temp_dir, "test_file.txt")
        with open(self.test_file_path, 'w') as f:
            f.write("This is a test file for helpers module testing.")
        
        # Create a sample test image
        self.test_image_path = os.path.join(self.temp_dir, "test_image.jpg")
        try:
            # Try to create a simple test image
            from PIL import Image
            img = Image.new('RGB', (100, 100), color = 'red')
            img.save(self.test_image_path)
        except ImportError:
            # If PIL is not available, create an empty file
            with open(self.test_image_path, 'w') as f:
                f.write("Dummy image content")
        
        # Sample OCR result
        self.test_result = {
            "vehicle": {
                "make": "Toyota",
                "model": "Camry",
                "year": "2020"
            },
            "owner": {
                "name": "John Doe"
            }
        }

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def test_generate_unique_id(self):
        """Test generating unique IDs."""
        # Test default prefix
        id1 = generate_unique_id()
        self.assertTrue(id1.startswith("ocr_"))
        self.assertEqual(len(id1), 4 + 1 + 32)  # prefix + underscore + uuid4 hex
        
        # Test custom prefix
        id2 = generate_unique_id("test")
        self.assertTrue(id2.startswith("test_"))
        self.assertEqual(len(id2), 5 + 1 + 32)  # custom prefix + underscore + uuid4 hex
        
        # Test uniqueness
        id3 = generate_unique_id()
        self.assertNotEqual(id1, id3)

    @patch('PIL.Image.open')
    @patch('PIL.ImageEnhance.Contrast')
    @patch('PIL.ImageEnhance.Sharpness')
    def test_enhance_image(self, mock_sharpness, mock_contrast, mock_open):
        """Test image enhancement function."""
        # Mock the image enhancement process
        mock_img = MagicMock()
        mock_open.return_value = mock_img
        
        mock_contrast_enhancer = MagicMock()
        mock_contrast.return_value = mock_contrast_enhancer
        mock_contrast_enhancer.enhance.return_value = mock_img
        
        mock_sharpness_enhancer = MagicMock()
        mock_sharpness.return_value = mock_sharpness_enhancer
        mock_sharpness_enhancer.enhance.return_value = mock_img
        
        # Test with default output path
        output_path = enhance_image(self.test_image_path)
        expected_path = os.path.join(self.temp_dir, "test_image_enhanced.jpg")
        self.assertEqual(output_path, expected_path)
        
        # Test with custom output path
        custom_path = os.path.join(self.temp_dir, "custom_output.jpg")
        output_path = enhance_image(self.test_image_path, custom_path)
        self.assertEqual(output_path, custom_path)
        
        # Verify the mocks were called correctly
        mock_open.assert_called_with(self.test_image_path)
        mock_contrast.assert_called_with(mock_img)
        mock_contrast_enhancer.enhance.assert_called_with(1.5)
        mock_sharpness.assert_called_with(mock_img)
        mock_sharpness_enhancer.enhance.assert_called_with(1.5)
        mock_img.save.assert_called()

    def test_save_json_result(self):
        """Test saving OCR results as JSON."""
        # Create output directory in temp folder
        output_dir = os.path.join(self.temp_dir, "output")
        
        # Test with default filename
        file_path = save_json_result(self.test_result, output_dir)
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.startswith(output_dir))
        self.assertTrue(file_path.endswith(".json"))
        
        # Verify the content of the saved file
        with open(file_path, 'r') as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data, self.test_result)
        
        # Test with custom filename without extension
        custom_filename = "custom_result"
        file_path = save_json_result(self.test_result, output_dir, custom_filename)
        expected_path = os.path.join(output_dir, f"{custom_filename}.json")
        self.assertEqual(file_path, expected_path)
        self.assertTrue(os.path.exists(file_path))
        
        # Test with custom filename with extension
        custom_filename = "custom_result2.json"
        file_path = save_json_result(self.test_result, output_dir, custom_filename)
        expected_path = os.path.join(output_dir, custom_filename)
        self.assertEqual(file_path, expected_path)
        self.assertTrue(os.path.exists(file_path))

    def test_calculate_hash(self):
        """Test file hash calculation."""
        # Calculate hash of test file
        file_hash = calculate_hash(self.test_file_path)
        
        # Verify the hash is a valid SHA-256 hash
        self.assertEqual(len(file_hash), 64)  # SHA-256 produces 64 hex characters
        self.assertTrue(all(c in "0123456789abcdef" for c in file_hash))
        
        # Create a copy of the file with different content
        modified_file_path = os.path.join(self.temp_dir, "modified_file.txt")
        with open(modified_file_path, 'w') as f:
            f.write("This is a modified test file with different content.")
        
        # Calculate hash of modified file
        modified_hash = calculate_hash(modified_file_path)
        
        # Verify the hashes are different
        self.assertNotEqual(file_hash, modified_hash)

    def test_validate_field(self):
        """Test field validation."""
        # Test validation of correct types
        is_valid, error = validate_field("string_field", "test", str)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        is_valid, error = validate_field("int_field", 123, int)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        is_valid, error = validate_field("dict_field", {"key": "value"}, dict)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")
        
        # Test validation of incorrect types
        is_valid, error = validate_field("int_field", "123", int)
        self.assertFalse(is_valid)
        self.assertIn("incorrect type", error)
        self.assertIn("int_field", error)
        self.assertIn("int", error)
        self.assertIn("str", error)

    def test_get_file_info(self):
        """Test getting file information."""
        # Get info for test file
        file_info = get_file_info(self.test_file_path)
        
        # Verify the info contains expected keys
        self.assertEqual(file_info["path"], self.test_file_path)
        self.assertIn("size", file_info)
        self.assertIn("created", file_info)
        self.assertIn("modified", file_info)
        self.assertIn("hash", file_info)
        
        # Verify the hash matches the calculated hash
        expected_hash = calculate_hash(self.test_file_path)
        self.assertEqual(file_info["hash"], expected_hash)
        
        # Test with non-existent file
        non_existent_path = os.path.join(self.temp_dir, "non_existent.txt")
        error_info = get_file_info(non_existent_path)
        self.assertEqual(error_info["path"], non_existent_path)
        self.assertIn("error", error_info)


if __name__ == '__main__':
    unittest.main() 