import unittest
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OCR.core.extractor import VehicleDataExtractor


class TestVehicleDataExtractor(unittest.TestCase):
    """Test cases for the VehicleDataExtractor class."""

    def setUp(self):
        """Set up test cases."""
        self.extractor = VehicleDataExtractor()
        
        # Sample data for testing
        self.valid_json_string = '{"vehicle": {"make": "Toyota", "model": "Camry"}}'
        self.invalid_json_string = '{"vehicle": {"make": "Toyota", "model": "Camry"'
        self.text_with_json = f"Some text before {self.valid_json_string} and some text after"
        
        # Sample card data for merging
        self.front_data = {
            "vehicle": {
                "make": "Toyota",
                "model": "Camry",
                "year": "2020"
            },
            "owner": {
                "name": "John Doe"
            }
        }
        
        self.back_data = {
            "vehicle": {
                "vin": "1HGCM82633A123456"
            },
            "registration": {
                "expiry_date": "2025-01-01"
            }
        }

    def test_extract_json_valid(self):
        """Test extracting valid JSON from text."""
        result = self.extractor.extract_json(self.valid_json_string)
        self.assertEqual(result, json.loads(self.valid_json_string))
        self.assertTrue(self.extractor.is_valid_json(result))

    def test_extract_json_from_text(self):
        """Test extracting JSON embedded in text."""
        result = self.extractor.extract_json(self.text_with_json)
        self.assertEqual(result, json.loads(self.valid_json_string))
        self.assertTrue(self.extractor.is_valid_json(result))

    def test_extract_json_invalid(self):
        """Test extracting invalid JSON."""
        result = self.extractor.extract_json(self.invalid_json_string)
        self.assertIn("error", result)
        self.assertFalse(self.extractor.is_valid_json(result))

    def test_merge_card_data(self):
        """Test merging data from front and back sides of a card."""
        merged = self.extractor.merge_card_data(self.front_data, self.back_data)
        
        # Check that data from both sides is present
        self.assertEqual(merged["vehicle"]["make"], "Toyota")
        self.assertEqual(merged["vehicle"]["model"], "Camry")
        self.assertEqual(merged["vehicle"]["vin"], "1HGCM82633A123456")
        self.assertEqual(merged["registration"]["expiry_date"], "2025-01-01")
        
        # Check that metadata is set correctly
        self.assertTrue(merged["metadata"]["front_processed"])
        self.assertTrue(merged["metadata"]["back_processed"])

    def test_merge_card_data_no_back(self):
        """Test merging when only front data is available."""
        merged = self.extractor.merge_card_data(self.front_data, {})
        
        # Check that front data is preserved
        self.assertEqual(merged["vehicle"]["make"], "Toyota")
        self.assertEqual(merged["owner"]["name"], "John Doe")

    def test_process_extracted_json(self):
        """Test processing extracted JSON with validation."""
        valid_json = json.loads(self.valid_json_string)
        result = self.extractor.process_extracted_json(valid_json)
        
        # Check that validation status is added
        self.assertIn("validation", result)
        self.assertTrue(result["validation"]["is_valid_format"])
        self.assertIn("format_validation_time", result["validation"])

    def test_is_valid_json(self):
        """Test JSON validation function."""
        # Valid JSON (no error key)
        valid_json = {"data": "value"}
        self.assertTrue(self.extractor.is_valid_json(valid_json))
        
        # Invalid JSON (has error key)
        invalid_json = {"error": "Invalid JSON format"}
        self.assertFalse(self.extractor.is_valid_json(invalid_json))


if __name__ == '__main__':
    unittest.main()
