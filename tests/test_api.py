import unittest
import json
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from OCR.core.qwen_service import QwenVisionService


class TestQwenVisionService(unittest.TestCase):
    """Test cases for the QwenVisionService class."""

    @patch('transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained')
    @patch('transformers.AutoProcessor.from_pretrained')
    def setUp(self, mock_processor, mock_model):
        """Set up test fixtures."""
        # Mock the model and processor
        self.mock_model = mock_model.return_value
        self.mock_processor = mock_processor.return_value
        
        # Initialize service with mocked components
        self.service = QwenVisionService(
            model_name="test_model",
            device_map="cpu",
            use_flash_attention=False
        )
        
        # Sample data for testing
        self.test_image_path = "/path/to/test_image.jpg"
        self.test_prompt = "Extract information from this image"
        
        # Sample model output for testing
        self.model_output_with_json = "Here is the extracted data: {\"vehicle\": {\"make\": \"Toyota\", \"model\": \"Camry\"}}"
        self.model_output_no_json = "Could not extract any structured data from this image."
        
        # Sample front and back data
        self.front_data = {
            "vehicle": {"make": "Toyota", "model": "Camry"},
            "card_side": "front"
        }
        
        self.back_data = {
            "registration": {"expiry_date": "2025-01-01"},
            "card_side": "back"
        }

    @patch('torch.cuda.is_available', return_value=False)
    def test_init_cpu(self, mock_cuda):
        """Test initialization with CPU device."""
        service = QwenVisionService(device_map="cpu")
        self.assertEqual(service.device_map, "cpu")
        self.assertEqual(service.model_name, "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")

    @patch('torch.cuda.is_available', return_value=True)
    def test_init_gpu(self, mock_cuda):
        """Test initialization with GPU device and flash attention."""
        with patch('transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained') as mock_model:
            service = QwenVisionService(use_flash_attention=True)
            mock_model.assert_called_with(
                "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                torch_dtype=unittest.mock.ANY,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )

    @patch('OCR.core.qwen_service.process_vision_info', return_value=([], []))
    @patch('torch.no_grad')
    def test_process_image(self, mock_no_grad, mock_process_vision):
        """Test processing an image with a text prompt."""
        # Configure mocks
        context_manager_mock = MagicMock()
        mock_no_grad.return_value = context_manager_mock
        context_manager_mock.__enter__.return_value = None
        
        # Mock processor response
        self.mock_processor.apply_chat_template.return_value = "processed_template"
        self.mock_processor.return_value = MagicMock(input_ids=MagicMock())
        
        # Mock model generate method
        mock_generated_ids = MagicMock()
        self.mock_model.generate.return_value = mock_generated_ids
        
        # Mock batch_decode to return our test output
        self.mock_processor.batch_decode.return_value = [self.model_output_with_json]
        
        # Call the method
        result = self.service.process_image(
            self.test_image_path, 
            self.test_prompt
        )
        
        # Verify the result
        self.assertEqual(result, self.model_output_with_json)
        
        # Verify the mocks were called correctly
        self.mock_processor.apply_chat_template.assert_called_once()
        mock_process_vision.assert_called_once()
        self.mock_model.generate.assert_called_once()
        self.mock_processor.batch_decode.assert_called_once()

    def test_extract_json(self):
        """Test extracting JSON from model output text."""
        # Valid JSON in text
        valid_result = self.service._extract_json(self.model_output_with_json)
        self.assertEqual(valid_result, {"vehicle": {"make": "Toyota", "model": "Camry"}})
        
        # No JSON in text
        no_json_result = self.service._extract_json(self.model_output_no_json)
        self.assertIn("raw_text", no_json_result)
        self.assertIn("parse_error", no_json_result)

    @patch('OCR.core.qwen_service.QwenVisionService.process_image')
    def test_extract_vehicle_info_front_only(self, mock_process_image):
        """Test extracting vehicle info from front image only."""
        # Mock the process_image method to return our test output
        mock_process_image.return_value = self.model_output_with_json
        
        # Call the method with front image only
        result = self.service.extract_vehicle_info(self.test_image_path)
        
        # Verify the result
        self.assertEqual(result["vehicle"]["make"], "Toyota")
        self.assertEqual(result["card_side"], "front")
        self.assertTrue(result["metadata"]["front_image_processed"])
        self.assertFalse(result["metadata"]["back_image_processed"])
        
        # Verify process_image was called once
        mock_process_image.assert_called_once()

    @patch('OCR.core.qwen_service.QwenVisionService.process_image')
    def test_extract_vehicle_info_both_sides(self, mock_process_image):
        """Test extracting vehicle info from both front and back images."""
        # Mock the process_image method to return different outputs for each call
        mock_process_image.side_effect = [
            json.dumps(self.front_data),
            json.dumps(self.back_data)
        ]
        
        # Call the method with both images
        result = self.service.extract_vehicle_info(
            self.test_image_path, 
            back_image_path=self.test_image_path
        )
        
        # Verify the combined result
        self.assertEqual(result["vehicle"]["make"], "Toyota")
        self.assertEqual(result["registration"]["expiry_date"], "2025-01-01")
        self.assertTrue(result["metadata"]["front_image_processed"])
        self.assertTrue(result["metadata"]["back_image_processed"])
        
        # Verify process_image was called twice
        self.assertEqual(mock_process_image.call_count, 2)

    def test_merge_card_data(self):
        """Test merging data from front and back sides of a card."""
        # Test merging with different keys
        merged = self.service._merge_card_data(
            {"field1": "value1", "card_side": "front"},
            {"field2": "value2", "card_side": "back"}
        )
        self.assertEqual(merged["field1"], "value1")
        self.assertEqual(merged["field2"], "value2")
        
        # Test merging with overlapping dictionary fields
        merged = self.service._merge_card_data(
            {"vehicle": {"make": "Toyota"}, "card_side": "front"},
            {"vehicle": {"model": "Camry"}, "card_side": "back"}
        )
        self.assertEqual(merged["vehicle"]["make"], "Toyota")
        self.assertEqual(merged["vehicle"]["model"], "Camry")
        
        # Test merging with conflicting non-dictionary fields
        merged = self.service._merge_card_data(
            {"field": "front_value", "card_side": "front"},
            {"field": "back_value", "card_side": "back"}
        )
        self.assertEqual(merged["field_front"], "front_value")
        self.assertEqual(merged["field_back"], "back_value")


if __name__ == '__main__':
    unittest.main()
