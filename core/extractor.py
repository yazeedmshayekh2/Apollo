import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class VehicleDataExtractor:
    """
    Specialized extractor for vehicle registration cards.
    Validates JSON format and extracts structured data without validating specific field values.
    Supports Unicode and Arabic character extraction.
    """
    
    def __init__(self):
        """Initialize the vehicle data extractor."""
        logger.info("Vehicle data extractor initialized")

    def extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract JSON from text, identifying its beginning and end.
        Only validates the JSON format without checking field values.
        Preserves Unicode characters including Arabic.
        
        Args:
            text: The text output potentially containing JSON
            
        Returns:
            Extracted JSON as a dictionary or error information
        """
        logger.info("Extracting JSON from text")
        
        # Clean the text - remove markdown code block markers
        text = re.sub(r'```(json)?|```', '', text)
        
        # Clean any potential leading/trailing whitespace
        text = text.strip()
        
        # Try different approaches for extracting JSON
        
        # Approach 1: Find JSON using braces
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            try:
                # Extract the JSON substring
                json_str = text[json_start:json_end]
                logger.debug(f"Found JSON between positions {json_start} and {json_end}")
                
                # Parse the JSON with Unicode support
                parsed_json = json.loads(json_str)
                logger.info("Successfully parsed JSON")
                return parsed_json
                
            except json.JSONDecodeError as e:
                logger.warning(f"First JSON extraction attempt failed: {e}")
                # Continue to next approach
        
        # Approach 2: Fix common JSON syntax issues and try again
        try:
            # Find potential JSON section
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                
                # Log the problematic JSON for debugging
                logger.debug(f"Attempting to fix JSON: {json_str[:200]}...")
                
                # Fix common JSON syntax errors
                # 1. Fix missing quotes around keys
                json_str = re.sub(r'([{,])\s*([a-zA-Z0-9_]+)\s*:', r'\1"\2":', json_str)
                # 2. Fix missing quotes around string values
                json_str = re.sub(r':\s*([a-zA-Z0-9_]+)([,}])', r':"\1"\2', json_str)
                # 3. Remove any trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                # 4. Ensure proper newline handling
                json_str = json_str.replace('\n', ' ').replace('\r', ' ')
                
                parsed_json = json.loads(json_str)
                logger.info("Successfully parsed JSON after syntax fixes")
                return parsed_json
        except json.JSONDecodeError as e:
            logger.warning(f"Second JSON extraction attempt failed: {e}")
            # Continue to next approach
        
        # Approach 3: Try to reconstruct a valid JSON object from key-value pairs
        try:
            # Extract key-value patterns that might form a valid JSON
            # Look for patterns like "key": "value" or "key": value
            kv_pattern = r'["\'"]?([a-zA-Z0-9_]+)["\'"]?\s*:\s*["\'"]?((?:[^"\'",{}]+)|(?:"[^"]*")|(?:\'[^\']*\'))["\'"]?'
            matches = re.findall(kv_pattern, text)
            
            if matches:
                # Create a valid JSON from extracted key-value pairs
                reconstructed_json = "{"
                for i, (key, value) in enumerate(matches):
                    if i > 0:
                        reconstructed_json += ","
                    
                    # Ensure key has quotes
                    if not (key.startswith('"') and key.endswith('"')):
                        key = f'"{key}"'
                        
                    # Ensure value has appropriate quotes for string values
                    if not (value.startswith('"') and value.endswith('"')) and not value.replace('.', '').isdigit() and value.lower() not in ('true', 'false', 'null'):
                        value = f'"{value}"'
                        
                    reconstructed_json += f"{key}:{value}"
                reconstructed_json += "}"
                
                parsed_json = json.loads(reconstructed_json)
                logger.info("Successfully reconstructed JSON from key-value pairs")
                return parsed_json
        except Exception as e:
            logger.warning(f"Key-value reconstruction failed: {e}")
        
        # Final fallback: If all else fails, try to extract any meaningful data
        try:
            # Look for field-value patterns in a more relaxed way
            patterns = {
                'make': r'[Mm]ake:\s*([A-Za-z0-9\s]+)',
                'model': r'[Mm]odel:\s*([A-Za-z0-9\s]+)',
                'year': r'[Yy]ear:\s*(\d{4})',
                'vin': r'(?:[Vv][Ii][Nn]|[Vv]ehicle\s*[Ii]dentification\s*[Nn]umber):\s*([A-Z0-9]+)',
                'plate': r'(?:[Pp]late|[Ll]icense):\s*([A-Z0-9]+)',
                'owner': r'[Oo]wner:\s*([A-Za-z\s]+)',
                'expiration': r'[Ee]xpiration:\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
            }
            
            minimal_json = {}
            for field, pattern in patterns.items():
                match = re.search(pattern, text)
                if match:
                    minimal_json[field] = match.group(1).strip()
            
            if minimal_json:
                logger.info("Created minimal JSON from text patterns")
                return minimal_json
        except Exception as e:
            logger.error(f"Pattern extraction failed: {e}")
        
        # If no extraction method worked, return error with the raw text
        logger.error("No valid JSON could be extracted from the text")
        return {
            "error": "No valid JSON format found",
            "raw_text": text[:500] + "..." if len(text) > 500 else text
        }
    
    def merge_card_data(self, front_data: Dict[str, Any], back_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge data from front and back sides of the card.
        Simple merge without validation of field values.
        
        Args:
            front_data: Data extracted from front side
            back_data: Data extracted from back side
            
        Returns:
            Merged data dictionary
        """
        logger.info("Merging data from front and back sides")
        
        # Create a copy of front data as the base
        merged = front_data.copy()
        
        if not back_data:
            return merged
        
        # For any top-level dictionaries, merge them
        for key, value in back_data.items():
            if key not in merged:
                # If key doesn't exist in merged, add it
                merged[key] = value
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # If both are dictionaries, merge them
                merged[key] = {**merged[key], **value}
            else:
                # Otherwise, keep both with side indicators
                merged[f"{key}_front"] = merged[key]
                merged[f"{key}_back"] = value
                merged.pop(key)
        
        # Add a flag indicating both sides were processed
        merged["metadata"] = merged.get("metadata", {})
        merged["metadata"]["front_processed"] = True
        merged["metadata"]["back_processed"] = True
        
        return merged
    
    def is_valid_json(self, json_obj: Dict[str, Any]) -> bool:
        """
        Check if the extracted JSON is valid (not an error response).
        
        Args:
            json_obj: The extracted JSON object
            
        Returns:
            True if valid JSON, False if it contains error indicators
        """
        return "error" not in json_obj
        
    def process_extracted_json(self, extracted_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the extracted JSON data by validating its format only.
        Does not validate specific field values.
        
        Args:
            extracted_json: The raw JSON data
            
        Returns:
            Processed data with validation status
        """
        # Add validation status
        result = extracted_json.copy()
        result["validation"] = {
            "is_valid_format": self.is_valid_json(extracted_json),
            "format_validation_time": self._get_timestamp()
        }
        
        return result
        
    def _get_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()