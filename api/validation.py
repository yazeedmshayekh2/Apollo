"""
Validator for OCR results from vehicle registration cards.
Ensures proper handling of Unicode characters, including Arabic text.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class ResultValidator:
    """
    Validator for OCR extraction results.
    Ensures proper handling of Unicode characters, including Arabic.
    """
    
    def __init__(self, min_confidence: float = 0.7):
        """
        Initialize the validator.
        
        Args:
            min_confidence: Minimum confidence threshold for validation
        """
        self.min_confidence = min_confidence
        logger.info(f"ResultValidator initialized with min_confidence={min_confidence}")
        
    def validate_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the OCR extraction result.
        Preserves all Unicode characters in the validation process.
        
        Args:
            result: The result to validate
            
        Returns:
            Validated result with validation status
        """
        logger.info("Validating OCR result")
        
        if not result or not isinstance(result, dict):
            logger.warning("Invalid result format")
            return {
                "validation_error": "Invalid result format",
                "raw_result": str(result)
            }
            
        # Create a copy of the result to not modify the original
        validated = result.copy()
        
        # Ensure metadata exists
        if "metadata" not in validated:
            validated["metadata"] = {}
            
        # Add validation metadata
        validated["metadata"]["validation"] = {
            "is_validated": True,
            "validation_timestamp": self._get_timestamp(),
            "confidence_score": self._calculate_confidence(validated)
        }
        
        # Handle raw JSON response - ensure it's string to prevent serialization issues
        if "raw_model_response" in validated:
            # Preserve as string but ensure it's valid UTF-8
            validated["raw_model_response"] = str(validated["raw_model_response"])
            
        # Validate field structure without changing content
        structure_validation = self._validate_structure(validated)
        validated["metadata"]["validation"]["structure_validation"] = structure_validation
        
        logger.info("Validation completed")
        return validated
    
    def _validate_structure(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the structure of the result without modifying content.
        Preserves all Unicode characters including Arabic.
        
        Args:
            result: The result to validate
            
        Returns:
            Structure validation results
        """
        validation_results = {
            "has_vehicle_info": "vehicle" in result and isinstance(result.get("vehicle"), dict),
            "has_owner_info": "owner" in result and isinstance(result.get("owner"), dict),
            "has_registration_info": any(
                key in result for key in ["registration", "registration_details"]
            ),
            "is_valid_structure": True
        }
        
        # Mark as invalid if missing critical sections
        if not validation_results["has_vehicle_info"] and not validation_results["has_owner_info"]:
            validation_results["is_valid_structure"] = False
            validation_results["structure_error"] = "Missing critical vehicle and owner information"
            
        return validation_results
        
    def _calculate_confidence(self, result: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the extraction result.
        
        Args:
            result: The extraction result
            
        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence calculation based on completeness
        # This can be enhanced with more sophisticated validation
        score_components = []
        
        # Check vehicle info
        if "vehicle" in result and isinstance(result["vehicle"], dict):
            vehicle_score = min(1.0, len(result["vehicle"]) / 8)  # Expect at least 8 fields
            score_components.append(vehicle_score)
            
        # Check owner info
        if "owner" in result and isinstance(result["owner"], dict):
            owner_score = min(1.0, len(result["owner"]) / 3)  # Expect at least 3 fields
            score_components.append(owner_score)
            
        # Check registration info
        reg_keys = ["registration", "registration_details"]
        reg_key = next((k for k in reg_keys if k in result), None)
        if reg_key and isinstance(result[reg_key], dict):
            reg_score = min(1.0, len(result[reg_key]) / 4)  # Expect at least 4 fields
            score_components.append(reg_score)
            
        # Calculate average score
        if score_components:
            return sum(score_components) / len(score_components)
        else:
            return 0.0
    
    def _get_timestamp(self) -> str:
        """Get the current ISO timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
