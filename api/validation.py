"""
Validation module for document extraction results.

Provides validation for different document types including vehicle registration cards
and ID cards, ensuring that the extracted data meets quality standards.
"""

import logging
import re
from typing import Dict, Any, List, Optional

from utils.config import Config

logger = logging.getLogger(__name__)

class ResultValidator:
    """
    Validator for document extraction results.
    Validates extracted information based on document type.
    """
    
    def __init__(self, min_confidence: float = 0.7):
        """
        Initialize the validator.
        
        Args:
            min_confidence: Minimum confidence threshold for validation
        """
        self.min_confidence = min_confidence
        logger.info(f"Initialized validator with min_confidence={min_confidence}")
    
    def validate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate extraction results based on document type.
        
        Args:
            result: Extraction result to validate
            
        Returns:
            Validation result with details
        """
        # Determine document type
        document_type = result.get("document_type", "unknown")
        
        # Select appropriate validation method based on document type
        if document_type == "vehicle_registration":
            return self._validate_vehicle_registration(result)
        elif document_type in ["id_card_front", "id_card_back", "id_card"]:
            return self._validate_id_card(result)
        else:
            logger.warning(f"Unknown document type for validation: {document_type}")
            return {
                "valid": False,
                "confidence": 0.0,
                "errors": [f"Unknown document type: {document_type}"],
                "warnings": []
            }
    
    def _validate_vehicle_registration(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate vehicle registration card extraction results.
        
        Args:
            result: Extraction result to validate
            
        Returns:
            Validation result with details
        """
        errors = []
        warnings = []
        validation_checks = []
        
        # Check for required sections
        required_sections = ["vehicle_info", "owner_info", "registration_info"]
        for section in required_sections:
            if section not in result:
                errors.append(f"Missing required section: {section}")
            elif not isinstance(result[section], dict):
                errors.append(f"Section {section} is not a dictionary")
        
        # If there are errors in required sections, return early
        if errors:
            return {
                "valid": False,
                "confidence": 0.0,
                "errors": errors,
                "warnings": warnings
            }
        
        # Check vehicle info fields
        vehicle_info = result.get("vehicle_info", {})
        required_vehicle_fields = ["make", "plate_number", "chassis_number"]
        for field in required_vehicle_fields:
            if field not in vehicle_info or not vehicle_info[field]:
                errors.append(f"Missing required vehicle field: {field}")
            else:
                validation_checks.append(1.0)
        
        # Check owner info fields
        owner_info = result.get("owner_info", {})
        required_owner_fields = ["name", "id"]
        for field in required_owner_fields:
            if field not in owner_info or not owner_info[field]:
                errors.append(f"Missing required owner field: {field}")
            else:
                validation_checks.append(1.0)
        
        # Check registration info fields
        registration_info = result.get("registration_info", {})
        required_registration_fields = ["expiry_date"]
        for field in required_registration_fields:
            if field not in registration_info or not registration_info[field]:
                errors.append(f"Missing required registration field: {field}")
            else:
                validation_checks.append(1.0)
        
        # Calculate overall confidence score
        confidence = sum(validation_checks) / (len(required_vehicle_fields) + 
                                               len(required_owner_fields) + 
                                               len(required_registration_fields)) if validation_checks else 0.0
        
        return {
            "valid": len(errors) == 0 and confidence >= self.min_confidence,
            "confidence": confidence,
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_id_card(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ID card extraction results.
        
        Args:
            result: Extraction result to validate
            
        Returns:
            Validation result with details
        """
        errors = []
        warnings = []
        validation_checks = []
        
        # Check for document info section
        if "document_info" not in result:
            errors.append("Missing required section: document_info")
        elif not isinstance(result["document_info"], dict):
            errors.append("Section document_info is not a dictionary")
            
        # Check for personal info section for front side
        if "personal_info" not in result and "additional_info" not in result:
            errors.append("Missing required section: personal_info or additional_info")
            
        # If there are errors in required sections, return early
        if errors:
            return {
                "valid": False,
                "confidence": 0.0,
                "errors": errors,
                "warnings": warnings
            }
        
        # Check document info fields
        document_info = result.get("document_info", {})
        required_document_fields = ["document_type", "issuing_authority"]
        for field in required_document_fields:
            if field not in document_info or not document_info[field]:
                warnings.append(f"Missing document field: {field}")
            else:
                validation_checks.append(1.0)
        
        # Check personal info fields if present (front side)
        personal_info = result.get("personal_info", {})
        if personal_info:
            required_personal_fields = ["id_number", "name", "nationality"]
            for field in required_personal_fields:
                if field not in personal_info or not personal_info[field]:
                    warnings.append(f"Missing personal info field: {field}")
                else:
                    validation_checks.append(1.0)
        
        # Check additional info fields if present (back side)
        additional_info = result.get("additional_info", {})
        if additional_info:
            required_additional_fields = ["passport_number"]
            for field in required_additional_fields:
                if field not in additional_info or not additional_info[field]:
                    warnings.append(f"Missing additional info field: {field}")
                else:
                    validation_checks.append(1.0)
        
        # Calculate overall confidence score
        total_checks = len(required_document_fields)
        if personal_info:
            total_checks += len(["id_number", "name", "nationality"])
        if additional_info:
            total_checks += len(["passport_number"])
            
        confidence = sum(validation_checks) / total_checks if validation_checks and total_checks > 0 else 0.0
        
        return {
            "valid": len(errors) == 0 and confidence >= self.min_confidence,
            "confidence": confidence,
            "errors": errors,
            "warnings": warnings
        }
    
    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """
        Validate a date string format.
        Supports multiple date formats (YYYY-MM-DD, DD/MM/YYYY, etc.)
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if valid date format, False otherwise
        """
        if not date_str:
            return False
            
        # Common date formats
        date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',                # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',                # DD/MM/YYYY
            r'^\d{2}-\d{2}-\d{4}$',                # DD-MM-YYYY
            r'^\d{1,2}/\d{1,2}/\d{2,4}$',          # D/M/YY or D/M/YYYY
            r'^\d{1,2}-\d{1,2}-\d{2,4}$',          # D-M-YY or D-M-YYYY
            r'^\d{4}/\d{1,2}/\d{1,2}$',            # YYYY/M/D
        ]
        
        return any(re.match(pattern, date_str) for pattern in date_patterns)
    
    @staticmethod
    def validate_numeric(value: str) -> bool:
        """
        Validate that a value contains only digits.
        
        Args:
            value: String to validate
            
        Returns:
            True if valid numeric value, False otherwise
        """
        if not value:
            return False
            
        return re.match(r'^\d+$', value) is not None
