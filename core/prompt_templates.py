"""
Prompt Templates for Document Processing

This module provides prompt templates for different document types
such as vehicle registration cards and ID cards (front and back).
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PromptTemplates:
    """
    Provides prompt templates for different document types.
    """
    
    @staticmethod
    def get_vehicle_registration_prompt() -> str:
        """
        Get the prompt template for vehicle registration cards.
        
        Returns:
            Prompt template string
        """
        return (
            """
            Extract all shown information from this vehicle registration card and return them as JSON."
            Output must be valid JSON only, no explanations.
            Output must be in the following format:
            {
            "vehicle_info": {
            "make": "The vehicle manufacturer (e.g., Nissan, Toyota)",
            "car_model": "The specific model of the vehicle",
            "body_type": "The type or category of vehicle (e.g., ستيشن واجن)",
            "plate_type": "Type of license plate (e.g., Private/خصوصي)",
            "plate_number": "The vehicle registration number",
            "year_of_manufacture": "Year the vehicle was manufactured",
            "country_of_manufacture": "Country where the vehicle was manufactured, it must be a country name (e.g., Jordan, Saudi Arabia, etc.)",
            "cylinders": "Number of cylinders in the engine",
            "seats": "Number of seats in the vehicle - it couldn't be a zero (it could be 001, 002, etc.)",
            "chassis_number": "Vehicle identification/chassis number",
            "engine_number": "Engine identification number",
            "first_registration_date": "Date of first registration if available"
            },
            "owner_info": {
                "name": "Full name of the vehicle owner",
                "id": "Owner ID number",
                "nationality": "Nationality of the owner"
            },
            "registration_info": {
                "registration_date": "Date when the registration was issued",
                "expiry_date": "Date when the registration expires",
                "renew_date": "Date when the registration needs to be renewed"
            },
            "insurance_info": {
                "insurance_company": "Name of the insurance company",
                "policy_number": "Insurance policy number",
                "expiry_date": "Date when the insurance expires"
            }
            """
        )
    
    @staticmethod
    def get_id_card_front_prompt() -> str:
        """
        Get the prompt template for ID card front side.
        
        Returns:
            Prompt template string
        """
        return (
            """
            Extract all information from this ID card or residency permit front side and return it as JSON.
            Output must be valid JSON only, no explanations.
            Output must be in the following format:
            {
              "document_info": {
                "document_type": "The type of document (e.g., 'Qatar Residency Permit', 'ID Card')",
                "issuing_authority": "Authority that issued the document (e.g., 'State of Qatar')"
              },
              "personal_info": {
                "id_number": "The ID or permit number (e.g., '28140001175')",
                "name": "Full name of the person as shown on the ID",
                "arabic_name": "Name in Arabic if present",
                "nationality": "Nationality of the ID holder (e.g., 'JORDAN', 'PALESTINE')",
                "date_of_birth": "Birth date in the format shown on the ID (e.g., '07/04/1981')",
                "expiry_date": "Expiration date of the ID (e.g., '07/07/2027')",
                "occupation": "Job title or occupation of the person"
              }
            }
            
            Make sure to preserve all original formatting of dates and numbers.
            Include all Arabic text exactly as shown, especially names.
            """
        )
    
    @staticmethod
    def get_id_card_back_prompt() -> str:
        """
        Get the prompt template for ID card back side.
        
        Returns:
            Prompt template string
        """
        return (
            """
            Extract all information from this ID card or residency permit back side and return it as JSON.
            Output must be valid JSON only, no explanations.
            Output must be in the following format:
            {
              "document_info": {
                "document_type": "The type of document (e.g., 'Qatar Residency Permit Back')",
                "issuing_authority": "Authority that issued the document if shown"
              },
              "additional_info": {
                "passport_number": "Passport number if shown",
                "passport_expiry": "Passport expiration date if shown",
                "residency_type": "Type of residency permit (e.g., 'Work', 'Family')",
                "employer": "Employer name if shown",
                "serial_number": "Serial number of the document if shown",
                "other_fields": {
                  "field_name": "field_value"
                }
              }
            }
            
            Make sure to preserve all original formatting of dates and numbers.
            Include all Arabic text exactly as shown.
            If any field is not visible or not present, omit it from the JSON.
            """
        )
    
    @staticmethod
    def get_prompt_for_document_type(document_type: str) -> str:
        """
        Get the appropriate prompt template based on document type.
        
        Args:
            document_type: Type of document ("vehicle_registration", "id_card_front", "id_card_back")
            
        Returns:
            Prompt template string
        """
        if document_type == "vehicle_registration":
            return PromptTemplates.get_vehicle_registration_prompt()
        elif document_type == "id_card_front":
            return PromptTemplates.get_id_card_front_prompt()
        elif document_type == "id_card_back":
            return PromptTemplates.get_id_card_back_prompt()
        else:
            # Default to vehicle registration if unknown
            logger.warning(f"Unknown document type: {document_type}, defaulting to vehicle registration prompt")
            return PromptTemplates.get_vehicle_registration_prompt() 