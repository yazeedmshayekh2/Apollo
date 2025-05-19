"""
Document Type Detector

This module provides functionality to detect document types from images
using the same Qwen VLM model used for document processing.
"""

import logging
import os
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import io
import torch

from utils.config import Config
from .registry import get_registry

logger = logging.getLogger(__name__)

class DocumentTypeDetector:
    """
    Detector for identifying document types (Vehicle Registration Card or ID Card).
    Uses the Qwen model to classify document types.
    """
    
    def __init__(self):
        """
        Initialize the document type detector.
        """
        # Define document types
        self.DOCUMENT_TYPES = {
            "VEHICLE_REGISTRATION": "vehicle_registration",
            "ID_CARD_FRONT": "id_card_front",
            "ID_CARD_BACK": "id_card_back",
            "UNKNOWN": "unknown"
        }
        
        # Define keyword sets for each document type (fallback method)
        self.keywords = {
            self.DOCUMENT_TYPES["VEHICLE_REGISTRATION"]: [
                "vehicle", "registration", "traffic", "ministry", "interior", 
                "vehicle no", "plate", "chassis", "engine", "car", "license",
                "make", "model", "year", "owner", "تسجيل", "مركبة", "المرور", "ترخيص"
            ],
            self.DOCUMENT_TYPES["ID_CARD_FRONT"]: [
                "id", "card", "identity", "identification", "national", 
                "residence", "residency", "permit", "qatar", "state of qatar",
                "name", "nationality", "هوية", "بطاقة", "إقامة", "قطر", "الاسم",
                "الجنسية", "تاريخ الميلاد", "الرقم الشخصي"
            ],
            self.DOCUMENT_TYPES["ID_CARD_BACK"]: [
                "passport", "number", "serial", "issue", "expiry",
                "employer", "signature", "جواز", "رقم", "سفر", "تاريخ",
                "صلاحية", "توقيع", "المستخدم"
            ]
        }
        
        # The detector will use the QwenService model - will be accessed via api.routes
        self.qwen_service = None
        
        logger.info("Document type detector initialized")
    
    def _create_detection_prompt(self) -> str:
        """Create the instruction prompt for document type detection."""
        return (
            """
            Analyze this document image carefully. Your task is to classify it as one of these types:
            1. "vehicle_registration" - If it's a vehicle registration card/document issued by a traffic department
            2. "id_card_front" - If it's the front side of an ID card or residency permit showing a person's photo
            3. "id_card_back" - If it's the back side of an ID card or residency permit
            
            Look for distinctive elements in the document:
            - Vehicle registration: Contains vehicle details, plate number, chassis number, owner information
            - ID card front: Has a person's photo, ID number, name, nationality, birth date
            - ID card back: Contains additional information like passport number, no photo visible
            
            Return ONLY the document type as one of: "vehicle_registration", "id_card_front", "id_card_back"
            """
        )
    
    def set_qwen_service(self, service):
        """Set the QwenService instance to use for detection."""
        self.qwen_service = service
    
    def detect_document_type(self, image) -> str:
        """
        Detect the type of document from the image.
        
        Args:
            image: Image input (file path, PIL Image, or bytes)
            
        Returns:
            Document type: "vehicle_registration", "id_card_front", "id_card_back", or "unknown"
        """
        try:
            # Make sure we have access to the QwenService
            if not self.qwen_service:
                # If no service is available, fallback to keyword-based detection
                logger.warning("QwenService not available, falling back to keywords")
                return self._detect_type_with_keywords(image)
            
            # Process the image to ensure it's in the right format
            processed_image = self.qwen_service._process_image(image)
            
            # Prepare the message with the detection prompt
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "text", "text": self._create_detection_prompt()},
                    {"type": "image", "image": processed_image}
                ]
            }]
            
            # Process with the existing QwenService
            # Generate a short response to determine document type
            max_tokens = 64
            try:
                from qwen_vl_utils import process_vision_info
            except ImportError:
                logger.error("Error importing process_vision_info")
                return self.DOCUMENT_TYPES["UNKNOWN"]
                
            # Prepare inputs for the model
            text = self.qwen_service.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.qwen_service.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to the same device as model
            device = self.qwen_service.device
            inputs = inputs.to(device)
            
            # Generate response - only need a short response for type detection
            generated_ids = self.qwen_service.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=False
            )
            
            # Trim input IDs from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.qwen_service.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0].strip().lower()
            
            logger.info(f"Document detection result: {output_text}")
            
            # Determine document type from output
            if "vehicle" in output_text or "registration" in output_text:
                return self.DOCUMENT_TYPES["VEHICLE_REGISTRATION"]
            elif ("id" in output_text or "card" in output_text or "residency" in output_text) and "front" in output_text:
                return self.DOCUMENT_TYPES["ID_CARD_FRONT"]
            elif ("id" in output_text or "card" in output_text or "residency" in output_text) and "back" in output_text:
                return self.DOCUMENT_TYPES["ID_CARD_BACK"]
            else:
                return self.DOCUMENT_TYPES["UNKNOWN"]
            
        except Exception as e:
            logger.error(f"Error in document type detection: {str(e)}")
            # Fallback to keyword-based detection if the model approach fails
            return self._detect_type_with_keywords(image)
    
    def _detect_type_with_keywords(self, image) -> str:
        """Fallback method using OCR and keywords to detect document type."""
        try:
            import pytesseract
            
            # Process the image
            if isinstance(image, str) and os.path.exists(image):
                # Load and preprocess the image
                img = Image.open(image).convert('L')
            elif isinstance(image, Image.Image):
                img = image.convert('L')
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image)).convert('L')
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return self.DOCUMENT_TYPES["UNKNOWN"]
            
            # Extract text using OCR
            text = pytesseract.image_to_string(img, lang='eng+ara').lower()
            
            # Count keyword matches for each document type
            scores = {
                self.DOCUMENT_TYPES["VEHICLE_REGISTRATION"]: 0,
                self.DOCUMENT_TYPES["ID_CARD_FRONT"]: 0,
                self.DOCUMENT_TYPES["ID_CARD_BACK"]: 0
            }
            
            for doc_type, keywords in self.keywords.items():
                for keyword in keywords:
                    if keyword.lower() in text:
                        scores[doc_type] += 1
            
            # Get the document type with the highest score
            best_match = max(scores, key=scores.get)
            
            # If score is too low, return unknown
            if scores[best_match] < 2:
                logger.warning("Inconclusive document type detection with keywords")
                return self.DOCUMENT_TYPES["UNKNOWN"]
                
            logger.info(f"Document type detected with keywords: {best_match}")
            return best_match
            
        except Exception as e:
            logger.error(f"Error in keyword-based detection: {str(e)}")
            return self.DOCUMENT_TYPES["UNKNOWN"] 