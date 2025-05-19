"""
Qwen Vision Language Model Service for Document OCR

This module provides a service to extract information from various document types
(vehicle registration cards and ID cards) using the Qwen 2.5 Vision Language Model.
It detects the document type and uses appropriate prompts for information extraction.
"""

import os
import logging
import base64
import json
import gc
from typing import Dict, Any, List, Tuple, Optional, Union
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import io

from utils.config import Config
from utils.helpers import generate_unique_id, calculate_hash
from .registry import ModelRegistry
from .extractor import VehicleDataExtractor
from .document_detector import DocumentTypeDetector
from .prompt_templates import PromptTemplates
from .face_detector import FaceDetector

# Import process_vision_info from qwen_vl_utils
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError(
        "qwen_vl_utils module not found. Please make sure you have installed "
        "the correct version of the Qwen2.5-VL package."
    )

logger = logging.getLogger(__name__)

class QwenService:
    """
    Qwen Vision Language Model service for extracting information from 
    various document types including vehicle registration cards and ID cards.
    """
    
    def __init__(self, model_id: Optional[str] = None):
        """
        Initialize the Qwen Vision Language Model service.
        
        Args:
            model_id: Optional model ID from the registry to use
        """
        self.registry = ModelRegistry()
        self.model_id = model_id or self.registry.get_default_model_id()
        self.model_config = self.registry.get_model_by_id(self.model_id)
        
        if not self.model_config:
            logger.warning(f"Model ID {self.model_id} not found. Using default configuration.")
            self.model_config = {
                "model_name": Config.get("MODEL_NAME"),
                "configuration": Config.model_config()
            }
            
        self.model_name = self.model_config.get("model_name", "unsloth/Qwen2.5-VL-7B-Instruct")
        self.config = self.model_config.get("configuration", {})
        
        # Initialize the model and processor
        self._initialize_model()
        
        # Initialize the extractor
        self.extractor = VehicleDataExtractor()
        
        # Initialize the document detector
        self.document_detector = DocumentTypeDetector()
        
        # Initialize the face detector (lazy loaded when needed)
        self.face_detector = None
        
        # Device property
        self.device = next(self.model.parameters()).device if hasattr(self, "model") else "cpu"
        
        logger.info(f"QwenService initialized with model: {self.model_name}")
    
    def _initialize_model(self):
        """Initialize the Qwen model and processor."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Get configuration parameters
        use_flash_attention = self.config.get("use_flash_attention", False)
        device_map = self.config.get("device_map", "auto")
        
        # Set environment variables for better memory management
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Load the model
        attn_implementation = "flash_attention_2" if use_flash_attention else None
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        logger.info("Model and processor loaded successfully")
    
    def _get_face_detector(self):
        """Get or initialize face detector."""
        if self.face_detector is None:
            logger.info("Initializing face detector")
            self.face_detector = FaceDetector()
        return self.face_detector
    
    def _create_prompt(self, document_type: str) -> str:
        """
        Create the instruction prompt for the model based on document type.
        
        Args:
            document_type: The type of document to process
            
        Returns:
            Appropriate prompt template for the document type
        """
        return PromptTemplates.get_prompt_for_document_type(document_type)
    
    def _prepare_message(self, front_image, back_image, document_type: str) -> List[Dict[str, Any]]:
        """
        Prepare the message for the model with front and back images.
        
        Args:
            front_image: Front image of the document (path, URL, or bytes)
            back_image: Back image of the document (path, URL, or bytes)
            document_type: Type of document to process
            
        Returns:
            Message in the format expected by the model
        """
        content = [
            {"type": "text", "text": self._create_prompt(document_type)},
            {"type": "image", "image": front_image}
        ]
        
        if back_image:
            content.append({"type": "image", "image": back_image})
            
        return [{"role": "user", "content": content}]
    
    def _process_image(self, image) -> Union[str, Image.Image]:
        """
        Process the image input to ensure it's in a format the model can use.
        Also resize large images to reduce memory usage.
        
        Args:
            image: Image input (file path, URL, PIL Image, or bytes)
            
        Returns:
            Processed image in a format accepted by the model
        """
        # Maximum dimensions to prevent memory issues
        MAX_WIDTH = 3000
        MAX_HEIGHT = 3000
        
        img = None
        
        # If image is a string, it could be a file path or URL
        if isinstance(image, str):
            # Check if it's a local file path
            if os.path.exists(image):
                img = Image.open(image)
            else:
                # Otherwise assume it's a URL
                return image
        # If image is bytes, convert to PIL Image        
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        # If image is a PIL Image, use it directly    
        elif isinstance(image, Image.Image):
            img = image
        # If it's a base64 string    
        elif isinstance(image, str) and image.startswith('data:image'):
            # Extract the base64 encoded image data
            base64_data = image.split(',')[1]
            image_data = base64.b64decode(base64_data)
            img = Image.open(io.BytesIO(image_data))
        # Unsupported type    
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Resize large images to reduce memory usage
        if img and (img.width > MAX_WIDTH or img.height > MAX_HEIGHT):
            # Calculate new dimensions maintaining aspect ratio
            ratio = min(MAX_WIDTH / img.width, MAX_HEIGHT / img.height)
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Resized image to {new_width}x{new_height} to reduce memory usage")
            
        return img
    
    def _process_id_card(self, image, document_type: str) -> Dict[str, Any]:
        """
        Process an ID card image to detect faces and extract embeddings.
        
        Args:
            image: Image of the ID card
            document_type: Type of document (id_card_front, id_card_back)
            
        Returns:
            Dictionary with face detection results and embeddings
        """
        # Only perform face detection on ID card front
        if document_type != "id_card_front":
            return {}
        
        try:
            logger.info("Processing ID card for face detection")
            
            # Get face detector
            face_detector = self._get_face_detector()
            
            # Process the image
            processed_image = self._process_image(image)
            
            # Detect faces and extract features
            results = face_detector.detect_face(processed_image)
            
            # Add timestamp
            results["timestamp"] = generate_unique_id("time")
            
            logger.info(f"Face detection results: {len(results.get('detections', []))} detections")
            
            return results
        except Exception as e:
            logger.error(f"Error processing ID card for face detection: {str(e)}")
            return {
                "detection_status": False,
                "error": str(e)
            }
    
    def extract_info(
        self, 
        front_image, 
        back_image=None, 
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Extract information from document images after detecting document type.
        
        Args:
            front_image: Front image of the document (path, URL, or bytes)
            back_image: Back image of the document (path, URL, or bytes)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Extracted information as a dictionary
        """
        process_id = generate_unique_id("ocr")
        logger.info(f"Processing request {process_id}")
        
        try:
            # Clear CUDA cache before processing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Process images to ensure they're in the right format
            front_processed = self._process_image(front_image)
            back_processed = self._process_image(back_image) if back_image else None
            
            # Detect document type from front image
            document_type = self.document_detector.detect_document_type(front_processed)
            logger.info(f"Detected document type: {document_type}")
            
            # Variables to store face detection results
            face_detection_results = {}
            
            # Handle ID card specific processing
            if document_type.startswith("id_card"):
                # Process ID card for face detection
                face_detection_results = self._process_id_card(front_processed, document_type)
            
            # Handle back image if present
            if back_processed and document_type == "id_card_front":
                # If the front is an ID card front, check if the back is an ID card back
                back_document_type = self.document_detector.detect_document_type(back_processed)
                logger.info(f"Detected back document type: {back_document_type}")
                
                # Update the back document type if it's an unknown
                if back_document_type == "unknown":
                    back_document_type = "id_card_back"
                
                # Process front and back separately for ID cards
                if back_document_type == "id_card_back":
                    # Process front image
                    front_result = self._process_single_image(front_processed, document_type, max_tokens)
                    
                    # Process back image
                    back_result = self._process_single_image(back_processed, back_document_type, max_tokens)
                    
                    # Merge the results
                    result = self._merge_id_card_results(front_result, back_result)
                    
                    # Add document type info
                    result["document_type"] = "id_card"
                    result["metadata"] = result.get("metadata", {})
                    result["metadata"]["front_processed"] = True
                    result["metadata"]["back_processed"] = True
                    
                    # Add face detection results if available
                    if face_detection_results and face_detection_results.get("detection_status"):
                        result["face_detection"] = face_detection_results
                    
                    return result
            
            # If it's a vehicle registration or only one side of ID card
            # Prepare the message for the model
            messages = self._prepare_message(front_processed, back_processed, document_type)
            
            # Start timing
            import time
            start_time = time.time()
            
            # Prepare inputs for the model
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process vision information
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            
            # Move inputs to the same device as model
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Generate response with shorter max_tokens to save memory
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=False  # Deterministic generation
            )
            
            # Trim input IDs from output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Ensure Unicode is preserved
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
            
            # Clear memory after generation
            del inputs, generated_ids, generated_ids_trimmed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Log the raw output for debugging unicode issues
            logger.debug(f"Raw model output: {output_text[:200]}...")
            
            # Extract JSON from the output text
            result_data = self.extractor.extract_json(output_text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add metadata to the result
            result = result_data.copy()
            if "metadata" not in result:
                result["metadata"] = {}
                
            result["metadata"].update({
                "process_id": process_id,
                "processing_time": processing_time,
                "model_id": self.model_id,
                "model_name": self.model_name,
                "document_type": document_type
            })
            
            # Add document type field at the top level too
            result["document_type"] = document_type
            
            # Add face detection results if available (for ID cards)
            if face_detection_results and face_detection_results.get("detection_status"):
                result["face_detection"] = face_detection_results
            
            # Try to log performance (catch errors to avoid breaking processing)
            try:
                self.registry.log_model_performance(
                    self.model_id,
                    {
                        "processing_time": processing_time,
                        "succeeded": self.extractor.is_valid_json(result_data),
                    },
                    sample_id=process_id
                )
            except Exception as e:
                logger.warning(f"Failed to log performance: {e}")
            
            logger.info(f"Request {process_id} processed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing request {process_id}: {str(e)}")
            return {
                "error": str(e),
                "metadata": {
                    "process_id": process_id,
                    "model_id": self.model_id,
                    "model_name": self.model_name,
                    "success": False
                }
            }
    
    def _process_single_image(self, image, document_type: str, max_tokens: int) -> Dict[str, Any]:
        """
        Process a single document image using the appropriate prompt.
        
        Args:
            image: Image to process (PIL.Image)
            document_type: Type of document
            max_tokens: Maximum tokens to generate
            
        Returns:
            Extracted information as a dictionary
        """
        # Prepare message with single image
        messages = self._prepare_message(image, None, document_type)
        
        # Start timing
        import time
        start_time = time.time()
        
        # Prepare inputs for the model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the same device as model
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        
        # Generate response
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=False
        )
        
        # Trim input IDs from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # Clear memory
        del inputs, generated_ids, generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Extract JSON from the output text
        result = self.extractor.extract_json(output_text)
        
        # Add metadata
        processing_time = time.time() - start_time
        if "metadata" not in result:
            result["metadata"] = {}
            
        result["metadata"].update({
            "processing_time": processing_time,
            "document_type": document_type
        })
        
        # If this is an ID card front, add face detection results
        if document_type == "id_card_front":
            face_detection_results = self._process_id_card(image, document_type)
            if face_detection_results and face_detection_results.get("detection_status"):
                result["face_detection"] = face_detection_results
        
        return result
    
    def _merge_id_card_results(self, front_result: Dict[str, Any], back_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge front and back ID card results into a single cohesive result.
        
        Args:
            front_result: Results from front of ID card
            back_result: Results from back of ID card
            
        Returns:
            Merged results
        """
        # Create a copy of front data as the base
        merged = front_result.copy()
        
        # If back result has document_info, merge it with front
        if "document_info" in merged and "document_info" in back_result:
            for key, value in back_result["document_info"].items():
                if key not in merged["document_info"]:
                    merged["document_info"][key] = value
        
        # Add the additional_info from back side if present
        if "additional_info" in back_result:
            merged["additional_info"] = back_result["additional_info"]
        
        # Merge metadata
        if "metadata" in back_result:
            # Calculate total processing time
            total_time = (
                front_result.get("metadata", {}).get("processing_time", 0) +
                back_result.get("metadata", {}).get("processing_time", 0)
            )
            merged["metadata"]["total_processing_time"] = total_time
            merged["metadata"]["back_processed"] = True
        
        return merged
            
    def is_available(self) -> bool:
        """Check if the service is available."""
        return hasattr(self, "model") and hasattr(self, "processor")
