"""
Qwen Vision Language Model Service for Vehicle Registration Card OCR

This module provides a service to extract information from vehicle registration cards
using the Qwen 2.5 Vision Language Model. It takes front and back images of a card
and returns structured JSON data with all extracted information.
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
    vehicle registration cards.
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
    
    def _create_prompt(self) -> str:
        """Create the instruction prompt for the model."""
        # Prompt with explicit instruction to maintain original script (Arabic, etc.)
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
    
    def _prepare_message(self, front_image, back_image) -> List[Dict[str, Any]]:
        """
        Prepare the message for the model with front and back images.
        
        Args:
            front_image: Front image of the registration card (path, URL, or bytes)
            back_image: Back image of the registration card (path, URL, or bytes)
            
        Returns:
            Message in the format expected by the model
        """
        content = [
            {"type": "text", "text": self._create_prompt()},
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
    
    def extract_info(
        self, 
        front_image, 
        back_image=None, 
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """
        Extract information from front and back images of a vehicle registration card.
        
        Args:
            front_image: Front image of the card (path, URL, or bytes)
            back_image: Back image of the card (path, URL, or bytes)
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
            
            # Prepare the message for the model
            messages = self._prepare_message(front_processed, back_processed)
            
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
            front_data = self.extractor.extract_json(output_text)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Add metadata to the result
            result = front_data.copy()
            if "metadata" not in result:
                result["metadata"] = {}
                
            result["metadata"].update({
                "process_id": process_id,
                "processing_time": processing_time,
                "model_id": self.model_id,
                "model_name": self.model_name,
            })
            
            # Try to log performance (catch errors to avoid breaking processing)
            try:
                self.registry.log_model_performance(
                    self.model_id,
                    {
                        "processing_time": processing_time,
                        "succeeded": self.extractor.is_valid_json(front_data),
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
            
    def is_available(self) -> bool:
        """Check if the service is available."""
        return hasattr(self, "model") and hasattr(self, "processor")
