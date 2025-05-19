"""
Face and Person Detection with Feature Extraction

This module provides functionality to:
1. Detect persons/faces in ID card images using YOLO11
2. Extract embeddings using ResNet-50 for the detected faces
"""

import os
import logging
import torch
import numpy as np
import cv2
from typing import Dict, Any, List, Tuple, Optional, Union
from PIL import Image
import io
import base64
import tempfile
from pathlib import Path
import time

from torchvision import models, transforms
from ultralytics import YOLO

from utils.config import Config

logger = logging.getLogger(__name__)

class FaceDetector:
    """
    Detector for faces and persons in ID card images.
    Uses YOLO11 for detection and ResNet-50 for feature extraction.
    """
    
    def __init__(self):
        """Initialize the face detector and feature extractor."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Face detector using device: {self.device}")
        
        # Initialize models
        self.detection_model = None
        self.feature_model = None
        
        # Define transformation for ResNet-50
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Download directory for models
        os.makedirs(Config.get("MODEL_CACHE_DIR", "model_cache"), exist_ok=True)
        
        logger.info("Face detector initialized")
    
    def _load_detection_model(self):
        """Load YOLO11 model for person detection."""
        if self.detection_model is not None:
            return
        
        try:
            logger.info("Loading YOLO11 model for person detection")
            model_name = Config.get("YOLO_MODEL", "yolo11l")
            cache_dir = Config.get("MODEL_CACHE_DIR", "model_cache")
            
            # Load YOLO11 model
            self.detection_model = YOLO(model_name)
            
            logger.info(f"YOLO11 model loaded successfully: {model_name}")
        except Exception as e:
            logger.error(f"Error loading YOLO11 model: {str(e)}")
            raise
    
    def _load_feature_model(self):
        """Load ResNet-50 model for feature extraction."""
        if self.feature_model is not None:
            return
        
        try:
            logger.info("Loading ResNet-50 model for feature extraction")
            
            # Load pre-trained ResNet-50
            self.feature_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            
            # Remove the last fully connected layer to get features
            self.feature_model = torch.nn.Sequential(*(list(self.feature_model.children())[:-1]))
            
            # Set to evaluation mode
            self.feature_model.eval()
            
            # Move to appropriate device
            self.feature_model = self.feature_model.to(self.device)
            
            logger.info("ResNet-50 model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading ResNet-50 model: {str(e)}")
            raise
    
    def _process_image(self, image) -> np.ndarray:
        """
        Process the image to ensure it's in a format suitable for detection.
        
        Args:
            image: Image input (file path, PIL Image, or bytes)
            
        Returns:
            numpy array image
        """
        # If image is a string (file path)
        if isinstance(image, str) and os.path.exists(image):
            return cv2.imread(image)
        
        # If image is bytes
        elif isinstance(image, bytes):
            nparr = np.frombuffer(image, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # If image is a PIL Image
        elif isinstance(image, Image.Image):
            # Convert PIL Image to numpy array
            return np.array(image.convert('RGB'))[:, :, ::-1].copy()  # Convert RGB to BGR for OpenCV
        
        # If it's a base64 string
        elif isinstance(image, str) and image.startswith('data:image'):
            # Extract the base64 encoded image data
            base64_data = image.split(',')[1]
            image_data = base64.b64decode(base64_data)
            nparr = np.frombuffer(image_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        else:
            logger.error(f"Unsupported image type: {type(image)}")
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def detect_face(self, image) -> Dict[str, Any]:
        """
        Detect faces/persons in an ID card image and extract features.
        
        Args:
            image: Image input (file path, PIL Image, or bytes)
            
        Returns:
            Dictionary with detection results and embeddings
        """
        try:
            # Load models if not already loaded
            self._load_detection_model()
            self._load_feature_model()
            
            # Process image to numpy array
            img_np = self._process_image(image)
            
            # Run YOLO11 detection
            start_time = time.time()
            results = self.detection_model(img_np, classes=[0], conf=0.25)  # class 0 is person in COCO
            detection_time = time.time() - start_time
            
            # Get detection results
            detections = []
            best_detection = None
            max_area = 0
            
            height, width = img_np.shape[:2]
            
            # Process detection results
            for i, detection in enumerate(results[0].boxes.data.cpu().numpy()):
                x1, y1, x2, y2, conf, cls = detection
                
                # Calculate area
                area = (x2 - x1) * (y2 - y1)
                
                # Add detection to list
                detections.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class": int(cls),
                    "area": float(area),
                    "area_percent": float(area) / (height * width) * 100
                })
                
                # Update best detection (largest area)
                if area > max_area:
                    max_area = area
                    best_detection = i
            
            # Extract features if a person was detected
            embeddings = None
            embeddings_dict = {}
            feature_extraction_time = 0
            
            if best_detection is not None and len(detections) > 0:
                # Get the best detection
                best_box = detections[best_detection]["box"]
                x1, y1, x2, y2 = [int(c) for c in best_box]
                
                # Crop the person from the image
                person_img = img_np[y1:y2, x1:x2]
                
                # Convert to PIL for transformation
                person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                
                # Apply transformation for ResNet
                person_tensor = self.transform(person_pil).unsqueeze(0).to(self.device)
                
                # Extract features
                start_time = time.time()
                with torch.no_grad():
                    features = self.feature_model(person_tensor)
                    # Flatten and convert to numpy
                    embeddings = features.squeeze().cpu().numpy()
                feature_extraction_time = time.time() - start_time
                
                # Convert embeddings to list for JSON serialization
                if embeddings is not None:
                    embeddings_list = embeddings.tolist()
                    
                    # Create a more compact representation for storage
                    embeddings_dict = {
                        "version": "resnet50_v1",
                        "shape": list(embeddings.shape),
                        "data": embeddings_list,
                        "extraction_time": feature_extraction_time
                    }
            
            # Return detection results and embeddings
            return {
                "detection_status": len(detections) > 0,
                "num_detections": len(detections),
                "detections": detections,
                "best_detection_index": best_detection,
                "detection_time": detection_time,
                "feature_extraction_time": feature_extraction_time,
                "embeddings": embeddings_dict,
                "image_shape": [height, width, 3] if img_np is not None else None
            }
            
        except Exception as e:
            logger.error(f"Error in face detection: {str(e)}")
            return {
                "detection_status": False,
                "error": str(e)
            }
    
    def extract_embeddings_from_cropped(self, image) -> Dict[str, Any]:
        """
        Extract embeddings from an already cropped face/person image.
        
        Args:
            image: Cropped image input (file path, PIL Image, or bytes)
            
        Returns:
            Dictionary with embeddings
        """
        try:
            # Load feature model if not already loaded
            self._load_feature_model()
            
            # Process image
            img_np = self._process_image(image)
            
            # Convert to PIL for transformation
            img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            
            # Apply transformation for ResNet
            img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            # Extract features
            start_time = time.time()
            with torch.no_grad():
                features = self.feature_model(img_tensor)
                # Flatten and convert to numpy
                embeddings = features.squeeze().cpu().numpy()
            extraction_time = time.time() - start_time
            
            # Convert embeddings to list for JSON serialization
            embeddings_dict = {
                "version": "resnet50_v1",
                "shape": list(embeddings.shape),
                "data": embeddings.tolist(),
                "extraction_time": extraction_time
            }
            
            return {
                "status": True,
                "embeddings": embeddings_dict
            }
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {str(e)}")
            return {
                "status": False,
                "error": str(e)
            } 