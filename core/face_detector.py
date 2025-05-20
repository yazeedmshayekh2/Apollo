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
import tensorflow as tf
from tensorflow import keras
import sys

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
        self.sia_model = None
        self.use_keras = False
        
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
        """Load YOLO model for face detection."""
        if self.detection_model is not None:
            return
        
        try:
            model_name = Config.get("YOLO_MODEL", "yolov9-face-detection.pt")
            logger.info(f"Loading YOLO model for face detection: {model_name}")
            
            # Check if it's a YOLOv9 model (based on filename)
            if "yolov9" in model_name.lower():
                # Simple approach just like in test.py
                self.detection_model = torch.hub.load('./yolov9', 'custom', 
                                                     path=os.path.abspath(model_name), 
                                                     force_reload=True, 
                                                     source='local')
                
                self.detection_model.to(self.device)
                logger.info(f"YOLOv9 model loaded successfully: {model_name}")
            else:
                # Use ultralytics YOLO for other models (v5-v8)
                cache_dir = Config.get("MODEL_CACHE_DIR", "model_cache")
                self.detection_model = YOLO(model_name)
                logger.info(f"YOLO model loaded successfully: {model_name}")
                
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise
    
    def _load_feature_model(self):
        """Load ResNet-50 model with custom weights for feature extraction."""
        if self.feature_model is not None:
            return
        
        try:
            logger.info("Loading ResNet-50 model for feature extraction")
            
            # First try to load the Keras model with custom weights
            try:
                # Load pre-trained model with custom weights
                weights_path = './resnet-50.h5'
                self.feature_model = keras.models.load_model(weights_path)
                # Remove the classification layer to get features
                self.feature_model = tf.keras.Model(inputs=self.feature_model.input, 
                                                outputs=self.feature_model.layers[-2].output)
                logger.info("Keras ResNet-50 model with custom weights loaded successfully")
                self.use_keras = True
            except Exception as e:
                logger.warning(f"Could not load Keras model: {e}. Falling back to PyTorch ResNet-50.")
                # Fallback to regular PyTorch ResNet-50
                self.feature_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                
                # Remove the last fully connected layer to get features
                self.feature_model = torch.nn.Sequential(*(list(self.feature_model.children())[:-1]))
                
                # Set to evaluation mode
                self.feature_model.eval()
                
                # Move to appropriate device
                self.feature_model = self.feature_model.to(self.device)
                
                self.use_keras = False
                logger.info("PyTorch ResNet-50 model loaded successfully")
                
        except Exception as e:
            logger.error(f"Error loading ResNet-50 model: {str(e)}")
            raise
    
    def _load_siamese_model(self):
        """Load Siamese network for similarity comparison."""
        if self.sia_model is not None:
            return
            
        try:
            logger.info("Loading Siamese network model")
            self.sia_model = keras.models.load_model('siamese.h5')
            logger.info("Siamese network model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Siamese network model: {str(e)}")
            return False
    
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
            
            # Get image dimensions
            height, width = img_np.shape[:2]
            
            # Run detection based on model type
            start_time = time.time()
            model_name = Config.get("YOLO_MODEL", "yolov9-face-detection.pt")
            
            detections = []
            best_detection = None
            max_area = 0
            
            # Handle detection differently based on model type
            if "yolov9" in model_name.lower() and isinstance(self.detection_model, torch.nn.Module):
                # Process for YOLOv9
                # Convert BGR to RGB for YOLOv9
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                
                # Run YOLOv9 detection
                results = self.detection_model(img_rgb)
                detection_data = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
                
                # Process detection results
                for i, detection in enumerate(detection_data):
                    x1, y1, x2, y2, conf, cls = detection
                    
                    # Convert to integers
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    
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
            else:
                # Process for YOLO (using ultralytics)
                results = self.detection_model(img_np, classes=[0], conf=0.25)  # class 0 is person in COCO
                
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
                        
            detection_time = time.time() - start_time
            
            # Extract features if a person/face was detected
            embeddings = None
            embeddings_dict = {}
            feature_extraction_time = 0
            
            if best_detection is not None and len(detections) > 0:
                # Get the best detection
                best_box = detections[best_detection]["box"]
                x1, y1, x2, y2 = [int(c) for c in best_box]
                
                # Crop the person/face from the image
                cropped_img = img_np[y1:y2, x1:x2]
                
                # Extract features based on the model type
                start_time = time.time()
                
                if self.use_keras:
                    # Process for Keras model
                    # Convert from BGR to RGB
                    face_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to the input shape the model expects
                    resized = cv2.resize(face_img_rgb, (224, 224))
                    
                    # Preprocess for Keras
                    x = np.expand_dims(resized, axis=0)
                    x = x / 255.0  # Normalize to [0,1]
                    
                    # Extract features
                    embeddings = self.feature_model.predict(x)
                    embeddings = embeddings[0]  # Get as 1D array
                else:
                    # Process for PyTorch model
                    # Convert to PIL for transformation
                    person_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                    
                    # Apply transformation for ResNet
                    person_tensor = self.transform(person_pil).unsqueeze(0).to(self.device)
                    
                    # Extract features
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
                        "version": "resnet50_custom" if self.use_keras else "resnet50_v1",
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
    
    def compare_embeddings(self, embedding1, embedding2) -> Dict[str, Any]:
        """
        Compare two face embeddings using Siamese network.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Dict with similarity score and method used
        """
        try:
            # Load Siamese model if not already loaded
            self._load_siamese_model()
            
            # Ensure embeddings are the right shape
            if isinstance(embedding1, list):
                embedding1 = np.array(embedding1)
            if isinstance(embedding2, list):
                embedding2 = np.array(embedding2)
                
            if embedding1.ndim > 1:
                embedding1 = embedding1.flatten()
            if embedding2.ndim > 1:
                embedding2 = embedding2.flatten()
                
            # Check if shapes match
            if embedding1.shape != embedding2.shape:
                logger.warning(f"Embedding shapes don't match: {embedding1.shape} vs {embedding2.shape}")
                # Try to resize if possible
                min_size = min(embedding1.size, embedding2.size)
                embedding1 = embedding1[:min_size]
                embedding2 = embedding2[:min_size]
            
            # Compute similarity
            method = "siamese_network"
            start_time = time.time()
            
            if self.sia_model is not None:
                # Use Siamese network
                # First compute cosine similarity
                from sklearn.metrics.pairwise import cosine_similarity
                similarity_matrix = cosine_similarity(
                    embedding1.reshape(1, -1), 
                    embedding2.reshape(1, -1)
                )
                
                # Pass to Siamese network for final similarity
                similarity = self.sia_model.predict(
                    np.array([similarity_matrix[0][0]]).reshape(1, -1)
                )
                similarity = float(similarity[0][0])  # Extract scalar value
            else:
                # Fallback to cosine similarity
                method = "cosine_similarity"
                from scipy.spatial.distance import cosine
                similarity = 1.0 - cosine(embedding1, embedding2)
            
            computation_time = time.time() - start_time
            
            return {
                "similarity": similarity,
                "method": method,
                "computation_time": computation_time
            }
            
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            return {
                "similarity": 0.0,
                "method": "error",
                "error": str(e)
            } 