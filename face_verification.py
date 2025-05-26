import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pymongo import MongoClient
import cv2
import os
import warnings
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

class FaceVerification:
    def __init__(self):
        print("Initializing Face Verification System...")
        
        # Initialize YOLOv11n face detection model from Hugging Face
        try:
            print("Loading YOLOv11n face detection model from Hugging Face...")
            model_path = hf_hub_download(
                repo_id="AdamCodd/YOLOv11n-face-detection", 
                filename="model.pt"
            )
            self.face_detector = YOLO(model_path)
            print("YOLOv11n face detection model loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv11n model: {e}")
            print("Falling back to OpenCV face detection")
            self.face_detector = None
        
        # Initialize ResNet50 for feature extraction
        try:
            print("Loading ResNet-50 model...")
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.resnet.fc = nn.Linear(2048, 512)  # Modify last layer for feature extraction
            self.resnet.eval()
            print("ResNet-50 model loaded successfully")
            
            # Image transformation pipeline for ResNet
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        except Exception as e:
            print(f"Error loading ResNet-50 model: {e}")
            print("Falling back to HOG feature extraction")
            self.resnet = None
            self.transform = None
        
        # Using only cosine similarity for face comparison
        print("Using cosine similarity for face comparison (no Siamese network)")
        self.siamese = None
        
        # Initialize MongoDB connection
        try:
            self.client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
            self.client.server_info()  # Will raise an exception if cannot connect
            print("Connected to MongoDB successfully")
            self.db = self.client['face_verification']
            self.collection = self.db['face_features']
        except Exception as e:
            print(f"Warning: Could not connect to MongoDB: {e}")
            print("Will continue without database functionality")
            self.client = None
            self.db = None
            self.collection = None
            
    # Siamese network implementation removed - using only cosine similarity

    def detect_face(self, image_path):
        """Detect face in the image using YOLOv11n or fall back to OpenCV"""
        print(f"Detecting face in: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image from {image_path}")
            return None
            
        # Try YOLOv11n face detection first
        if self.face_detector is not None:
            try:
                # Run inference with YOLOv11n model
                results = self.face_detector(img, verbose=False)
                
                # Process YOLOv11n results
                if len(results) > 0 and len(results[0].boxes) > 0:
                    # Get the detection with highest confidence
                    boxes = results[0].boxes
                    confidences = boxes.conf.cpu().numpy()
                    best_idx = np.argmax(confidences)
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().astype(int)
                    conf = confidences[best_idx]
                    
                    print(f"Face detected with YOLOv11n: conf={conf:.3f}, box=[{x1},{y1},{x2},{y2}]")
                    
                    # Extract face region with some padding
                    h, w = img.shape[:2]
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    face = img[y1:y2, x1:x2]
                    
                    # Validate face region
                    if face.shape[0] > 20 and face.shape[1] > 20:
                        return face
                    else:
                        print("Detected face region too small, falling back to OpenCV")
                else:
                    print("No face detected with YOLOv11n")
            except Exception as e:
                print(f"Error during YOLOv11n face detection: {e}")
        
        # Fall back to OpenCV Haar Cascade if YOLOv11n fails or is not available
        print("Falling back to OpenCV Haar Cascade for face detection")
        try:
            # Convert to grayscale for OpenCV face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Load OpenCV's pre-trained face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # If faces detected, return the first one
            if len(faces) > 0:
                print(f"Face detected with OpenCV: {faces[0]}")
                x, y, w, h = faces[0]
                face = img[y:y+h, x:x+w]
                return face
            else:
                print("No face detected with OpenCV Haar Cascade")
                # As a fallback, return the center portion of the image
                h, w = img.shape[:2]
                center_x, center_y = w // 2, h // 2
                size = min(w, h) // 2
                face = img[center_y-size:center_y+size, center_x-size:center_x+size]
                print(f"Using center portion as fallback: {center_x-size}, {center_y-size}, {center_x+size}, {center_y+size}")
                return face
        except Exception as e:
            print(f"Error during OpenCV face detection: {e}")
            return None

    def extract_features(self, face_image):
        """Extract features using ResNet-50 or fall back to OpenCV's HOG descriptor"""
        if face_image is None:
            print("Cannot extract features from None image")
            return None
        
        # Try ResNet-50 feature extraction first
        if self.resnet is not None and self.transform is not None:
            try:
                # Convert BGR to RGB (OpenCV uses BGR, PyTorch expects RGB)
                face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                
                # Apply transformations and add batch dimension
                face_tensor = self.transform(face_pil).unsqueeze(0)
                
                # Extract features using ResNet-50
                with torch.no_grad():
                    features = self.resnet(face_tensor)
                    
                print(f"Extracted ResNet-50 feature vector with shape {features.shape}")
                return features.squeeze().cpu().numpy()
            except Exception as e:
                print(f"Error extracting features with ResNet-50: {e}")
                print("Falling back to HOG feature extraction")
        
        # Fall back to HOG if ResNet-50 fails or is not available
        try:
            print("Using HOG for feature extraction")
            # Resize for consistency
            face_resized = cv2.resize(face_image, (64, 128))
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # Initialize HOG descriptor
            hog = cv2.HOGDescriptor()
            features = hog.compute(gray)
            
            # Normalize features
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)
                
            print(f"Extracted HOG feature vector with shape {features.shape}")
            return features.flatten()
        except Exception as e:
            print(f"Error extracting features with HOG: {e}")
            return None

    def save_to_database(self, user_id, features):
        """Save face features to MongoDB"""
        if self.collection is None:
            print("Database not available, cannot save features")
            # Save to local file as fallback
            np.save(f"{user_id}_features.npy", features)
            print(f"Saved features to local file: {user_id}_features.npy")
            return
            
        try:
            self.collection.update_one(
                {'user_id': user_id},
                {'$set': {'features': features.tolist()}},
                upsert=True
            )
            print(f"Features saved to database for user: {user_id}")
        except Exception as e:
            print(f"Error saving to database: {e}")
            # Save to local file as fallback
            np.save(f"{user_id}_features.npy", features)
            print(f"Saved features to local file: {user_id}_features.npy")

    def verify_face(self, image_path, user_id):
        """Verify if the face in the image matches the stored face using Siamese network or fallback methods"""
        # Detect face in the new image
        face = self.detect_face(image_path)
        if face is None:
            return False, "No face detected"

        # Extract features
        features = self.extract_features(face)
        if features is None:
            return False, "Could not extract features"

        # Get stored features
        stored_features = None
        if self.collection is not None:
            try:
                stored_data = self.collection.find_one({'user_id': user_id})
                if stored_data is not None:
                    stored_features = np.array(stored_data['features'])
                    print(f"Retrieved features from database for user: {user_id}")
            except Exception as e:
                print(f"Error retrieving from database: {e}")
        
        # Try loading from file if not found in database
        if stored_features is None:
            try:
                feature_file = f"{user_id}_features.npy"
                if os.path.exists(feature_file):
                    stored_features = np.load(feature_file)
                    print(f"Loaded features from file: {feature_file}")
                else:
                    return False, "User not found in database or local storage"
            except Exception as e:
                return False, f"Error loading features: {e}"
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0  # Handle zero vectors
            return dot_product / (norm_a * norm_b)
            
        cosine_sim = cosine_similarity(features, stored_features)
        print(f"Cosine similarity: {cosine_sim:.4f}")
        
        # Determine match based on cosine similarity
        threshold = 0.7  # Threshold for determining a match
        is_match = cosine_sim > threshold
        print(f"Match determination: {cosine_sim:.4f} > {threshold} = {is_match}")
        
        # Return result
        return is_match, {
            'similarity_score': cosine_sim,
            'cosine_similarity': cosine_sim,
            'threshold': threshold
        } 