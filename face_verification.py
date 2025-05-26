import torch
# import torch.nn as nn # No longer needed for custom ResNet fc layer
# import torchvision.models as models # No longer needed for ResNet
# import torchvision.transforms as transforms # No longer needed for ResNet
from PIL import Image # Still potentially useful if DeepFace needs PIL, though it often handles cv2 images
import numpy as np
from pymongo import MongoClient
import cv2
import os
import warnings
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from deepface import DeepFace # Added DeepFace

class FaceVerification:
    def __init__(self):
        print("Initializing Face Verification System with Facenet512...")
        
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
            print("Falling back to OpenCV face detection for YOLO part") # Clarified this fallback
            self.face_detector = None # This remains the trigger for OpenCV fallback in detect_face
        
        # Initialize Facenet512 model using DeepFace
        try:
            print("Loading Facenet512 model via DeepFace...")
            DeepFace.build_model("Facenet512") # Pre-load the model
            print("Facenet512 model loaded successfully.")
            self.feature_extractor_model_name = "Facenet512"
        except Exception as e:
            print(f"Error loading Facenet512 model via DeepFace: {e}")
            print("Face feature extraction will not be available.")
            self.feature_extractor_model_name = None
        
        # Using only cosine similarity for face comparison
        print("Using cosine similarity for face comparison.")
        # self.siamese = None # This line can be removed if siamese was fully removed previously
        
        # Initialize MongoDB connection
        try:
            self.client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
            self.client.server_info()  # Will raise an exception if cannot connect
            print("Connected to MongoDB successfully")
            self.db = self.client['face_verification']
            # Updated collection name to reflect Facenet512 features
            self.collection = self.db['face_features_facenet512'] 
        except Exception as e:
            print(f"Warning: Could not connect to MongoDB: {e}")
            print("Will continue without database functionality (using local files).")
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
                        print("Detected face region too small with YOLOv11n, falling back to OpenCV if configured.")
                else:
                    print("No face detected with YOLOv11n.")
            except Exception as e:
                print(f"Error during YOLOv11n face detection: {e}")
        
        # Fall back to OpenCV Haar Cascade if YOLOv11n fails or is not available
        print("Falling back to OpenCV Haar Cascade for face detection.")
        try:
            # Convert to grayscale for OpenCV face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Load OpenCV's pre-trained face detector
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                print(f"Error: Haar Cascade file not found at {cascade_path}")
                return None # Cannot proceed with OpenCV detection
            face_cascade = cv2.CascadeClassifier(cascade_path)

            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # If faces detected, return the first one
            if len(faces) > 0:
                print(f"Face detected with OpenCV: {faces[0]}")
                x, y, w_face, h_face = faces[0] # Renamed w, h to avoid conflict
                face = img[y:y+h_face, x:x+w_face]
                return face
            else:
                print("No face detected with OpenCV Haar Cascade.")
                # As a fallback, return the center portion of the image (Optional, can be removed)
                # h_img_center, w_img_center = img.shape[:2]
                # center_x, center_y = w_img_center // 2, h_img_center // 2
                # size = min(w_img_center, h_img_center) // 2
                # face = img[center_y-size:center_y+size, center_x-size:center_x+size]
                # print(f"Using center portion as fallback: {center_x-size}, {center_y-size}, {center_x+size}, {center_y+size}")
                # return face
                return None # Prefer None if no face truly detected
        except Exception as e:
            print(f"Error during OpenCV face detection: {e}")
            return None

    def extract_features(self, face_image):
        """Extract features using Facenet512 via DeepFace."""
        if self.feature_extractor_model_name is None:
            print("Facenet512 model not loaded. Cannot extract features.")
            return None
            
        if face_image is None or face_image.size == 0: # Check if face_image is valid
            print("Cannot extract features from None or empty image.")
            return None
        
        try:
            print(f"Extracting features using {self.feature_extractor_model_name}...")
            # DeepFace.represent expects BGR numpy array or image path.
            # face_image from detect_face is already a BGR numpy array.
            embedding_objs = DeepFace.represent(
                img_path=face_image,
                model_name=self.feature_extractor_model_name,
                enforce_detection=False, # We've already detected the face
                detector_backend='skip'  # Skip DeepFace's internal detection
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                # DeepFace.represent returns a list of dicts (one per face if multiple found, though we send one crop)
                # For a single crop with enforce_detection=False, it should be one item.
                embedding = embedding_objs[0]['embedding']
                print(f"Extracted {self.feature_extractor_model_name} feature vector with length {len(embedding)}.")
                return np.array(embedding) # Ensure it's a numpy array
            else:
                print(f"{self.feature_extractor_model_name} feature extraction did not return embeddings.")
                return None
        except Exception as e:
            print(f"Error extracting features with {self.feature_extractor_model_name}: {e}")
            return None

    def _save_features_local_fallback(self, user_id, features):
        """Fallback to save features to a local .npy file."""
        try:
            # Ensure features are a numpy array for saving
            if not isinstance(features, np.ndarray):
                features = np.array(features)

            os.makedirs("user_features_facenet512_local", exist_ok=True) # Directory for local fallback
            filepath = os.path.join("user_features_facenet512_local", f"{user_id}_features.npy")
            np.save(filepath, features)
            print(f"Saved features to local fallback file: {filepath}")
        except Exception as e_save:
            print(f"Error saving features to local fallback file: {e_save}")


    def _load_features_local_fallback(self, user_id):
        """Fallback to load features from a local .npy file."""
        try:
            filepath = os.path.join("user_features_facenet512_local", f"{user_id}_features.npy")
            if os.path.exists(filepath):
                features = np.load(filepath)
                print(f"Loaded features from local fallback file: {filepath}")
                return features
            return None
        except Exception as e_load:
            print(f"Error loading features from local fallback file: {e_load}")
            return None

    def save_to_database(self, user_id, features):
        """Save face features to MongoDB, with local file fallback."""
        if features is None:
            print(f"Cannot save None features for user {user_id}.")
            return

        # Ensure features are in list format for MongoDB
        features_list = features.tolist() if isinstance(features, np.ndarray) else features

        if self.collection is not None:
            try:
                self.collection.update_one(
                    {'user_id': user_id},
                    {'$set': {'features': features_list}},
                    upsert=True
                )
                print(f"Features saved to database for user: {user_id}")
                return # Successfully saved to DB
            except Exception as e:
                print(f"Error saving to database: {e}. Falling back to local file.")
        
        # Fallback to local file storage
        self._save_features_local_fallback(user_id, features)


    def verify_face(self, image_path, user_id):
        """Verify if the face in the image matches the stored face using Facenet512 features."""
        # Detect face in the new image
        face = self.detect_face(image_path)
        if face is None:
            return False, {"message": "No face detected in image.", "similarity_score": 0.0, "threshold": 0.70, "is_match": False}

        # Extract features
        features = self.extract_features(face)
        if features is None:
            return False, {"message": "Could not extract features from image.", "similarity_score": 0.0, "threshold": 0.70, "is_match": False}

        # Get stored features
        stored_features = None
        if self.collection is not None:
            try:
                stored_data = self.collection.find_one({'user_id': user_id})
                if stored_data is not None and 'features' in stored_data:
                    stored_features = np.array(stored_data['features'])
                    print(f"Retrieved features from database for user: {user_id}")
            except Exception as e:
                print(f"Error retrieving from database: {e}. Attempting local fallback.")
        
        # Try loading from local fallback file if not found in database or DB error
        if stored_features is None:
            stored_features = self._load_features_local_fallback(user_id)
            if stored_features is None:
                 return False, {"message": f"User {user_id} not found or no features stored.", "similarity_score": 0.0, "threshold": 0.70, "is_match": False}
            else:
                print(f"Loaded features from local fallback for user: {user_id}")
        
        # Calculate cosine similarity
        def cosine_similarity(a, b):
            # Ensure inputs are numpy arrays for dot product and norm
            a_np = np.asarray(a)
            b_np = np.asarray(b)
            dot_product = np.dot(a_np, b_np)
            norm_a = np.linalg.norm(a_np)
            norm_b = np.linalg.norm(b_np)
            if norm_a == 0 or norm_b == 0: # Handle zero vectors
                return 0.0
            return dot_product / (norm_a * norm_b)
            
        cosine_sim = cosine_similarity(features, stored_features)
        print(f"Cosine similarity: {cosine_sim:.4f}")
        
        # Determine match based on cosine similarity
        threshold = 0.70  # Threshold for Facenet512 with cosine similarity
        is_match = cosine_sim >= threshold # Using >= for similarity
        print(f"Match determination: {cosine_sim:.4f} >= {threshold} = {is_match}")
        
        return is_match, {
            'similarity_score': float(cosine_sim), # Ensure float for JSON if needed
            'threshold': threshold,
            'is_match': is_match
        } 
# Example Usage (you can adapt or extend this for your main application flow)
if __name__ == '__main__':
    verifier = FaceVerification()

    # --- Create Dummy Images for Testing (replace with real images) ---
    def create_dummy_image(filepath, color=(0, 255, 0), text="dummy"):
        if not os.path.exists(os.path.dirname(filepath)) and os.path.dirname(filepath) != "":
            os.makedirs(os.path.dirname(filepath))

        if not os.path.exists(filepath):
            img = np.zeros((250, 250, 3), dtype=np.uint8)
            cv2.rectangle(img, (30, 30), (220, 220), color, -1) # A large colored area
            # Add some text to distinguish, though detection models won't use this
            cv2.putText(img, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.imwrite(filepath, img)
            print(f"Created dummy image: {filepath} (USE REAL FACE IMAGES FOR ACTUAL TESTING)")

    # IMPORTANT: Replace these paths with actual images containing faces for meaningful testing.
    # The dummy images will likely fail face detection or yield meaningless results.
    test_image_dir = "test_images_main_app" 
    img_person_a1 = os.path.join(test_image_dir, "personA_enroll.jpg")
    img_person_a2 = os.path.join(test_image_dir, "personA_verify.jpg")
    img_person_b1 = os.path.join(test_image_dir, "personB_verify.jpg")

    create_dummy_image(img_person_a1, (255, 0, 0), "Person A1") # Red
    create_dummy_image(img_person_a2, (200, 50, 50), "Person A2") # Darker Red
    create_dummy_image(img_person_b1, (0, 0, 255), "Person B1") # Blue

    # Prompt to replace dummy images
    input(f"Dummy images created in '{os.path.abspath(test_image_dir)}'. "
          f"PLEASE REPLACE THEM WITH REAL FACE IMAGES and press Enter to continue test...")

    user_id_a = "user_main_app_A"
    
    # 1. Enroll User A
    print(f"\\n--- Attempting to enroll {user_id_a} using {img_person_a1} ---")
    # For enrollment, we usually just need to detect face and save features
    face_to_enroll = verifier.detect_face(img_person_a1)
    if face_to_enroll is not None:
        features_to_enroll = verifier.extract_features(face_to_enroll)
        if features_to_enroll is not None:
            verifier.save_to_database(user_id_a, features_to_enroll)
            print(f"Enrollment successful for {user_id_a}.")
        else:
            print(f"Enrollment failed for {user_id_a}: Could not extract features.")
    else:
        print(f"Enrollment failed for {user_id_a}: No face detected in {img_person_a1}.")

    # 2. Verify User A with a different image of User A
    print(f"\\n--- Verifying {user_id_a} with image {img_person_a2} (should be a match) ---")
    is_match_aa, details_aa = verifier.verify_face(img_person_a2, user_id_a)
    print(f"Verification for {user_id_a} (same person): Match = {is_match_aa}, Details = {details_aa}")

    # 3. Verify User B against User A's enrollment
    print(f"\\n--- Verifying with image {img_person_b1} against {user_id_a} (should NOT be a match) ---")
    is_match_ab, details_ab = verifier.verify_face(img_person_b1, user_id_a)
    print(f"Verification for different person against {user_id_a}: Match = {is_match_ab}, Details = {details_ab}")

    print("\\n--- Main app test complete ---")
    print("Check console output for detection, extraction, and verification steps.")
    print("If MongoDB is connected, features are in 'face_features_facenet512'.")
    print("Otherwise, check 'user_features_facenet512_local' directory for .npy files.") 