#!/usr/bin/env python3
"""
Script to retrieve embeddings from MongoDB database and compare with new images.
"""

import logging
import sys
import argparse
import os
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from scipy.spatial.distance import cosine
import cv2
import torch
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import database modules
try:
    from database import db, ocr_store
    logger.info("Successfully imported database modules")
except ImportError as e:
    logger.error(f"Error importing database modules: {e}")
    sys.exit(1)

# Import face detector from project
try:
    from core.face_detector import FaceDetector
    logger.info("Successfully imported face detector module")
except ImportError as e:
    logger.error(f"Error importing face detector module: {e}")
    logger.warning("Will fall back to simple implementation")
    HAS_FACE_DETECTOR = False
else:
    HAS_FACE_DETECTOR = True

class SimpleProcessor:
    """A simple class for face detection and embedding extraction."""
    
    def __init__(self):
        """Initialize the processor."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.face_cascade = None
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_face_detector(self):
        """Load OpenCV face detector."""
        if self.face_cascade is not None:
            return
            
        try:
            # Use OpenCV's Haar cascade for face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                logger.error(f"Failed to load cascade classifier from {cascade_path}")
                return False
            logger.info("Loaded OpenCV Haar cascade for face detection")
            return True
        except Exception as e:
            logger.error(f"Error loading face detector: {e}")
            return False
    
    def load_feature_extractor(self):
        """Load ResNet50 model for feature extraction."""
        if self.model is not None:
            return True
            
        try:
            # Load pre-trained ResNet-50
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            
            # Remove the last fully connected layer to get features
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
            
            # Set to evaluation mode
            self.model.eval()
            
            # Move to device
            self.model = self.model.to(self.device)
            
            logger.info("Loaded ResNet-50 model for feature extraction")
            return True
        except Exception as e:
            logger.error(f"Error loading feature extraction model: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """
        Preprocess the image for face detection.
        
        Args:
            image_path: Path to the image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
                
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Equalize histogram to improve contrast
            gray = cv2.equalizeHist(gray)
            
            return {
                'original': img,
                'gray': gray
            }
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None
    
    def detect_faces(self, img_dict):
        """
        Detect faces in the image.
        
        Args:
            img_dict: Dictionary with original and grayscale images
            
        Returns:
            List of detected face regions (x, y, w, h)
        """
        if not self.load_face_detector():
            return []
            
        try:
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                img_dict['gray'],
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                logger.warning("No faces detected")
                
            # Convert to list of tuples (x, y, w, h)
            return [tuple(map(int, face)) for face in faces]
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def crop_face(self, img, face, padding=0.2):
        """
        Crop the face from the image with padding.
        
        Args:
            img: Image to crop from
            face: Face region (x, y, w, h)
            padding: Padding factor
            
        Returns:
            numpy.ndarray: Cropped face image
        """
        try:
            # Extract face coordinates
            x, y, w, h = face
            
            # Add padding
            pad_w = int(w * padding)
            pad_h = int(h * padding)
            
            # Calculate new coordinates with padding
            x1 = max(0, x - pad_w)
            y1 = max(0, y - pad_h)
            x2 = min(img.shape[1], x + w + pad_w)
            y2 = min(img.shape[0], y + h + pad_h)
            
            # Crop the face
            face_img = img[y1:y2, x1:x2]
            
            return face_img
        except Exception as e:
            logger.error(f"Error cropping face: {e}")
            return None
    
    def extract_features(self, face_img):
        """
        Extract features from a face image using ResNet-50.
        
        Args:
            face_img: Face image
            
        Returns:
            numpy.ndarray: Feature vector
        """
        if not self.load_feature_extractor():
            return None
            
        try:
            # Convert from BGR to RGB
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(face_img_rgb)
            
            # Apply transformations
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(img_tensor)
                
            # Convert to numpy array
            features = features.squeeze().cpu().numpy()
            
            return features
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def process_image(self, image_path):
        """
        Process an image to extract face embeddings.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Dict with detection results and embeddings
        """
        try:
            # Step 1: Preprocess image
            logger.info("Preprocessing image...")
            img_dict = self.preprocess_image(image_path)
            if img_dict is None:
                return None
                
            # Step 2: Detect faces
            logger.info("Detecting faces...")
            faces = self.detect_faces(img_dict)
            if not faces:
                logger.warning("No faces detected")
                # Try using the whole image if no faces are detected
                logger.info("Using whole image as fallback...")
                h, w = img_dict['original'].shape[:2]
                faces = [(0, 0, w, h)]
            
            # Get the largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            logger.info(f"Selected face region: {largest_face}")
            
            # Step 3: Crop the face
            logger.info("Cropping face region...")
            face_img = self.crop_face(img_dict['original'], largest_face)
            if face_img is None:
                return None
                
            # Draw rectangle on debug image (just for visualization)
            debug_img = img_dict['original'].copy()
            x, y, w, h = largest_face
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save debug image
            debug_path = f"{os.path.splitext(image_path)[0]}_debug.jpg"
            cv2.imwrite(debug_path, debug_img)
            logger.info(f"Saved debug image to {debug_path}")
            
            # Step 4: Extract features
            logger.info("Extracting face features...")
            features = self.extract_features(face_img)
            if features is None:
                return None
                
            logger.info(f"Extracted features with shape: {features.shape}")
            
            return {
                'detections': [
                    {
                        'box': largest_face,
                        'confidence': 1.0
                    }
                ],
                'embeddings': features,
                'face_image': face_img
            }
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return None

def get_user_embeddings(user_id: str):
    """
    Retrieve all embeddings for a user from the embeddings collection.
    
    Args:
        user_id: User ID to retrieve embeddings for
    """
    # Connect to MongoDB
    if not db.connect():
        logger.error("Failed to connect to MongoDB")
        return
    
    logger.info(f"Retrieving embeddings for user_id={user_id}")
    
    # Get embeddings from embeddings collection
    embeddings = db.get_user_embeddings(user_id)
    
    if embeddings:
        logger.info(f"Retrieved {len(embeddings)} embeddings from the embeddings collection")
        for i, emb in enumerate(embeddings):
            logger.info(f"Embedding {i+1}: shape={emb['embedding'].shape}, dim={emb['embedding_dim']}")
            if 'metadata' in emb:
                logger.info(f"Metadata: {emb['metadata']}")
    else:
        logger.info("No embeddings found in embeddings collection")

def extract_nested_embedding(doc):
    """
    Extract embedding from a nested document structure.
    
    Args:
        doc: Document retrieved from MongoDB
        
    Returns:
        tuple: (has_embedding, embedding_data, embedding_info)
    """
    # Check for top-level embedding (as in our original implementation)
    if 'embedding' in doc:
        return True, doc['embedding'], {"location": "top_level", "type": "binary"}
    
    # Check for nested embedding in extracted_data -> face_detection -> embeddings -> data
    if 'extracted_data' in doc:
        extracted_data = doc['extracted_data']
        
        # Check for face detection embeddings
        if 'face_detection' in extracted_data and extracted_data['face_detection'] is not None:
            face_detection = extracted_data['face_detection']
            
            if 'embeddings' in face_detection and face_detection['embeddings'] is not None:
                embeddings = face_detection['embeddings']
                
                if 'data' in embeddings and embeddings['data'] is not None:
                    return True, np.array(embeddings['data']), {
                        "location": "extracted_data.face_detection.embeddings.data",
                        "shape": embeddings.get('shape'),
                        "version": embeddings.get('version')
                    }
    
    # Check for other potential embedding locations
    # Add more checks here if there are other paths where embeddings might be stored
    
    return False, None, {}

def get_document_embeddings(user_id: Optional[str] = None, document_id: Optional[str] = None):
    """
    Retrieve embeddings from OCR documents.
    
    Args:
        user_id: Optional user ID to filter by
        document_id: Optional document ID to retrieve specific document
        
    Returns:
        List of tuples (document_id, embedding, metadata)
    """
    # Connect to MongoDB
    if not db.connect():
        logger.error("Failed to connect to MongoDB")
        return []
    
    results = []
    
    if document_id:
        # Get specific document
        logger.info(f"Retrieving document with ID: {document_id}")
        
        # Get the raw document (include_embedding only works for top-level embedding)
        doc = ocr_store.get_document(document_id, include_embedding=True)
        
        if doc:
            # Extract nested embedding
            has_embedding, embedding_data, embedding_info = extract_nested_embedding(doc)
            
            if has_embedding:
                logger.info(f"Document found with embedding shape: {embedding_data.shape}")
                logger.info(f"Embedding location: {embedding_info['location']}")
                if 'version' in embedding_info and embedding_info['version']:
                    logger.info(f"Embedding version: {embedding_info['version']}")
                if 'shape' in embedding_info and embedding_info['shape']:
                    logger.info(f"Declared shape: {embedding_info['shape']}")
                logger.info(f"Document type: {doc.get('document_type')}")
                logger.info(f"Created at: {doc.get('created_at')}")
                
                results.append((document_id, embedding_data, {
                    "document_type": doc.get('document_type'),
                    "created_at": doc.get('created_at'),
                    **embedding_info
                }))
            else:
                logger.info(f"Document found but has no embedding")
        else:
            logger.info(f"No document found with ID: {document_id}")
    
    elif user_id:
        # Get all documents for a user
        logger.info(f"Retrieving documents for user_id={user_id}")
        docs = ocr_store.get_user_documents(user_id, include_embeddings=True)
        
        if docs:
            docs_with_embeddings = 0
            logger.info(f"Retrieved {len(docs)} documents")
            
            for i, doc in enumerate(docs):
                # Extract nested embedding
                has_embedding, embedding_data, embedding_info = extract_nested_embedding(doc)
                
                if has_embedding:
                    docs_with_embeddings += 1
                    doc_id = doc.get('document_id')
                    logger.info(f"Document {i+1}: ID={doc_id}, " 
                               f"type={doc.get('document_type')}, "
                               f"embedding shape={embedding_data.shape}, "
                               f"location={embedding_info['location']}")
                    
                    results.append((doc_id, embedding_data, {
                        "document_type": doc.get('document_type'),
                        "created_at": doc.get('created_at'),
                        **embedding_info
                    }))
            
            logger.info(f"Found {docs_with_embeddings} documents with embeddings")
        else:
            logger.info(f"No documents found for user_id={user_id}")
    
    else:
        logger.error("Either user_id or document_id must be provided")
        
    return results

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Similarity score (1.0 = identical, 0.0 = completely different)
    """
    # Ensure embeddings are the right shape
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
    
    # Compute cosine similarity (1 - cosine distance)
    similarity = 1.0 - cosine(embedding1, embedding2)
    return similarity

def process_image_and_compare(image_path, document_ids=None, user_id=None, similarity_threshold=0.8, use_simple=False):
    """
    Process an image, extract embeddings, and compare with stored embeddings.
    
    Args:
        image_path: Path to the image
        document_ids: Optional list of specific document IDs to compare with
        user_id: Optional user ID to compare with all their documents
        similarity_threshold: Threshold for similarity (0.0 to 1.0)
        use_simple: Whether to use the simple processor
        
    Returns:
        List of matching documents with similarity scores
    """
    # Check if image exists
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return []
    
    # Process the image to extract embeddings
    face_embedding = None
    
    if use_simple or not HAS_FACE_DETECTOR:
        # Use the simple processor
        logger.info("Using simple processor for face detection and feature extraction")
        processor = SimpleProcessor()
        result = processor.process_image(image_path)
        
        if result and 'embeddings' in result:
            face_embedding = result['embeddings']
        else:
            logger.error("Failed to process image with simple processor")
            return []
    else:
        # Use the project's face detector
        try:
            logger.info("Using project's FaceDetector")
            face_detector = FaceDetector()
            
            # Process image and detect faces
            logger.info(f"Processing image: {image_path}")
            result = face_detector.detect_face(image_path)
            
            # Check if faces were detected
            if not result or 'detections' not in result or not result['detections']:
                logger.error("No faces detected in the image")
                return []
            
            # Check if embeddings were extracted
            if 'embeddings' not in result or result['embeddings'] is None:
                logger.error("Failed to extract embeddings from the image")
                return []
            
            # Get the embeddings
            face_embedding = result['embeddings']
            
            # Convert embeddings to NumPy array if it's not already one
            if isinstance(face_embedding, dict) and 'data' in face_embedding:
                face_embedding = np.array(face_embedding['data'])
                logger.info(f"Converted dictionary embedding to array with shape: {face_embedding.shape}")
            elif isinstance(face_embedding, dict):
                logger.error(f"Embedding has unexpected format: {list(face_embedding.keys())}")
                return []
            
        except Exception as e:
            logger.error(f"Error processing image with FaceDetector: {e}")
            return []
    
    logger.info(f"Successfully extracted embedding with shape: {face_embedding.shape}")
    
    # Get stored embeddings
    stored_embeddings = []
    
    if document_ids:
        for doc_id in document_ids:
            # Get embeddings for specific documents
            results = get_document_embeddings(document_id=doc_id)
            stored_embeddings.extend(results)
    elif user_id:
        # Get all embeddings for a user
        stored_embeddings = get_document_embeddings(user_id=user_id)
    else:
        logger.error("Either document_ids or user_id must be provided")
        return []
    
    if not stored_embeddings:
        logger.error("No stored embeddings found for comparison")
        return []
    
    # Compare with stored embeddings
    matches = []
    
    for doc_id, stored_embedding, metadata in stored_embeddings:
        similarity = compute_similarity(face_embedding, stored_embedding)
        
        match_info = {
            "document_id": doc_id,
            "similarity": similarity,
            "metadata": metadata
        }
        
        if similarity >= similarity_threshold:
            logger.info(f"MATCH FOUND: Document {doc_id} with similarity {similarity:.4f}")
            matches.append(match_info)
        else:
            logger.info(f"No match: Document {doc_id} with similarity {similarity:.4f}")
    
    # Sort matches by similarity (highest first)
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    return matches

def main():
    """Run the main program."""
    parser = argparse.ArgumentParser(description="Retrieve and compare embeddings from MongoDB")
    
    # Basic options
    parser.add_argument("--user-id", help="User ID to retrieve embeddings for")
    parser.add_argument("--document-id", help="Document ID to retrieve embedding for")
    parser.add_argument("--collection", choices=["embeddings", "documents", "both"], 
                        default="both", help="Collection to retrieve embeddings from")
    
    # Image comparison options
    parser.add_argument("--image", help="Path to an image for embedding extraction and comparison")
    parser.add_argument("--threshold", type=float, default=0.8, 
                        help="Similarity threshold (0.0 to 1.0, default: 0.8)")
    parser.add_argument("--simple", action="store_true", 
                        help="Use simple processor instead of FaceDetector")
    
    args = parser.parse_args()
    
    # Check for valid operation modes
    if args.image:
        # Image comparison mode
        if not (args.user_id or args.document_id):
            parser.error("When using --image, either --user-id or --document-id must be provided")
        
        logger.info(f"Processing image and comparing with stored embeddings. Threshold: {args.threshold}")
        
        matches = process_image_and_compare(
            args.image,
            document_ids=[args.document_id] if args.document_id else None,
            user_id=args.user_id,
            similarity_threshold=args.threshold,
            use_simple=args.simple
        )
        
        if matches:
            logger.info(f"Found {len(matches)} matches above threshold {args.threshold}")
            for i, match in enumerate(matches):
                logger.info(f"Match {i+1}: Document {match['document_id']} with similarity {match['similarity']:.4f}")
                if 'document_type' in match['metadata']:
                    logger.info(f"  Document type: {match['metadata']['document_type']}")
                if 'created_at' in match['metadata']:
                    logger.info(f"  Created at: {match['metadata']['created_at']}")
        else:
            logger.info("No matches found above threshold")
    
    else:
        # Database query mode
        if not args.user_id and not args.document_id:
            parser.error("Either --user-id or --document-id must be provided")
        
        # Get embeddings based on specified collection
        if args.collection in ["embeddings", "both"]:
            if args.user_id:
                get_user_embeddings(args.user_id)
        
        if args.collection in ["documents", "both"]:
            get_document_embeddings(args.user_id, args.document_id)
    
    logger.info("Operation completed")
    db.close()

if __name__ == "__main__":
    main()
