#!/usr/bin/env python3
"""
Test script for ID card face verification
This demonstrates how to:
1. Extract a face from an ID card
2. Store the face features in the database
3. Verify a different image against the stored features
"""

import os
import cv2
import numpy as np
import argparse
from datetime import datetime
from face_verification import FaceVerification

def main():
    print("Starting ID Card Face Verification Test")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test face verification with ID cards")
    parser.add_argument("--id-card", default="sample_images/id_card.jpg", help="Path to ID card image")
    parser.add_argument("--test-face", default="sample_images/test_face.jpg", help="Path to test face image")
    parser.add_argument("--user-id", default=None, help="User ID (default: auto-generated with timestamp)")
    args = parser.parse_args()
    
    # Generate user ID if not provided
    user_id = args.user_id
    if user_id is None:
        user_id = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize face verification system
    print("\nInitializing Face Verification System...")
    face_verifier = FaceVerification()
    
    # Verify images exist
    id_card_image = args.id_card
    test_image = args.test_face
    
    if not os.path.exists(id_card_image):
        print(f"Error: ID card image not found at {id_card_image}")
        return
    if not os.path.exists(test_image):
        print(f"Error: Test image not found at {test_image}")
        return
    
    print(f"\n--- Step 1: Process ID Card ---")
    print(f"ID Card Image: {id_card_image}")
    print(f"User ID: {user_id}")
    
    # Detect face in ID card
    print("Detecting face in ID card...")
    face = face_verifier.detect_face(id_card_image)
    if face is None:
        print("No face detected in ID card image")
        return
    
    # Save detected face for reference
    face_path = f"sample_images/detected_face_{user_id}.jpg"
    cv2.imwrite(face_path, face)
    print(f"Detected face saved to {face_path}")
    
    # Extract features
    print("\nExtracting features from ID card face...")
    features = face_verifier.extract_features(face)
    if features is None:
        print("Could not extract features from ID card image")
        return
    
    print(f"Features extracted successfully. Feature vector shape: {features.shape}")
    
    # Save features to database
    print("\nSaving features to database...")
    face_verifier.save_to_database(user_id, features)
    print(f"Features saved to database for user: {user_id}")
    
    print(f"\n--- Step 2: Verify Test Face ---")
    print(f"Test Face Image: {test_image}")
    
    # Verify test face against stored features
    print("Verifying test face against stored features...")
    is_match, result = face_verifier.verify_face(test_image, user_id)
    
    # Display results
    if isinstance(result, str):
        print(f"Error during verification: {result}")
    else:
        print("\nVerification Results:")
        print(f"Match: {is_match}")
        print(f"Similarity Score: {result['similarity_score']:.4f} (threshold: {result.get('threshold', 0.7):.2f})")
        
        # Additional analysis
        print("\nAnalysis:")
        threshold = result.get('threshold', 0.7)
        if result['similarity_score'] > threshold:
            print(f"- Face verification SUCCESSFUL ({result['similarity_score']:.4f} > {threshold:.2f})")
            print("- The person in the test image matches the ID card")
        else:
            print(f"- Face verification FAILED ({result['similarity_score']:.4f} <= {threshold:.2f})")
            print("- The person in the test image does NOT match the ID card")

if __name__ == "__main__":
    main()
