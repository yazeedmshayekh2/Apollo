from face_verification import FaceVerification
import os
import cv2
import numpy as np
from datetime import datetime

def test_face_verification():
    print("Initializing Face Verification System...")
    try:
        face_verifier = FaceVerification()
    except Exception as e:
        print(f"Error initializing Face Verification System: {str(e)}")
        return

    # Test image paths
    id_card_image = "sample_images/id_card.jpg"
    test_image = "sample_images/test_face.jpg"
    
    # Verify images exist
    if not os.path.exists(id_card_image):
        print(f"Error: ID card image not found at {id_card_image}")
        return
    if not os.path.exists(test_image):
        print(f"Error: Test image not found at {test_image}")
        return

    # Test user ID with timestamp to avoid conflicts
    test_user_id = f"test_user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\nStep 1: Processing ID card image...")
    # Detect face in ID card
    face = face_verifier.detect_face(id_card_image)
    if face is None:
        print("No face detected in ID card image")
        return
    
    # Save detected face for verification
    face_path = "sample_images/detected_face.jpg"
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
    face_verifier.save_to_database(test_user_id, features)
    print(f"Features saved to database for user: {test_user_id}")
    
    # Step 2: Test face verification
    print("\nStep 2: Testing face verification...")
    is_match, result = face_verifier.verify_face(test_image, test_user_id)
    
    if isinstance(result, str):
        print(f"Error during verification: {result}")
    else:
        print("\nVerification Results:")
        print(f"Match: {is_match}")
        
        # Print score and threshold
        print(f"Similarity Score: {result['similarity_score']:.4f} (threshold: {result.get('threshold', 0.7):.2f})")
        
        # Additional analysis
        print("\nAnalysis:")
        threshold = result.get('threshold', 0.7)
        if result['similarity_score'] > threshold:
            print(f"- Cosine similarity indicates a match ({result['similarity_score']:.4f} > {threshold:.2f})")
        else:
            print(f"- Cosine similarity indicates no match ({result['similarity_score']:.4f} <= {threshold:.2f})")

if __name__ == "__main__":
    test_face_verification() 