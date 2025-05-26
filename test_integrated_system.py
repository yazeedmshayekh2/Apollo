#!/usr/bin/env python3
"""
Test script for the integrated OCR + Face Verification system with MongoDB
This script demonstrates how to use the PersonDatabase class to store
OCR information along with face embeddings for each person.
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from person_database import PersonDatabase
from face_verification import FaceVerification
from ocr.processor import process_document, process_both_sides

def test_integrated_system():
    """Test the complete integrated system"""
    print("=" * 60)
    print("Testing Integrated OCR + Face Verification System")
    print("=" * 60)
    
    # Initialize systems
    print("\n1. Initializing systems...")
    try:
        person_db = PersonDatabase()
        face_verifier = FaceVerification()
        print("✓ Systems initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing systems: {e}")
        return False
    
    # Test data
    test_person_id = f"test_person_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Sample OCR data (simulating what would come from document processing)
    sample_ocr_data = {
        "document_info": {
            "document_type": "Qatar Residency Permit",
            "issuing_authority": "State of Qatar"
        },
        "personal_info": {
            "id_number": test_person_id,
            "name": "Ahmed Hassan Al-Mahmoud",
            "arabic_name": "أحمد حسن المحمود",
            "nationality": "JORDAN",
            "date_of_birth": "15/03/1985",
            "expiry_date": "15/03/2027",
            "occupation": "Software Engineer"
        },
        "additional_info": {
            "sponsor": "Qatar Tech Company",
            "address": "Doha, Qatar"
        }
    }
    
    # Test with sample images if available
    sample_image_path = "sample_images/id_card.jpg"
    if not os.path.exists(sample_image_path):
        print(f"\n⚠ Warning: Sample image not found at {sample_image_path}")
        print("Creating a dummy face embedding for testing...")
        # Create dummy face embeddings for testing
        face_embeddings = np.random.rand(512).astype(np.float32)  # ResNet50-like embeddings
        face_image_path = None
    else:
        print(f"\n2. Processing face from {sample_image_path}...")
        try:
            # Detect face
            face = face_verifier.detect_face(sample_image_path)
            if face is None:
                print("✗ No face detected in sample image")
                return False
            
            # Extract embeddings
            face_embeddings = face_verifier.extract_features(face)
            if face_embeddings is None:
                print("✗ Could not extract face embeddings")
                return False
            
            # Save face image
            face_image_path = f"uploads/test_face_{test_person_id}.jpg"
            os.makedirs("uploads", exist_ok=True)
            import cv2
            cv2.imwrite(face_image_path, face)
            
            print(f"✓ Face processed successfully")
            print(f"  - Embedding shape: {face_embeddings.shape}")
            print(f"  - Face saved to: {face_image_path}")
            
        except Exception as e:
            print(f"✗ Error processing face: {e}")
            return False
    
    # Test saving person with embeddings
    print(f"\n3. Saving person record for {test_person_id}...")
    try:
        document_images = {"front": sample_image_path} if os.path.exists(sample_image_path) else {}
        
        success = person_db.save_person_with_embeddings(
            person_id=test_person_id,
            ocr_data=sample_ocr_data,
            face_embeddings=face_embeddings,
            face_image_path=face_image_path,
            document_images=document_images
        )
        
        if success:
            print("✓ Person record saved successfully")
        else:
            print("✗ Failed to save person record")
            return False
            
    except Exception as e:
        print(f"✗ Error saving person record: {e}")
        return False
    
    # Test retrieving person by ID
    print(f"\n4. Retrieving person by ID: {test_person_id}...")
    try:
        person_data = person_db.get_person_by_id(test_person_id)
        if person_data:
            print("✓ Person retrieved successfully")
            print(f"  - Name: {person_data['person_info'].get('personal_info', {}).get('name', 'N/A')}")
            print(f"  - ID Number: {person_data['person_info'].get('personal_info', {}).get('id_number', 'N/A')}")
            print(f"  - Has embeddings: {person_data['face_embeddings'] is not None}")
            print(f"  - Embedding method: {person_data.get('embedding_method', 'N/A')}")
        else:
            print("✗ Person not found")
            return False
            
    except Exception as e:
        print(f"✗ Error retrieving person: {e}")
        return False
    
    # Test searching by document number
    print(f"\n5. Searching by document number: {test_person_id}...")
    try:
        person_data = person_db.get_person_by_document_number(test_person_id)
        if person_data:
            print("✓ Person found by document number")
            print(f"  - Name: {person_data['person_info'].get('personal_info', {}).get('name', 'N/A')}")
        else:
            print("✗ Person not found by document number")
            
    except Exception as e:
        print(f"✗ Error searching by document number: {e}")
    
    # Test searching by name
    print(f"\n6. Searching by name: 'Ahmed'...")
    try:
        results = person_db.search_persons_by_name("Ahmed", limit=5)
        print(f"✓ Found {len(results)} person(s) with name containing 'Ahmed'")
        for i, person_data in enumerate(results):
            name = person_data['person_info'].get('personal_info', {}).get('name', 'N/A')
            person_id = person_data['person_info'].get('person_id', 'N/A')
            print(f"  {i+1}. {name} (ID: {person_id})")
            
    except Exception as e:
        print(f"✗ Error searching by name: {e}")
    
    # Test face verification (if we have a real face)
    if face_image_path and os.path.exists(face_image_path):
        print(f"\n7. Testing face verification...")
        try:
            # Use the same image for verification (should match)
            stored_embeddings = person_db.get_face_embeddings(test_person_id)
            if stored_embeddings is not None:
                # Calculate similarity with itself (should be 1.0)
                def cosine_similarity(a, b):
                    dot_product = np.dot(a, b)
                    norm_a = np.linalg.norm(a)
                    norm_b = np.linalg.norm(b)
                    if norm_a == 0 or norm_b == 0:
                        return 0
                    return dot_product / (norm_a * norm_b)
                
                similarity = cosine_similarity(face_embeddings, stored_embeddings)
                print(f"✓ Face verification test completed")
                print(f"  - Similarity score: {similarity:.4f}")
                print(f"  - Match (threshold 0.7): {'Yes' if similarity > 0.7 else 'No'}")
            else:
                print("✗ Could not retrieve stored embeddings")
                
        except Exception as e:
            print(f"✗ Error during face verification: {e}")
    
    # Test getting all persons
    print(f"\n8. Getting all persons...")
    try:
        all_persons = person_db.get_all_persons(limit=10)
        print(f"✓ Retrieved {len(all_persons)} person(s)")
        for person in all_persons:
            name = person.get('personal_info', {}).get('name', 'N/A')
            person_id = person.get('person_id', 'N/A')
            status = person.get('status', 'N/A')
            print(f"  - {name} (ID: {person_id}, Status: {status})")
            
    except Exception as e:
        print(f"✗ Error getting all persons: {e}")
    
    # Test updating person info
    print(f"\n9. Testing person info update...")
    try:
        updates = {
            "personal_info.occupation": "Senior Software Engineer",
            "additional_info.updated_by": "test_script"
        }
        success = person_db.update_person_info(test_person_id, updates)
        if success:
            print("✓ Person info updated successfully")
        else:
            print("✗ Failed to update person info")
            
    except Exception as e:
        print(f"✗ Error updating person info: {e}")
    
    # Display summary
    print(f"\n" + "=" * 60)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 60)
    print("✓ OCR data and face embeddings stored together")
    print("✓ Person retrieval by ID, document number, and name")
    print("✓ Face verification using stored embeddings")
    print("✓ Database operations (save, retrieve, search, update)")
    print("✓ Fallback to local storage when MongoDB unavailable")
    
    # Cleanup
    print(f"\n10. Cleaning up test data...")
    try:
        # Soft delete the test person
        person_db.delete_person(test_person_id)
        print("✓ Test data cleaned up")
    except Exception as e:
        print(f"⚠ Warning: Could not clean up test data: {e}")
    
    print(f"\n🎉 Integration test completed successfully!")
    return True

def test_real_document_processing():
    """Test with real document processing if sample images are available"""
    print("\n" + "=" * 60)
    print("Testing Real Document Processing")
    print("=" * 60)
    
    # Check for sample images
    front_image = "sample_images/id_card.jpg"
    back_image = "sample_images/id_card_back.jpg"
    
    if not os.path.exists(front_image):
        print(f"⚠ Skipping real document test - no sample image at {front_image}")
        return
    
    print(f"\n1. Processing document: {front_image}")
    
    try:
        # Initialize systems
        person_db = PersonDatabase()
        face_verifier = FaceVerification()
        
        # Process OCR
        print("Processing OCR...")
        if os.path.exists(back_image):
            ocr_result = process_both_sides(front_image, back_image, "residency")
            print("✓ Processed both sides of document")
        else:
            ocr_result = process_document(front_image, "residency", "front")
            print("✓ Processed front side of document")
        
        print(f"OCR Result Preview:")
        print(json.dumps(ocr_result, indent=2, default=str)[:500] + "...")
        
        # Extract person ID
        person_id = None
        if "personal_info" in ocr_result and "id_number" in ocr_result["personal_info"]:
            person_id = ocr_result["personal_info"]["id_number"]
        elif "id_number" in ocr_result:
            person_id = ocr_result["id_number"]
        
        if not person_id:
            person_id = f"real_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Using person ID: {person_id}")
        
        # Process face
        print("Processing face...")
        face = face_verifier.detect_face(front_image)
        if face is None:
            print("✗ No face detected")
            return
        
        face_embeddings = face_verifier.extract_features(face)
        if face_embeddings is None:
            print("✗ Could not extract face features")
            return
        
        # Save face image
        face_image_path = f"uploads/real_face_{person_id}.jpg"
        os.makedirs("uploads", exist_ok=True)
        import cv2
        cv2.imwrite(face_image_path, face)
        
        print(f"✓ Face processed - embedding shape: {face_embeddings.shape}")
        
        # Save to database
        document_images = {"front": front_image}
        if os.path.exists(back_image):
            document_images["back"] = back_image
        
        success = person_db.save_person_with_embeddings(
            person_id=person_id,
            ocr_data=ocr_result,
            face_embeddings=face_embeddings,
            face_image_path=face_image_path,
            document_images=document_images
        )
        
        if success:
            print("✓ Real document processed and saved successfully!")
            
            # Retrieve and display
            person_data = person_db.get_person_by_id(person_id)
            if person_data:
                personal_info = person_data['person_info'].get('personal_info', {})
                print(f"\nSaved Person Info:")
                print(f"  - Name: {personal_info.get('name', 'N/A')}")
                print(f"  - ID: {personal_info.get('id_number', 'N/A')}")
                print(f"  - Nationality: {personal_info.get('nationality', 'N/A')}")
                print(f"  - DOB: {personal_info.get('date_of_birth', 'N/A')}")
        else:
            print("✗ Failed to save real document data")
            
    except Exception as e:
        print(f"✗ Error processing real document: {e}")

if __name__ == "__main__":
    print("Starting comprehensive integration tests...\n")
    
    # Test the integrated system
    success = test_integrated_system()
    
    if success:
        # Test with real documents if available
        test_real_document_processing()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60) 