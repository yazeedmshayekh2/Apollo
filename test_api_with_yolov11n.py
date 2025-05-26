#!/usr/bin/env python3
"""
Test script for the integrated OCR + Face Verification API with YOLOv11n face detection
"""

import requests
import json
import os
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"

def test_document_processing_with_face():
    """Test the document processing with face extraction endpoint"""
    print("=" * 60)
    print("Testing Document Processing with YOLOv11n Face Detection")
    print("=" * 60)
    
    # Check if sample image exists
    sample_image = "sample_images/front_iqama-1-front.jpg"
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        return False, None
    
    print(f"📄 Processing document: {sample_image}")
    
    # Prepare the request
    url = f"{BASE_URL}/api/process-document-with-face"
    
    with open(sample_image, 'rb') as f:
        files = {
            'front_file': ('id_card.jpg', f, 'image/jpeg')
        }
        data = {
            'document_type': 'residency'
        }
        
        print("🚀 Sending request to API...")
        try:
            response = requests.post(url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Request successful!")
                print("\n📊 Response Summary:")
                print(f"  - Success: {result.get('success', False)}")
                print(f"  - Person ID: {result.get('person_id', 'N/A')}")
                print(f"  - Message: {result.get('message', 'N/A')}")
                
                # Face verification details
                face_verification = result.get('face_verification', {})
                if face_verification:
                    print(f"\n👤 Face Verification:")
                    print(f"  - Face detected: {face_verification.get('face_detected', False)}")
                    print(f"  - Features extracted: {face_verification.get('features_extracted', False)}")
                    print(f"  - Embedding method: {face_verification.get('embedding_method', 'N/A')}")
                    print(f"  - Face image path: {face_verification.get('face_image_path', 'N/A')}")
                
                # OCR results summary
                ocr_result = result.get('ocr_result', {})
                if ocr_result:
                    personal_info = ocr_result.get('personal_info', {})
                    print(f"\n📋 OCR Results:")
                    print(f"  - Name: {personal_info.get('name', 'N/A')}")
                    print(f"  - ID Number: {personal_info.get('id_number', 'N/A')}")
                    print(f"  - Nationality: {personal_info.get('nationality', 'N/A')}")
                    print(f"  - Date of Birth: {personal_info.get('date_of_birth', 'N/A')}")
                
                return result.get('success', False), result.get('person_id')
                
            else:
                print(f"❌ Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False, None
                
        except requests.exceptions.Timeout:
            print("❌ Request timed out")
            return False, None
        except Exception as e:
            print(f"❌ Error during request: {e}")
            return False, None

def test_person_verification(person_id):
    """Test person verification with the same image"""
    if not person_id:
        print("⚠️ Skipping verification test - no person ID")
        return
    
    print(f"\n" + "=" * 60)
    print("Testing Person Verification")
    print("=" * 60)
    
    sample_image = "sample_images/id_card.jpg"
    if not os.path.exists(sample_image):
        print(f"❌ Sample image not found: {sample_image}")
        return
    
    print(f"🔍 Verifying person {person_id} with same image...")
    
    url = f"{BASE_URL}/api/verify-person"
    
    with open(sample_image, 'rb') as f:
        files = {
            'test_image': ('test_image.jpg', f, 'image/jpeg')
        }
        data = {
            'person_id': person_id
        }
        
        try:
            response = requests.post(url, files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Verification request successful!")
                
                verification_result = result.get('verification_result', {})
                print(f"\n🎯 Verification Results:")
                print(f"  - Is Match: {verification_result.get('is_match', False)}")
                print(f"  - Similarity Score: {verification_result.get('similarity_score', 0):.4f}")
                print(f"  - Threshold: {verification_result.get('threshold', 0.7)}")
                
                person_info = result.get('person_info', {}).get('personal_info', {})
                if person_info:
                    print(f"\n👤 Person Information:")
                    print(f"  - Name: {person_info.get('name', 'N/A')}")
                    print(f"  - ID: {person_info.get('id_number', 'N/A')}")
                    print(f"  - Nationality: {person_info.get('nationality', 'N/A')}")
                
            else:
                print(f"❌ Verification failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Error during verification: {e}")

def test_person_retrieval(person_id):
    """Test retrieving person information"""
    if not person_id:
        print("⚠️ Skipping retrieval test - no person ID")
        return
    
    print(f"\n" + "=" * 60)
    print("Testing Person Retrieval")
    print("=" * 60)
    
    print(f"📖 Retrieving person information for {person_id}...")
    
    url = f"{BASE_URL}/api/person/{person_id}"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Retrieval successful!")
            
            person_info = result.get('person_info', {})
            print(f"\n📋 Person Information:")
            print(f"  - Person ID: {person_info.get('person_id', 'N/A')}")
            print(f"  - Created: {person_info.get('created_at', 'N/A')}")
            print(f"  - Status: {person_info.get('status', 'N/A')}")
            print(f"  - Has Face Embeddings: {result.get('has_face_embeddings', False)}")
            print(f"  - Embedding Method: {result.get('embedding_method', 'N/A')}")
            
            personal_info = person_info.get('personal_info', {})
            if personal_info:
                print(f"\n👤 Personal Details:")
                print(f"  - Name: {personal_info.get('name', 'N/A')}")
                print(f"  - ID Number: {personal_info.get('id_number', 'N/A')}")
                print(f"  - Nationality: {personal_info.get('nationality', 'N/A')}")
                print(f"  - DOB: {personal_info.get('date_of_birth', 'N/A')}")
                print(f"  - Occupation: {personal_info.get('occupation', 'N/A')}")
            
        else:
            print(f"❌ Retrieval failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during retrieval: {e}")

def test_search_functionality():
    """Test search functionality"""
    print(f"\n" + "=" * 60)
    print("Testing Search Functionality")
    print("=" * 60)
    
    # Test search by name
    print("🔍 Searching for persons with name 'Ahmed'...")
    
    url = f"{BASE_URL}/api/search/by-name/Ahmed"
    
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            results = result.get('results', [])
            count = result.get('count', 0)
            
            print(f"✅ Search successful! Found {count} person(s)")
            
            for i, person in enumerate(results[:3], 1):  # Show first 3 results
                person_info = person.get('person_info', {})
                personal_info = person_info.get('personal_info', {})
                print(f"\n  {i}. {personal_info.get('name', 'N/A')} (ID: {person_info.get('person_id', 'N/A')})")
                print(f"     Nationality: {personal_info.get('nationality', 'N/A')}")
                print(f"     Has Embeddings: {person.get('has_face_embeddings', False)}")
            
        else:
            print(f"❌ Search failed with status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error during search: {e}")

def main():
    """Run all tests"""
    print("🚀 Starting API Tests with YOLOv11n Face Detection")
    print("Server URL:", BASE_URL)
    
    # Test 1: Document processing with face extraction
    success, person_id = test_document_processing_with_face()
    
    if success and person_id:
        # Test 2: Person verification
        test_person_verification(person_id)
        
        # Test 3: Person retrieval
        test_person_retrieval(person_id)
    
    # Test 4: Search functionality
    test_search_functionality()
    
    print(f"\n" + "=" * 60)
    print("🎉 API Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main() 