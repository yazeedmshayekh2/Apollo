#!/usr/bin/env python3
"""
Simple Two-Image Face Verification Test
This script demonstrates how to test face verification with two images:
1. An ID card image (to register the person)
2. A verification image (to test if it's the same person)
"""

import requests
import os
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_two_images(id_card_image, verification_image):
    """
    Test face verification with two images
    
    Args:
        id_card_image: Path to the ID card image
        verification_image: Path to the verification image
    """
    print("🧪 Testing Face Verification with Two Images")
    print("=" * 60)
    print(f"ID Card Image: {id_card_image}")
    print(f"Verification Image: {verification_image}")
    print("=" * 60)
    
    # Check if images exist
    if not os.path.exists(id_card_image):
        print(f"❌ ID card image not found: {id_card_image}")
        return
    
    if not os.path.exists(verification_image):
        print(f"❌ Verification image not found: {verification_image}")
        return
    
    # Step 1: Register the person with ID card
    print("\n📋 Step 1: Registering person from ID card...")
    
    with open(id_card_image, 'rb') as f:
        files = {'front_file': ('id_card.jpg', f, 'image/jpeg')}
        data = {'document_type': 'residency'}
        
        response = requests.post(f"{BASE_URL}/api/process-document-with-face", 
                               files=files, data=data, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                person_id = result.get('person_id')
                print(f"✅ Person registered successfully!")
                print(f"   Person ID: {person_id}")
                
                # Show extracted information
                ocr_result = result.get('ocr_result', {})
                personal_info = ocr_result.get('personal_info', {})
                print(f"   Name: {personal_info.get('name', 'N/A')}")
                print(f"   ID Number: {personal_info.get('id_number', 'N/A')}")
                
                face_verification = result.get('face_verification', {})
                print(f"   Face Detected: {face_verification.get('face_detected', False)}")
                print(f"   Features Extracted: {face_verification.get('features_extracted', False)}")
            else:
                print(f"❌ Registration failed: {result.get('message')}")
                return
        else:
            print(f"❌ Registration request failed: {response.status_code}")
            return
    
    # Wait for database update
    print("\n⏳ Waiting for database update...")
    time.sleep(2)
    
    # Step 2: Verify with second image
    print("\n🔍 Step 2: Verifying with second image...")
    
    with open(verification_image, 'rb') as f:
        files = {'test_image': ('verification.jpg', f, 'image/jpeg')}
        data = {'person_id': person_id}
        
        response = requests.post(f"{BASE_URL}/api/verify-person", 
                               files=files, data=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                verification_result = result.get('verification_result', {})
                is_match = verification_result.get('is_match', False)
                similarity_score = verification_result.get('similarity_score', 0)
                threshold = verification_result.get('threshold', 0.7)
                
                print(f"✅ Verification completed!")
                print(f"   Result: {'✅ MATCH' if is_match else '❌ NO MATCH'}")
                print(f"   Similarity Score: {similarity_score:.4f}")
                print(f"   Threshold: {threshold}")
                
                # Interpretation
                if is_match:
                    if similarity_score > 0.9:
                        print("   📊 Very high confidence - definitely the same person")
                    elif similarity_score > 0.8:
                        print("   📊 High confidence - likely the same person")
                    else:
                        print("   📊 Above threshold - probably the same person")
                else:
                    if similarity_score < 0.3:
                        print("   📊 Very low similarity - definitely different people")
                    elif similarity_score < 0.5:
                        print("   📊 Low similarity - likely different people")
                    else:
                        print("   📊 Below threshold but somewhat similar - could be related people")
                
            else:
                print(f"❌ Verification failed: {result.get('message')}")
        else:
            print(f"❌ Verification request failed: {response.status_code}")

def main():
    """
    Main function - you can modify the image paths here
    """
    print("🚀 Two-Image Face Verification Test")
    print("Server URL:", BASE_URL)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print("✅ Server is running\n")
    except:
        print("❌ Server is not running. Please start the server first.")
        return
    
    # Example 1: Test with the same image (should match)
    print("📋 Example 1: Testing with the same image (should match)")
    if os.path.exists("sample_images/front_iqama-1-front.jpg"):
        test_two_images("sample_images/front_iqama-1-front.jpg", 
                       "/home/user/Apollo/sample_images/mohammad_jamous.jpeg")
    else:
        print("⚠️ Sample image not found. Please provide your own images.")
        
        # Interactive mode
        print("\n🎮 Interactive Mode:")
        print("Please provide paths to your images:")
        
        id_card = input("Enter path to ID card image: ").strip()
        verification = input("Enter path to verification image: ").strip()
        
        if id_card and verification:
            test_two_images(id_card, verification)
        else:
            print("❌ No images provided")

if __name__ == "__main__":
    main() 