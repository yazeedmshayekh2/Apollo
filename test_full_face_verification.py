#!/usr/bin/env python3
"""
Comprehensive Face Verification Test
This script tests the complete face verification pipeline:
1. Process an ID card image to extract OCR data and face embeddings
2. Verify the same person using a different image
3. Test with different people to ensure the system can distinguish between them
"""

import requests
import json
import os
from pathlib import Path
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_full_verification_pipeline(id_card_path, verification_image_path, expected_match=True):
    """
    Test the complete verification pipeline with two images
    
    Args:
        id_card_path: Path to the ID card image
        verification_image_path: Path to the verification image
        expected_match: Whether we expect these images to match (True for same person, False for different people)
    """
    print("=" * 80)
    print(f"🧪 FULL VERIFICATION TEST")
    print(f"ID Card: {id_card_path}")
    print(f"Verification Image: {verification_image_path}")
    print(f"Expected Match: {expected_match}")
    print("=" * 80)
    
    # Check if files exist
    if not os.path.exists(id_card_path):
        print(f"❌ ID card image not found: {id_card_path}")
        return False
    
    if not os.path.exists(verification_image_path):
        print(f"❌ Verification image not found: {verification_image_path}")
        return False
    
    # Step 1: Process the ID card to create a person record
    print("\n📋 STEP 1: Processing ID Card")
    print("-" * 40)
    
    url = f"{BASE_URL}/api/process-document-with-face"
    
    with open(id_card_path, 'rb') as f:
        files = {
            'front_file': ('id_card.jpg', f, 'image/jpeg')
        }
        data = {
            'document_type': 'residency'
        }
        
        print("🚀 Uploading ID card and extracting face embeddings...")
        try:
            response = requests.post(url, files=files, data=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success', False):
                    person_id = result.get('person_id')
                    print(f"✅ ID card processed successfully!")
                    print(f"   Person ID: {person_id}")
                    
                    # Display OCR results
                    ocr_result = result.get('ocr_result', {})
                    if ocr_result:
                        personal_info = ocr_result.get('personal_info', {})
                        print(f"   Name: {personal_info.get('name', 'N/A')}")
                        print(f"   ID Number: {personal_info.get('id_number', 'N/A')}")
                        print(f"   Nationality: {personal_info.get('nationality', 'N/A')}")
                    
                    # Display face verification details
                    face_verification = result.get('face_verification', {})
                    if face_verification:
                        print(f"   Face Detected: {face_verification.get('face_detected', False)}")
                        print(f"   Features Extracted: {face_verification.get('features_extracted', False)}")
                        print(f"   Embedding Method: {face_verification.get('embedding_method', 'N/A')}")
                    
                else:
                    print(f"❌ Failed to process ID card: {result.get('message', 'Unknown error')}")
                    return False
                    
            else:
                print(f"❌ Request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error during ID card processing: {e}")
            return False
    
    # Step 2: Wait a moment for the database to be updated
    print("\n⏳ Waiting for database update...")
    time.sleep(2)
    
    # Step 3: Verify with the second image
    print("\n🔍 STEP 2: Face Verification")
    print("-" * 40)
    
    url = f"{BASE_URL}/api/verify-person"
    
    with open(verification_image_path, 'rb') as f:
        files = {
            'test_image': ('verification_image.jpg', f, 'image/jpeg')
        }
        data = {
            'person_id': person_id
        }
        
        print(f"🔍 Verifying person {person_id} with verification image...")
        try:
            response = requests.post(url, files=files, data=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success', False):
                    verification_result = result.get('verification_result', {})
                    is_match = verification_result.get('is_match', False)
                    similarity_score = verification_result.get('similarity_score', 0)
                    threshold = verification_result.get('threshold', 0.7)
                    
                    print(f"✅ Verification completed!")
                    print(f"   Is Match: {is_match}")
                    print(f"   Similarity Score: {similarity_score:.4f}")
                    print(f"   Threshold: {threshold}")
                    
                    # Determine if the result matches expectations
                    if expected_match == is_match:
                        print(f"🎉 RESULT: CORRECT! Expected {expected_match}, got {is_match}")
                        result_status = "PASS"
                    else:
                        print(f"⚠️  RESULT: UNEXPECTED! Expected {expected_match}, got {is_match}")
                        result_status = "FAIL"
                    
                    # Additional analysis
                    if expected_match:
                        if similarity_score > 0.9:
                            print("   📊 Analysis: Very high confidence match")
                        elif similarity_score > threshold:
                            print("   📊 Analysis: Good match above threshold")
                        else:
                            print("   📊 Analysis: Below threshold - possible false negative")
                    else:
                        if similarity_score < 0.3:
                            print("   📊 Analysis: Very low similarity - clearly different people")
                        elif similarity_score < threshold:
                            print("   📊 Analysis: Below threshold - correctly identified as different")
                        else:
                            print("   📊 Analysis: Above threshold - possible false positive")
                    
                    return result_status == "PASS"
                    
                else:
                    print(f"❌ Verification failed: {result.get('message', 'Unknown error')}")
                    return False
                    
            else:
                print(f"❌ Verification request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error during verification: {e}")
            return False

def test_with_sample_images():
    """
    Test with sample images if available
    """
    print("\n🔍 Looking for sample images...")
    
    # Look for sample images in common directories
    sample_dirs = ["sample_images", "test_images", "images", "."]
    id_card_patterns = ["*id*", "*card*", "*iqama*", "*front*"]
    verification_patterns = ["*person*", "*face*", "*verify*", "*test*"]
    
    id_cards = []
    verification_images = []
    
    for sample_dir in sample_dirs:
        if os.path.exists(sample_dir):
            for pattern in id_card_patterns:
                id_cards.extend(Path(sample_dir).glob(f"{pattern}.jpg"))
                id_cards.extend(Path(sample_dir).glob(f"{pattern}.jpeg"))
                id_cards.extend(Path(sample_dir).glob(f"{pattern}.png"))
            
            for pattern in verification_patterns:
                verification_images.extend(Path(sample_dir).glob(f"{pattern}.jpg"))
                verification_images.extend(Path(sample_dir).glob(f"{pattern}.jpeg"))
                verification_images.extend(Path(sample_dir).glob(f"{pattern}.png"))
    
    print(f"Found {len(id_cards)} potential ID card images")
    print(f"Found {len(verification_images)} potential verification images")
    
    if id_cards:
        print("\nID Card images found:")
        for i, img in enumerate(id_cards[:5], 1):  # Show first 5
            print(f"  {i}. {img}")
    
    if verification_images:
        print("\nVerification images found:")
        for i, img in enumerate(verification_images[:5], 1):  # Show first 5
            print(f"  {i}. {img}")
    
    # If we have the specific sample image from previous tests, use it
    if os.path.exists("sample_images/front_iqama-1-front.jpg"):
        print(f"\n🎯 Using known sample image: sample_images/front_iqama-1-front.jpg")
        return "sample_images/front_iqama-1-front.jpg"
    
    return None

def interactive_test():
    """
    Interactive test where user provides image paths
    """
    print("\n" + "=" * 80)
    print("🎮 INTERACTIVE FACE VERIFICATION TEST")
    print("=" * 80)
    
    print("\nThis test will:")
    print("1. Process your ID card image to extract face embeddings")
    print("2. Verify a person using another image")
    print("3. Show whether the system correctly identifies if it's the same person")
    
    # Get ID card image path
    while True:
        id_card_path = input("\n📋 Enter path to ID card image: ").strip()
        if os.path.exists(id_card_path):
            break
        print(f"❌ File not found: {id_card_path}")
    
    # Get verification image path
    while True:
        verification_path = input("🔍 Enter path to verification image: ").strip()
        if os.path.exists(verification_path):
            break
        print(f"❌ File not found: {verification_path}")
    
    # Ask if it should be the same person
    while True:
        same_person = input("🤔 Are these images of the same person? (y/n): ").strip().lower()
        if same_person in ['y', 'yes', 'n', 'no']:
            expected_match = same_person in ['y', 'yes']
            break
        print("Please enter 'y' for yes or 'n' for no")
    
    # Run the test
    success = test_full_verification_pipeline(id_card_path, verification_path, expected_match)
    
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n❌ Test failed or produced unexpected results")
    
    return success

def automated_test_scenarios():
    """
    Run automated test scenarios with sample images
    """
    print("\n" + "=" * 80)
    print("🤖 AUTOMATED TEST SCENARIOS")
    print("=" * 80)
    
    # Test scenario 1: Same person verification
    sample_image = test_with_sample_images()
    if sample_image:
        print(f"\n📋 Scenario 1: Same person verification")
        print(f"Using the same image for both ID card and verification")
        print(f"Expected result: MATCH")
        
        success1 = test_full_verification_pipeline(sample_image, sample_image, expected_match=True)
        
        if success1:
            print("✅ Scenario 1 PASSED")
        else:
            print("❌ Scenario 1 FAILED")
    else:
        print("⚠️ No sample images found for automated testing")
        success1 = False
    
    return success1

def main():
    """
    Main test function
    """
    print("🚀 COMPREHENSIVE FACE VERIFICATION TEST SUITE")
    print("Server URL:", BASE_URL)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        print("✅ Server is running")
    except:
        print("❌ Server is not running. Please start the server first.")
        return
    
    print("\nChoose test mode:")
    print("1. Interactive test (you provide image paths)")
    print("2. Automated test (uses sample images)")
    print("3. Both")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Please enter 1, 2, or 3")
    
    results = []
    
    if choice in ['1', '3']:
        print("\n" + "🎮" * 20)
        results.append(interactive_test())
    
    if choice in ['2', '3']:
        print("\n" + "🤖" * 20)
        results.append(automated_test_scenarios())
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests PASSED! The face verification system is working correctly.")
    else:
        print("⚠️ Some tests FAILED. Please check the results above.")
    
    print("\n💡 Tips for better results:")
    print("- Use clear, well-lit images")
    print("- Ensure faces are clearly visible and not obscured")
    print("- Use images with similar lighting conditions")
    print("- Avoid heavily compressed or low-resolution images")

if __name__ == "__main__":
    main() 