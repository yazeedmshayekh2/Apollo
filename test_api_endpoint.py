#!/usr/bin/env python3
"""
Script to test the API endpoint directly with curl-like requests.
"""

import requests
import json
import uuid
import os
from pprint import pprint

# API endpoint
API_BASE_URL = "http://localhost:8000/api"

# Test user ID
TEST_USER_ID = f"test_direct_{uuid.uuid4()}"

# Test image path
SAMPLE_IMAGES_DIR = "sample_data/test_images"
FRONT_IMAGE = os.path.join(SAMPLE_IMAGES_DIR, "iqama-1-front.jpg")
BACK_IMAGE = os.path.join(SAMPLE_IMAGES_DIR, "iqama-1-back.jpg")

def test_process_endpoint():
    """Test the /process endpoint directly."""
    
    url = f"{API_BASE_URL}/process"
    
    # Check if image files exist
    if not os.path.exists(FRONT_IMAGE) or not os.path.exists(BACK_IMAGE):
        print(f"Error: Image files not found: {FRONT_IMAGE}, {BACK_IMAGE}")
        return
    
    # Prepare files to upload
    files = {
        'front': (os.path.basename(FRONT_IMAGE), open(FRONT_IMAGE, 'rb'), 'image/jpeg'),
    }
    
    if os.path.exists(BACK_IMAGE):
        files['back'] = (os.path.basename(BACK_IMAGE), open(BACK_IMAGE, 'rb'), 'image/jpeg')
    
    # Prepare form data
    data = {
        'output_format': 'json',
        'user_id': TEST_USER_ID
    }
    
    print(f"Testing /process endpoint with user_id={TEST_USER_ID}")
    print(f"Request data: {data}")
    
    # Make the request
    response = requests.post(url, files=files, data=data)
    
    # Process response
    print(f"Response status code: {response.status_code}")
    
    try:
        response_data = response.json()
        print("Response data:")
        pprint(response_data)
        
        # Return job ID for further testing
        return response_data.get('job_id')
    except:
        print(f"Failed to parse response as JSON: {response.text}")
        return None

def test_get_user_documents(user_id):
    """Test retrieving user documents endpoint."""
    
    url = f"{API_BASE_URL}/users/{user_id}/documents"
    
    print(f"Testing /users/{user_id}/documents endpoint")
    
    # Make the request
    response = requests.get(url)
    
    # Process response
    print(f"Response status code: {response.status_code}")
    
    try:
        response_data = response.json()
        print("Response data:")
        pprint(response_data)
    except:
        print(f"Failed to parse response as JSON: {response.text}")

def check_mongodb_directly():
    """Check MongoDB directly through Python."""
    
    print("Checking MongoDB directly...")
    
    try:
        from pymongo import MongoClient
        
        # Connect to MongoDB
        client = MongoClient("mongodb://localhost:27017/")
        db = client["ocr_db"]
        
        # Check collections
        print("Collections:")
        print(db.list_collection_names())
        
        # Check if we have the test user in the OCR data collection
        ocr_data = db["ocr_data"]
        docs = list(ocr_data.find({"user_id": TEST_USER_ID}))
        
        print(f"Found {len(docs)} documents for user_id={TEST_USER_ID}")
        
        # Check a few more things
        if len(docs) > 0:
            print("Document sample:")
            doc = docs[0]
            # Remove binary data for cleaner output
            if "embedding" in doc:
                doc["embedding"] = f"<Binary data, {doc.get('embedding_dim', 0)} dimensions>"
            pprint(doc)
    
    except Exception as e:
        print(f"Error accessing MongoDB directly: {e}")

if __name__ == "__main__":
    # Test process endpoint
    job_id = test_process_endpoint()
    
    if job_id:
        print(f"Successfully submitted document with job_id={job_id}")
        print("Waiting 20 seconds for processing to complete...")
        import time
        time.sleep(20)
        
        # Check user documents endpoint
        test_get_user_documents(TEST_USER_ID)
        
        # Check MongoDB directly
        check_mongodb_directly()
    else:
        print("Failed to submit document for processing") 