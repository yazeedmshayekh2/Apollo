#!/usr/bin/env python3
"""
Script to test the API endpoint directly with curl-like requests.
"""

import requests
import json
import uuid
import os
import time
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

def get_id_number_from_job(job_id):
    """Extract the ID number from a job result."""
    url = f"{API_BASE_URL}/jobs/{job_id}"
    
    # Wait a bit for processing to complete
    print("Waiting for processing to complete...")
    for _ in range(30):  # Try for up to 30 seconds
        response = requests.get(url)
        data = response.json()
        
        if data.get('status') in ('completed', 'completed_with_warnings'):
            # Processing is complete
            break
        
        if data.get('status') == 'failed':
            print("Processing failed!")
            return None
            
        print(".", end='', flush=True)
        time.sleep(1)
    
    print("\nProcessing completed")
    
    # Get the full job result
    response = requests.get(f"{API_BASE_URL}/jobs/{job_id}")
    
    if response.status_code != 200:
        print(f"Error retrieving job status: {response.status_code}")
        return None
    
    job_data = response.json()
    
    # Look for ID number in the extracted data
    id_number = None
    
    # First check if it's in personal_info.id_number
    if 'extracted_json' in job_data and isinstance(job_data['extracted_json'], dict):
        extracted_data = job_data['extracted_json']
        if 'personal_info' in extracted_data and 'id_number' in extracted_data['personal_info']:
            id_number = extracted_data['personal_info']['id_number']
    
    # If not found and we have extracted_json_str, try parsing that
    if not id_number and 'extracted_json_str' in job_data:
        try:
            extracted_data = json.loads(job_data['extracted_json_str'])
            if 'personal_info' in extracted_data and 'id_number' in extracted_data['personal_info']:
                id_number = extracted_data['personal_info']['id_number']
        except:
            pass
    
    if id_number:
        print(f"Found ID number: {id_number}")
    else:
        print("ID number not found in processing results")
        
    return id_number

def test_get_document_by_id_number(id_number):
    """Test retrieving a document by ID number."""
    if not id_number:
        print("No ID number provided for testing document retrieval")
        return
    
    url = f"{API_BASE_URL}/documents/id/{id_number}"
    
    print(f"Testing /documents/id/{id_number} endpoint")
    
    # Make the request
    response = requests.get(url)
    
    # Process response
    print(f"Response status code: {response.status_code}")
    
    try:
        if response.status_code == 200:
            response_data = response.json()
            print("Successfully retrieved document by ID number!")
            
            # Print document ID and extracted data for verification
            document_id = response_data.get('document_id', 'Not found')
            user_id = response_data.get('user_id', 'Not found')
            
            print(f"Document ID: {document_id}")
            print(f"User ID: {user_id}")
            
            # Verify the document ID matches the ID number
            if document_id == id_number:
                print("✓ SUCCESS: Document ID matches the ID number")
            else:
                print("✗ ERROR: Document ID does not match the ID number")
                
            return response_data
        else:
            print(f"Failed to retrieve document by ID number: {response.status_code}")
            if response.status_code == 404:
                print("Document not found - ID number might not be used for storage")
            return None
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

def check_mongodb_directly(user_id=None, id_number=None):
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
        
        # Check the OCR data collection
        ocr_data = db["ocr_data"]
        
        # Check if we have the document by ID number
        if id_number:
            print(f"Looking for document with document_id={id_number}")
            doc_by_id = ocr_data.find_one({"document_id": id_number})
            
            if doc_by_id:
                print(f"✓ SUCCESS: Found document with document_id={id_number}")
                # Remove binary data for cleaner output
                if "embedding" in doc_by_id:
                    doc_by_id["embedding"] = f"<Binary data, {doc_by_id.get('embedding_dim', 0)} dimensions>"
                print("Document sample (by ID number):")
                pprint(doc_by_id)
            else:
                print(f"✗ ERROR: No document found with document_id={id_number}")
                
                # Try looking for it in the extracted data
                print(f"Looking for document with extracted_data.personal_info.id_number={id_number}")
                doc_by_extracted = ocr_data.find_one({"extracted_data.personal_info.id_number": id_number}) 
                
                if doc_by_extracted:
                    print(f"✓ Found document with ID number in extracted data")
                    print(f"Document ID: {doc_by_extracted.get('document_id')}")
                    
        # Check if we have the test user in the OCR data collection
        if user_id:
            docs = list(ocr_data.find({"user_id": user_id}))
            
            print(f"Found {len(docs)} documents for user_id={user_id}")
            
            # Check a few more things
            if len(docs) > 0:
                print("Document sample by user_id:")
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
        
        # Get the ID number extracted from the processed document
        id_number = get_id_number_from_job(job_id)
        
        if id_number:
            # Test retrieving the document by ID number
            test_get_document_by_id_number(id_number)
            
            # Check MongoDB directly to verify ID number is used as document_id
            check_mongodb_directly(TEST_USER_ID, id_number)
        else:
            # Still check user documents endpoint if ID number not found
            test_get_user_documents(TEST_USER_ID)
            check_mongodb_directly(TEST_USER_ID)
    else:
        print("Failed to submit document for processing") 