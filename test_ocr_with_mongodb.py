#!/usr/bin/env python3
"""
Test script to process images through the OCR API and save results to MongoDB.
"""

import os
import sys
import logging
import json
import uuid
import requests
from pathlib import Path
import time
from pprint import pprint
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import database modules (for verification)
try:
    from database import db, ocr_store
    logger.info("Successfully imported database modules")
except ImportError as e:
    logger.error(f"Error importing database modules: {e}")
    sys.exit(1)

# Configuration
API_BASE_URL = "http://localhost:8000/api"
SAMPLE_IMAGES_DIR = "sample_data/test_images"

def process_document(front_image_path, back_image_path=None, user_id=None):
    """
    Process document through API and save to MongoDB.
    """
    # Create API endpoint URL
    url = f"{API_BASE_URL}/process"
    
    # Prepare files
    files = {
        'front': (os.path.basename(front_image_path), open(front_image_path, 'rb'), 'image/jpeg')
    }
    
    # Add back image if provided
    if back_image_path:
        files['back'] = (os.path.basename(back_image_path), open(back_image_path, 'rb'), 'image/jpeg')
    
    # Prepare data
    data = {
        'output_format': 'json'
    }
    
    # Add user_id if provided
    if user_id:
        data['user_id'] = user_id
    
    # Submit API request
    logger.info(f"Submitting document for processing with user_id={user_id}")
    response = requests.post(url, files=files, data=data)
    
    # Check for success
    if response.status_code == 200:
        job_data = response.json()
        job_id = job_data['job_id']
        logger.info(f"Document submitted successfully, job_id={job_id}")
        return job_id
    else:
        logger.error(f"Failed to submit document: {response.status_code} {response.text}")
        return None

def check_job_status(job_id):
    """
    Check status of a processing job.
    """
    url = f"{API_BASE_URL}/jobs/{job_id}"
    
    max_attempts = 30
    wait_seconds = 2
    
    for attempt in range(max_attempts):
        # Get job status
        response = requests.get(url)
        
        if response.status_code == 200:
            job_data = response.json()
            status = job_data['status']
            
            if status in ['completed', 'completed_with_warnings']:
                logger.info(f"Job {job_id} completed with status: {status}")
                return job_data
            elif status == 'failed':
                logger.error(f"Job {job_id} failed: {job_data.get('error', 'Unknown error')}")
                return None
            else:
                logger.info(f"Job {job_id} status: {status}, waiting...")
                time.sleep(wait_seconds)
        else:
            logger.error(f"Failed to check job status: {response.status_code} {response.text}")
            return None
    
    logger.error(f"Timed out waiting for job {job_id} to complete")
    return None

def verify_mongodb_data(user_id, job_id):
    """
    Verify data was saved to MongoDB.
    """
    # Try to find documents for the user
    documents = ocr_store.get_user_documents(user_id)
    
    if not documents:
        logger.error(f"No documents found in MongoDB for user_id={user_id}")
        return False
    
    # Find the specific document for this job
    job_documents = [doc for doc in documents if doc.get('job_id') == job_id]
    
    if not job_documents:
        logger.error(f"No document found in MongoDB for job_id={job_id}")
        return False
    
    # Print the document structure (without embedding)
    logger.info("Found document in MongoDB:")
    
    # Convert ObjectId to string for printing
    doc = job_documents[0]
    if '_id' in doc:
        doc['_id'] = str(doc['_id'])
    
    # Remove binary embedding for cleaner output
    if 'embedding' in doc:
        doc['embedding'] = f"<Binary data, {doc.get('embedding_dim', 0)} dimensions>"
    
    # Pretty print the document structure
    pprint(doc)
    
    return True

def main():
    """
    Run the test.
    """
    # Generate a test user ID
    user_id = f"test_user_{uuid.uuid4()}"
    logger.info(f"Using test user ID: {user_id}")
    
    # Use test images
    front_image = os.path.join(SAMPLE_IMAGES_DIR, "iqama-1-front.jpg")
    back_image = os.path.join(SAMPLE_IMAGES_DIR, "iqama-1-back.jpg")
    
    # Check if images exist
    if not os.path.exists(front_image) or not os.path.exists(back_image):
        logger.error(f"Sample images not found. Please check the paths: {front_image}, {back_image}")
        return
    
    # Process the document
    job_id = process_document(front_image, back_image, user_id)
    if not job_id:
        return
    
    # Check job status
    result = check_job_status(job_id)
    if not result:
        return
    
    # Wait a moment to ensure async MongoDB operations complete
    time.sleep(2)
    
    # Verify data in MongoDB
    verify_mongodb_data(user_id, job_id)

if __name__ == "__main__":
    main() 