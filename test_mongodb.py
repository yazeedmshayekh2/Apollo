#!/usr/bin/env python3
"""
MongoDB connection test script.
Tests the connection to MongoDB and basic operations with OCR data and embeddings.
"""

import os
import sys
import logging
import json
import uuid
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import database modules
try:
    from database import db, ocr_store, image_store, audit_log
    logger.info("Successfully imported database modules")
except ImportError as e:
    logger.error(f"Error importing database modules: {e}")
    sys.exit(1)

def test_mongodb_connection():
    """Test the MongoDB connection."""
    try:
        # Test the connection
        if db.connect():
            logger.info("Successfully connected to MongoDB")
            return True
        else:
            logger.error("Failed to connect to MongoDB")
            return False
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return False

def test_save_user_data():
    """Test saving user data to MongoDB."""
    try:
        # Generate test user ID
        user_id = f"test_user_{uuid.uuid4()}"
        
        # Create test user data
        user_data = {
            "name": "Test User",
            "email": "test@example.com",
            "created_at": datetime.now().isoformat(),
            "test_field": True
        }
        
        # Save to MongoDB
        result = db.save_user_data(user_id, user_data)
        
        if result:
            logger.info(f"Successfully saved user data for user_id={user_id}")
            
            # Verify by retrieving
            retrieved = db.get_user_data(user_id)
            if retrieved:
                logger.info(f"Successfully retrieved user data: {retrieved}")
                return user_id
            else:
                logger.error("Failed to retrieve saved user data")
                return None
        else:
            logger.error("Failed to save user data")
            return None
    
    except Exception as e:
        logger.error(f"Error in test_save_user_data: {e}")
        return None

def test_save_ocr_data(user_id):
    """Test saving OCR data with embeddings."""
    try:
        # Generate test document ID
        document_id = f"{user_id}_doc_{uuid.uuid4()}"
        
        # Create test embedding
        embedding = np.random.rand(100)  # 100-dimensional random vector
        
        # Create test extracted data
        extracted_data = {
            "document_number": "TEST12345",
            "issue_date": "2023-01-01",
            "expiry_date": "2028-01-01",
            "person": {
                "name": "John Doe",
                "date_of_birth": "1990-01-01",
                "address": "123 Test Street"
            },
            "test_field": True,
            "confidence_scores": {
                "document_number": 0.95,
                "name": 0.98
            }
        }
        
        # Save OCR data with embedding
        result = ocr_store.save_ocr_data(
            user_id=user_id,
            document_id=document_id,
            extracted_data=extracted_data,
            embedding=embedding,
            document_type="test_document",
            job_id=str(uuid.uuid4()),
            raw_text="This is a test document for John Doe with number TEST12345",
            metadata={"test": True, "purpose": "testing"}
        )
        
        if result:
            logger.info(f"Successfully saved OCR data for document_id={document_id}")
            
            # Verify by retrieving
            retrieved = ocr_store.get_document(document_id, include_embedding=True)
            if retrieved:
                logger.info(f"Successfully retrieved OCR data for document_id={document_id}")
                
                # Verify embedding
                if "embedding" in retrieved and isinstance(retrieved["embedding"], np.ndarray):
                    logger.info(f"Successfully retrieved embedding with shape {retrieved['embedding'].shape}")
                else:
                    logger.error("Failed to retrieve embedding or incorrect type")
                
                return document_id
            else:
                logger.error("Failed to retrieve saved OCR data")
                return None
        else:
            logger.error("Failed to save OCR data")
            return None
    
    except Exception as e:
        logger.error(f"Error in test_save_ocr_data: {e}")
        return None

def test_search_documents(user_id):
    """Test searching OCR documents."""
    try:
        # Search for documents
        results = ocr_store.search_user_documents(
            user_id=user_id,
            search_text="John Doe"
        )
        
        if results:
            logger.info(f"Search found {len(results)} documents")
            for doc in results:
                logger.info(f"Found document: {doc.get('document_id')}")
            return True
        else:
            logger.info("No documents found in search")
            return False
    
    except Exception as e:
        logger.error(f"Error in test_search_documents: {e}")
        return False

def test_audit_logging(user_id):
    """Test audit logging."""
    try:
        # Log an operation
        result = audit_log.log_operation(
            operation="test_operation",
            user_id=user_id,
            details={"test": True, "purpose": "testing"}
        )
        
        if result:
            logger.info("Successfully logged operation")
            
            # Retrieve logs
            logs = audit_log.get_user_logs(user_id)
            if logs:
                logger.info(f"Retrieved {len(logs)} audit logs")
                return True
            else:
                logger.error("Failed to retrieve audit logs")
                return False
        else:
            logger.error("Failed to log operation")
            return False
    
    except Exception as e:
        logger.error(f"Error in test_audit_logging: {e}")
        return False

def main():
    """Run the MongoDB tests."""
    # Test MongoDB connection
    if not test_mongodb_connection():
        logger.error("MongoDB connection test failed. Aborting further tests.")
        return
    
    # Test saving user data
    user_id = test_save_user_data()
    if not user_id:
        logger.error("User data test failed. Aborting further tests.")
        return
    
    # Test saving OCR data
    document_id = test_save_ocr_data(user_id)
    if not document_id:
        logger.error("OCR data test failed. Aborting further tests.")
    
    # Test searching documents
    test_search_documents(user_id)
    
    # Test audit logging
    test_audit_logging(user_id)
    
    logger.info("All tests completed")

if __name__ == "__main__":
    main() 