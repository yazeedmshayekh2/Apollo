#!/usr/bin/env python3
"""
Document OCR System
Based on Qwen 2.5 Vision Language Model

This application provides a FastAPI server for processing various document types
including vehicle registration cards and ID cards, extracting structured information
using the Qwen 2.5 Vision model.
"""

import os
import argparse
import logging
import uvicorn
import json
from typing import Dict, Any

# Set CUDA memory management configuration to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from api.routes import app
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=Config.get("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Config.get("LOG_FILE", "ocr_system.log"))
    ]
)
logger = logging.getLogger(__name__)

def check_mongodb_connection():
    """Check MongoDB connectivity and log details about database."""
    try:
        from database import db, ocr_store
        
        # Check MongoDB connection
        is_connected = db.client.admin.command('ping')
        if is_connected:
            logger.info("✅ MongoDB connection successful")
            
            # Log database information
            collections = db.db.list_collection_names()
            logger.info(f"MongoDB database: {db.db_name}")
            logger.info(f"Available collections: {', '.join(collections)}")
            
            # Check if any documents exist in the OCR data collection
            ocr_count = ocr_store.collection.count_documents({})
            logger.info(f"OCR data collection has {ocr_count} documents")
            
            if ocr_count > 0:
                # Log a sample document to show structure (without sensitive info)
                sample = ocr_store.collection.find_one({})
                if sample:
                    # Safely extract and log document structure (mask actual values)
                    doc_structure = {}
                    if "document_id" in sample:
                        doc_structure["document_id"] = "[MASKED]"
                    if "extracted_data" in sample:
                        doc_structure["extracted_data"] = {
                            field: "[OBJECT]" for field in sample["extracted_data"].keys()
                        }
                    logger.info(f"Sample document structure: {json.dumps(doc_structure)}")
                    
                    # Log if ID numbers are used as document_id
                    if "extracted_data" in sample and "personal_info" in sample["extracted_data"]:
                        personal_info = sample["extracted_data"]["personal_info"]
                        if "id_number" in personal_info:
                            logger.info(f"Using ID numbers as document_id for MongoDB storage.")
            
            return True
        return False
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        return False

def main():
    """Main entry point for the OCR API server."""
    parser = argparse.ArgumentParser(description='Document OCR API Server')
    
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind the server to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Set up storage directories
    os.makedirs(Config.get("LOCAL_STORAGE_PATH", "storage"), exist_ok=True)
    os.makedirs(Config.get("DEFAULT_OUTPUT_DIR", "output"), exist_ok=True)
    
    # Check MongoDB connectivity
    mongodb_ok = check_mongodb_connection()
    
    # Print startup banner
    print("\n========================================")
    print(" Document OCR API Server")
    print("========================================")
    print(f" Host: {args.host}")
    print(f" Port: {args.port}")
    print(f" Debug: {'Enabled' if args.debug else 'Disabled'}")
    print(f" Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    print(f" Workers: {args.workers}")
    print(f" MongoDB: {'Connected ✅' if mongodb_ok else 'Not Connected ❌'}")
    print("========================================\n")
    
    # Log important application settings
    logger.info(f"Starting Document OCR API Server on {args.host}:{args.port}")
    logger.info(f"MongoDB connection status: {'Connected' if mongodb_ok else 'Failed'}")
    logger.info(f"Storage path: {Config.get('LOCAL_STORAGE_PATH', 'storage')}")
    logger.info(f"Output directory: {Config.get('DEFAULT_OUTPUT_DIR', 'output')}")
    
    try:
        # Start the FastAPI server
        uvicorn.run(
            "api.routes:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers,
            log_level=Config.get("LOG_LEVEL", "info").lower()
        )
        return 0
        
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
