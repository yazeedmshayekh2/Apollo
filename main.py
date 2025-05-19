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
from typing import Dict, Any

# Set CUDA memory management configuration to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from api.routes import app
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=Config.get("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    # Print startup banner
    print("\n========================================")
    print(" Document OCR API Server")
    print("========================================")
    print(f" Host: {args.host}")
    print(f" Port: {args.port}")
    print(f" Debug: {'Enabled' if args.debug else 'Disabled'}")
    print(f" Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    print(f" Workers: {args.workers}")
    print("========================================\n")
    
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
