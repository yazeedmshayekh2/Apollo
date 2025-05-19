"""
Upload Handler for Document Images

This module handles file uploads, storing document images
and creating appropriate directories for job outputs.
"""

import os
import logging
import uuid
import mimetypes
from typing import Optional, Tuple, List
from fastapi import UploadFile, HTTPException
from PIL import Image
import io

from utils.config import Config

# Configure logging
logger = logging.getLogger(__name__)

class UploadHandler:
    """
    Handles the upload and storage of document images.
    Supports different document types and sides (front/back).
    """
    
    def __init__(self, job_id: str = None):
        """
        Initialize the upload handler.
        
        Args:
            job_id: Optional job ID to associate with the uploads
        """
        self.job_id = job_id or str(uuid.uuid4())
        self.storage_path = Config.get("LOCAL_STORAGE_PATH", "storage")
        self.job_dir = os.path.join(self.storage_path, self.job_id)
        self.output_dir = os.path.join(Config.get("DEFAULT_OUTPUT_DIR", "output"), self.job_id)
        
        # Create directories if they don't exist
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.job_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.debug(f"Initialized upload handler for job {self.job_id}")
    
    async def save_file(self, file: UploadFile, side: str = "front") -> str:
        """
        Save an uploaded file to the storage directory.
        
        Args:
            file: Uploaded file from FastAPI
            side: Which side of the document ("front" or "back")
            
        Returns:
            Path to the saved file
        """
        try:
            # Check if file is valid
            if not file.filename:
                raise ValueError("Filename cannot be empty")
                
            # Read file content
            content = await file.read()
            
            # Verify it's a valid image
            try:
                img = Image.open(io.BytesIO(content))
                width, height = img.size
                logger.debug(f"Received image: {file.filename} ({width}x{height})")
            except Exception as e:
                raise ValueError(f"Invalid image file: {e}")
            
            # Create a filename with side information
            ext = os.path.splitext(file.filename)[1].lower()
            filename = f"{self.job_id}_{side}{ext}"
            file_path = os.path.join(self.job_dir, filename)
            
            # Save the file
            with open(file_path, "wb") as f:
                f.write(content)
                
            logger.info(f"Saved {side} image to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=f"Error processing upload: {str(e)}")
    
    @classmethod
    def create_job_directory(cls, job_id: str) -> str:
        """
        Create directories for a given job ID.
        
        Args:
            job_id: Job ID for directory creation
            
        Returns:
            Path to output directory
        """
        storage_path = Config.get("LOCAL_STORAGE_PATH", "storage")
        output_dir = os.path.join(Config.get("DEFAULT_OUTPUT_DIR", "output"), job_id)
        
        # Create storage directory for this job
        job_storage_dir = os.path.join(storage_path, job_id)
        os.makedirs(job_storage_dir, exist_ok=True)
        
        # Create output directory for this job
        os.makedirs(output_dir, exist_ok=True)
        
        logger.debug(f"Created directories for job {job_id}")
        return output_dir
    
    @staticmethod
    def validate_file_type(content_type: str) -> bool:
        """
        Validate that the file is an allowed image type.
        
        Args:
            content_type: MIME type of the file
            
        Returns:
            True if valid image type, False otherwise
        """
        allowed_types = [
            'image/jpeg',
            'image/jpg',
            'image/png',
            'image/bmp',
            'image/tiff',
            'image/webp',
            'application/pdf'
        ]
        
        return content_type in allowed_types
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """
        Get the extension from a filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            File extension (including the dot)
        """
        ext = os.path.splitext(filename)[1].lower()
        
        # If no extension is provided, guess based on mimetype
        if not ext:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type:
                ext = mimetypes.guess_extension(mime_type) or '.dat'
                
        return ext or '.dat'  # Default to .dat if no extension found
    
    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """
        Check if a file is a valid image by attempting to open it.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except Exception:
            return False
