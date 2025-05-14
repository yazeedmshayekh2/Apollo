"""
File upload handlers for the Vehicle Registration Card OCR system.
Provides utilities for handling file uploads, validating file types, and storing files.
"""

import os
import uuid
import shutil
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from fastapi import UploadFile, HTTPException

from utils.config import Config

logger = logging.getLogger(__name__)

# Define allowed file types
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
MAX_FILE_SIZE = Config.get("MAX_UPLOAD_SIZE", 10 * 1024 * 1024)  # Default 10MB

class UploadHandler:
    """Handler for file uploads in the OCR system."""
    
    @staticmethod
    def validate_file(upload_file: UploadFile) -> Tuple[bool, str]:
        """
        Validate the uploaded file.
        
        Args:
            upload_file: The file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file extension
        _, ext = os.path.splitext(upload_file.filename)
        if ext.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            return False, f"Invalid file type. Allowed types: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}"
        
        # Check file size (note: this requires reading the file into memory)
        # For larger files, we would use a streaming approach
        file_size = upload_file.file.tell()
        upload_file.file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            max_mb = MAX_FILE_SIZE / (1024 * 1024)
            return False, f"File too large. Maximum size: {max_mb:.1f}MB"
        
        return True, ""
    
    @staticmethod
    async def save_file(upload_file: UploadFile, directory: Optional[str] = None) -> str:
        """
        Save an uploaded file with a unique name.
        
        Args:
            upload_file: The file to save
            directory: Directory to save the file in (defaults to config setting)
            
        Returns:
            Path to the saved file
        """
        # Validate file
        is_valid, error = UploadHandler.validate_file(upload_file)
        if not is_valid:
            logger.error(f"Invalid file upload: {error}")
            raise HTTPException(status_code=400, detail=error)
        
        # Get storage directory
        storage_dir = directory or Config.get("LOCAL_STORAGE_PATH", "storage")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Generate unique filename
        _, ext = os.path.splitext(upload_file.filename)
        unique_filename = f"{uuid.uuid4()}{ext}"
        file_path = os.path.join(storage_dir, unique_filename)
        
        # Save the file
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(upload_file.file, buffer)
            
            logger.info(f"Saved uploaded file to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            raise HTTPException(status_code=500, detail="Error saving file")
    
    @staticmethod
    async def save_multiple_files(
        upload_files: List[UploadFile], 
        directory: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save multiple uploaded files.
        
        Args:
            upload_files: List of files to save
            directory: Directory to save the files in
            
        Returns:
            Dictionary mapping original filenames to saved paths
        """
        result = {}
        
        for upload_file in upload_files:
            saved_path = await UploadHandler.save_file(upload_file, directory)
            result[upload_file.filename] = saved_path
        
        return result
    
    @staticmethod
    def create_job_directory(job_id: str, base_dir: Optional[str] = None) -> str:
        """
        Create a directory for a specific job.
        
        Args:
            job_id: The job ID
            base_dir: Base directory (defaults to config setting)
            
        Returns:
            Path to the created directory
        """
        base_dir = base_dir or Config.get("DEFAULT_OUTPUT_DIR", "output")
        job_dir = os.path.join(base_dir, job_id)
        
        os.makedirs(job_dir, exist_ok=True)
        logger.info(f"Created job directory: {job_dir}")
        
        return job_dir
    
    @staticmethod
    def cleanup_files(file_paths: List[str]) -> None:
        """
        Clean up files after processing.
        
        Args:
            file_paths: List of file paths to clean up
        """
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.info(f"Removed temporary file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove file {path}: {e}")
