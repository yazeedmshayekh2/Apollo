"""
ImageStore class for handling image storage and retrieval with MongoDB.
"""

import os
import logging
from typing import Dict, Any, List, Optional
import datetime
from pymongo.errors import PyMongoError

from .database import MongoDB
from utils.config import Config

logger = logging.getLogger(__name__)

class ImageStore:
    """Handles storage and retrieval of image metadata with MongoDB."""
    
    def __init__(self):
        """Initialize the ImageStore with MongoDB connection."""
        self.db = MongoDB()
        self.collection_name = Config.get("MONGODB_IMAGES_COLLECTION", "images")
        self.collection = self.db.db[self.collection_name]
        
        # Create indexes
        self.collection.create_index("image_id", unique=True)
        self.collection.create_index("user_id")
        self.collection.create_index("job_id")
        
    def save_image_metadata(
        self, 
        image_id: str, 
        user_id: str, 
        job_id: str, 
        file_path: str, 
        file_name: str,
        document_type: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Save image metadata to MongoDB.
        
        Args:
            image_id: Unique identifier for the image
            user_id: User who uploaded the image
            job_id: Processing job ID
            file_path: Path to the stored image
            file_name: Original filename
            document_type: Type of document detected
            metadata: Additional metadata about the image
            
        Returns:
            bool: Success status
        """
        try:
            if metadata is None:
                metadata = {}
                
            # Create document
            image_doc = {
                "image_id": image_id,
                "user_id": user_id,
                "job_id": job_id,
                "file_path": file_path,
                "file_name": file_name,
                "document_type": document_type,
                "metadata": metadata,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
            
            # Insert or update image metadata
            result = self.collection.update_one(
                {"image_id": image_id},
                {"$set": image_doc},
                upsert=True
            )
            
            logger.info(f"Image metadata saved for image_id={image_id}")
            return True
            
        except PyMongoError as e:
            logger.error(f"Error saving image metadata: {e}")
            return False
    
    def get_image_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve image metadata by image ID.
        
        Args:
            image_id: Image ID to retrieve
            
        Returns:
            Dict or None: Image metadata if found
        """
        try:
            image = self.collection.find_one({"image_id": image_id})
            return image
        except PyMongoError as e:
            logger.error(f"Error retrieving image metadata: {e}")
            return None
    
    def get_user_images(self, user_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve all images for a user.
        
        Args:
            user_id: User ID to retrieve images for
            limit: Maximum number of images to return
            offset: Offset for pagination
            
        Returns:
            List: List of image documents
        """
        try:
            cursor = self.collection.find({"user_id": user_id}).skip(offset).limit(limit)
            return list(cursor)
        except PyMongoError as e:
            logger.error(f"Error retrieving user images: {e}")
            return []
    
    def get_job_images(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all images for a job.
        
        Args:
            job_id: Job ID to retrieve images for
            
        Returns:
            List: List of image documents
        """
        try:
            cursor = self.collection.find({"job_id": job_id})
            return list(cursor)
        except PyMongoError as e:
            logger.error(f"Error retrieving job images: {e}")
            return []
