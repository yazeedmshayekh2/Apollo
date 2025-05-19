"""
MongoDB database connector for the OCR application.
Handles connections and operations for storing user data with embeddings.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, PyMongoError
import numpy as np
from bson.binary import Binary
import pickle

from utils.config import Config

logger = logging.getLogger(__name__)

class MongoDB:
    """MongoDB database connector for the OCR application."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one database connection."""
        if cls._instance is None:
            cls._instance = super(MongoDB, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the MongoDB connection if not already initialized."""
        if self._initialized:
            return
            
        self.mongo_uri = Config.get("MONGODB_URI", "mongodb://localhost:27017")
        self.db_name = Config.get("MONGODB_DB", "ocr_db")
        self.user_collection_name = Config.get("MONGODB_USER_COLLECTION", "users")
        self.embeddings_collection_name = Config.get("MONGODB_EMBEDDINGS_COLLECTION", "embeddings")
        
        self.client = None
        self.db = None
        self.user_collection = None
        self.embeddings_collection = None
        
        self.connect()
        self._initialized = True
    
    def connect(self) -> bool:
        """Connect to MongoDB and initialize collections."""
        try:
            logger.info(f"Connecting to MongoDB at {self.mongo_uri}")
            self.client = MongoClient(self.mongo_uri)
            
            # Test connection
            self.client.admin.command('ping')
            logger.info("MongoDB connection successful")
            
            # Get database and collections
            self.db = self.client[self.db_name]
            self.user_collection = self.db[self.user_collection_name]
            self.embeddings_collection = self.db[self.embeddings_collection_name]
            
            # Create indexes
            self.user_collection.create_index("user_id", unique=True)
            self.embeddings_collection.create_index("user_id")
            
            return True
        
        except ConnectionFailure as e:
            logger.error(f"MongoDB connection failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            return False
    
    def save_user_data(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """
        Save user data to MongoDB.
        
        Args:
            user_id: Unique identifier for the user
            user_data: Dictionary containing user information
            
        Returns:
            bool: Success status
        """
        try:
            # Add user_id to data if not present
            if "user_id" not in user_data:
                user_data["user_id"] = user_id
                
            # Insert or update user data
            result = self.user_collection.update_one(
                {"user_id": user_id},
                {"$set": user_data},
                upsert=True
            )
            
            logger.info(f"User data saved for user_id={user_id}")
            return True
            
        except PyMongoError as e:
            logger.error(f"Error saving user data: {e}")
            return False
    
    def save_embedding(self, user_id: str, embedding: np.ndarray, metadata: Dict[str, Any] = None) -> bool:
        """
        Save embedding for a user.
        
        Args:
            user_id: User ID associated with the embedding
            embedding: Numpy array containing the embedding vector
            metadata: Additional metadata about the embedding
            
        Returns:
            bool: Success status
        """
        try:
            if metadata is None:
                metadata = {}
                
            # Serialize the numpy array
            serialized_embedding = Binary(pickle.dumps(embedding, protocol=2))
            
            # Create document
            embedding_doc = {
                "user_id": user_id,
                "embedding": serialized_embedding,
                "embedding_dim": embedding.shape[0],
                "metadata": metadata
            }
            
            # Save to collection
            result = self.embeddings_collection.insert_one(embedding_doc)
            
            logger.info(f"Embedding saved for user_id={user_id}")
            return True
            
        except PyMongoError as e:
            logger.error(f"Error saving embedding: {e}")
            return False
        except Exception as e:
            logger.error(f"Error serializing embedding: {e}")
            return False
    
    def get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user data by user ID.
        
        Args:
            user_id: User ID to retrieve
            
        Returns:
            Dict or None: User data if found
        """
        try:
            user = self.user_collection.find_one({"user_id": user_id})
            return user
        except PyMongoError as e:
            logger.error(f"Error retrieving user data: {e}")
            return None
    
    def get_user_embeddings(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all embeddings for a user.
        
        Args:
            user_id: User ID to retrieve embeddings for
            
        Returns:
            List: List of embedding documents
        """
        try:
            cursor = self.embeddings_collection.find({"user_id": user_id})
            embeddings = []
            
            for doc in cursor:
                # Deserialize embedding
                if "embedding" in doc:
                    doc["embedding"] = pickle.loads(doc["embedding"])
                embeddings.append(doc)
                
            return embeddings
            
        except PyMongoError as e:
            logger.error(f"Error retrieving user embeddings: {e}")
            return []
    
    def close(self):
        """Close the MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
