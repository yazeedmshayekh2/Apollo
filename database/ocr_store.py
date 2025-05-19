"""
OCRStore for handling OCR extraction results and embeddings in MongoDB.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional, Union
import numpy as np
from pymongo.errors import PyMongoError
from bson.binary import Binary
import pickle
import json

from .database import MongoDB
from utils.config import Config

logger = logging.getLogger(__name__)

class OCRStore:
    """Handles storage and retrieval of OCR data with embeddings in MongoDB."""
    
    def __init__(self):
        """Initialize the OCRStore with MongoDB connection."""
        self.db = MongoDB()
        self.collection_name = Config.get("MONGODB_OCR_COLLECTION", "ocr_data")
        self.collection = self.db.db[self.collection_name]
        
        # Create indexes
        self.collection.create_index("user_id")
        self.collection.create_index("document_id", unique=True)
        self.collection.create_index("job_id")
        self.collection.create_index("document_type")
        self.collection.create_index([("extracted_text", "text")])  # Text index for search
        
    def save_ocr_data(
        self,
        user_id: str,
        document_id: str,
        extracted_data: Dict[str, Any],
        embedding: Optional[np.ndarray] = None,
        document_type: Optional[str] = None,
        job_id: Optional[str] = None,
        raw_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save OCR extracted data with optional embedding to MongoDB.
        
        Args:
            user_id: ID of the user who owns this document
            document_id: Unique ID for the document
            extracted_data: Structured data extracted from OCR
            embedding: Optional embedding vector as numpy array
            document_type: Type of document (e.g., "id_card", "vehicle_registration")
            job_id: Associated processing job ID
            raw_text: Raw extracted text
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            if metadata is None:
                metadata = {}
                
            # Create document
            ocr_doc = {
                "user_id": user_id,
                "document_id": document_id,
                "extracted_data": extracted_data,
                "document_type": document_type,
                "job_id": job_id,
                "raw_text": raw_text,
                "metadata": metadata,
                "created_at": datetime.datetime.utcnow(),
                "updated_at": datetime.datetime.utcnow()
            }
            
            # Add embedding if provided
            if embedding is not None:
                # Serialize the numpy array
                serialized_embedding = Binary(pickle.dumps(embedding, protocol=2))
                ocr_doc["embedding"] = serialized_embedding
                ocr_doc["embedding_dim"] = embedding.shape[0]
            
            # Extract text for text search
            extracted_text = self._extract_searchable_text(extracted_data, raw_text)
            if extracted_text:
                ocr_doc["extracted_text"] = extracted_text
            
            # Insert or update document
            result = self.collection.update_one(
                {"document_id": document_id},
                {"$set": ocr_doc},
                upsert=True
            )
            
            logger.info(f"OCR data saved for user_id={user_id}, document_id={document_id}")
            return True
            
        except PyMongoError as e:
            logger.error(f"Error saving OCR data: {e}")
            return False
        except Exception as e:
            logger.error(f"Error processing OCR data: {e}")
            return False
    
    def get_user_documents(
        self, 
        user_id: str, 
        document_type: Optional[str] = None,
        include_embeddings: bool = False,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all OCR documents for a user.
        
        Args:
            user_id: User ID to retrieve documents for
            document_type: Optional filter by document type
            include_embeddings: Whether to include embeddings in the result
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            List: List of OCR documents
        """
        try:
            # Build query
            query = {"user_id": user_id}
            if document_type:
                query["document_type"] = document_type
                
            # Define projection
            projection = None
            if not include_embeddings:
                projection = {"embedding": 0}
                
            # Execute query
            cursor = self.collection.find(query, projection).skip(offset).limit(limit)
            
            # Process results
            documents = []
            for doc in cursor:
                # Deserialize embedding if present and requested
                if include_embeddings and "embedding" in doc:
                    doc["embedding"] = pickle.loads(doc["embedding"])
                documents.append(doc)
                
            return documents
            
        except PyMongoError as e:
            logger.error(f"Error retrieving user documents: {e}")
            return []
    
    def get_document(
        self, 
        document_id: str,
        include_embedding: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific OCR document.
        
        Args:
            document_id: Document ID to retrieve
            include_embedding: Whether to include the embedding
            
        Returns:
            Dict or None: Document if found
        """
        try:
            # Define projection
            projection = None
            if not include_embedding:
                projection = {"embedding": 0}
                
            # Execute query
            doc = self.collection.find_one({"document_id": document_id}, projection)
            
            # Deserialize embedding if present and requested
            if doc and include_embedding and "embedding" in doc:
                doc["embedding"] = pickle.loads(doc["embedding"])
                
            return doc
            
        except PyMongoError as e:
            logger.error(f"Error retrieving document: {e}")
            return None
    
    def search_user_documents(
        self,
        user_id: str,
        search_text: str,
        document_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search user documents by text content.
        
        Args:
            user_id: User ID to search documents for
            search_text: Text to search for
            document_type: Optional filter by document type
            limit: Maximum number of results
            
        Returns:
            List: Matching documents
        """
        try:
            # Build query
            query = {
                "user_id": user_id,
                "$text": {"$search": search_text}
            }
            
            if document_type:
                query["document_type"] = document_type
                
            # Execute query with text score
            cursor = self.collection.find(
                query,
                {"embedding": 0, "score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(limit)
            
            return list(cursor)
            
        except PyMongoError as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _extract_searchable_text(self, extracted_data: Dict[str, Any], raw_text: Optional[str]) -> str:
        """
        Extract searchable text from structured data and raw text.
        
        Args:
            extracted_data: Structured data extracted from OCR
            raw_text: Raw extracted text
            
        Returns:
            str: Searchable text
        """
        text_parts = []
        
        # Add raw text if available
        if raw_text:
            text_parts.append(raw_text)
        
        # Flatten extracted data for searching
        try:
            def extract_values(data, prefix=""):
                if isinstance(data, dict):
                    for key, value in data.items():
                        yield from extract_values(value, f"{prefix}{key}_")
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        yield from extract_values(item, f"{prefix}{i}_")
                else:
                    if data is not None and str(data).strip():
                        yield str(data)
            
            text_parts.extend(extract_values(extracted_data))
            
        except Exception as e:
            logger.warning(f"Error extracting searchable text: {e}")
        
        return " ".join(text_parts) 