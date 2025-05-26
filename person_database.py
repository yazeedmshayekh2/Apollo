import torch
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import uuid
import json
from typing import Dict, List, Optional, Any
import cv2
import os

class PersonDatabase:
    """
    Enhanced MongoDB integration for storing OCR-extracted information 
    along with face embeddings for each person
    """
    
    def __init__(self, mongo_uri: str = 'mongodb://localhost:27017/', 
                 database_name: str = 'person_verification'):
        """
        Initialize the PersonDatabase with MongoDB connection
        
        Args:
            mongo_uri: MongoDB connection string
            database_name: Name of the database to use
        """
        try:
            self.client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            self.client.server_info()  # Test connection
            print("Connected to MongoDB successfully")
            
            self.db = self.client[database_name]
            
            # Collections for different types of data
            self.persons_collection = self.db['persons']  # Main person records
            self.documents_collection = self.db['documents']  # Document records
            self.face_embeddings_collection = self.db['face_embeddings']  # Face embeddings
            
            # Create indexes for better performance
            self._create_indexes()
            
        except Exception as e:
            print(f"Warning: Could not connect to MongoDB: {e}")
            print("Will continue without database functionality")
            self.client = None
            self.db = None
            self.persons_collection = None
            self.documents_collection = None
            self.face_embeddings_collection = None
    
    def _create_indexes(self):
        """Create database indexes for better query performance"""
        try:
            # Index on person ID for fast lookups
            self.persons_collection.create_index("person_id", unique=True)
            self.documents_collection.create_index("person_id")
            self.face_embeddings_collection.create_index("person_id")
            
            # Index on document number for fast document lookups
            self.persons_collection.create_index("document_info.id_number")
            self.documents_collection.create_index("document_info.id_number")
            
            # Index on name for search functionality
            self.persons_collection.create_index("personal_info.name")
            
            print("Database indexes created successfully")
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")
    
    def save_person_with_embeddings(self, 
                                  person_id: str,
                                  ocr_data: Dict[str, Any],
                                  face_embeddings: np.ndarray,
                                  face_image_path: Optional[str] = None,
                                  document_images: Optional[Dict[str, str]] = None) -> bool:
        """
        Save a complete person record with OCR data and face embeddings
        
        Args:
            person_id: Unique identifier for the person
            ocr_data: OCR extracted information from documents
            face_embeddings: Face feature vector/embeddings
            face_image_path: Path to the extracted face image
            document_images: Dictionary with paths to document images (front, back)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._is_connected():
            return self._save_locally(person_id, ocr_data, face_embeddings, face_image_path)
        
        try:
            timestamp = datetime.utcnow()
            
            # Prepare person record
            person_record = {
                "person_id": person_id,
                "created_at": timestamp,
                "updated_at": timestamp,
                "status": "active"
            }
            
            # Add OCR data to person record
            if ocr_data:
                # Extract key information for the main person record
                person_record.update({
                    "document_info": ocr_data.get("document_info", {}),
                    "personal_info": ocr_data.get("personal_info", {}),
                    "additional_info": ocr_data.get("additional_info", {})
                })
            
            # Add face image path if provided
            if face_image_path:
                person_record["face_image_path"] = face_image_path
            
            # Add document image paths if provided
            if document_images:
                person_record["document_images"] = document_images
            
            # Save/update person record
            self.persons_collection.update_one(
                {"person_id": person_id},
                {"$set": person_record},
                upsert=True
            )
            
            # Save detailed document record
            if ocr_data:
                document_record = {
                    "person_id": person_id,
                    "document_id": str(uuid.uuid4()),
                    "created_at": timestamp,
                    "ocr_data": ocr_data,
                    "document_images": document_images or {}
                }
                
                self.documents_collection.update_one(
                    {"person_id": person_id},
                    {"$set": document_record},
                    upsert=True
                )
            
            # Save face embeddings
            if face_embeddings is not None:
                embedding_record = {
                    "person_id": person_id,
                    "embedding_id": str(uuid.uuid4()),
                    "created_at": timestamp,
                    "face_embeddings": face_embeddings.tolist(),
                    "embedding_method": "ResNet50" if len(face_embeddings) == 512 else "HOG",
                    "face_image_path": face_image_path
                }
                
                self.face_embeddings_collection.update_one(
                    {"person_id": person_id},
                    {"$set": embedding_record},
                    upsert=True
                )
            
            print(f"Successfully saved person record for {person_id}")
            return True
            
        except Exception as e:
            print(f"Error saving person record: {e}")
            return self._save_locally(person_id, ocr_data, face_embeddings, face_image_path)
    
    def get_person_by_id(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete person information by person ID
        
        Args:
            person_id: Unique identifier for the person
            
        Returns:
            Dictionary containing person info, OCR data, and embeddings, or None if not found
        """
        if not self._is_connected():
            return self._load_locally(person_id)
        
        try:
            # Get main person record
            person = self.persons_collection.find_one({"person_id": person_id})
            if not person:
                return None
            
            # Get detailed document record
            document = self.documents_collection.find_one({"person_id": person_id})
            
            # Get face embeddings
            embeddings = self.face_embeddings_collection.find_one({"person_id": person_id})
            
            # Combine all information
            result = {
                "person_info": person,
                "document_data": document.get("ocr_data", {}) if document else {},
                "face_embeddings": np.array(embeddings["face_embeddings"]) if embeddings else None,
                "embedding_method": embeddings.get("embedding_method") if embeddings else None
            }
            
            return result
            
        except Exception as e:
            print(f"Error retrieving person record: {e}")
            return self._load_locally(person_id)
    
    def get_person_by_document_number(self, document_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve person information by document number (ID number)
        
        Args:
            document_number: Document/ID number to search for
            
        Returns:
            Dictionary containing person info or None if not found
        """
        if not self._is_connected():
            return None
        
        try:
            # Search in persons collection first
            person = self.persons_collection.find_one({
                "$or": [
                    {"personal_info.id_number": document_number},
                    {"document_info.id_number": document_number}
                ]
            })
            
            if person:
                return self.get_person_by_id(person["person_id"])
            
            return None
            
        except Exception as e:
            print(f"Error searching by document number: {e}")
            return None
    
    def search_persons_by_name(self, name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for persons by name (supports partial matching)
        
        Args:
            name: Name to search for
            limit: Maximum number of results to return
            
        Returns:
            List of matching person records
        """
        if not self._is_connected():
            return []
        
        try:
            # Create case-insensitive regex pattern
            pattern = {"$regex": name, "$options": "i"}
            
            # Search in both English and Arabic names
            query = {
                "$or": [
                    {"personal_info.name": pattern},
                    {"personal_info.arabic_name": pattern}
                ]
            }
            
            results = []
            for person in self.persons_collection.find(query).limit(limit):
                person_data = self.get_person_by_id(person["person_id"])
                if person_data:
                    results.append(person_data)
            
            return results
            
        except Exception as e:
            print(f"Error searching by name: {e}")
            return []
    
    def get_face_embeddings(self, person_id: str) -> Optional[np.ndarray]:
        """
        Get face embeddings for a specific person
        
        Args:
            person_id: Unique identifier for the person
            
        Returns:
            Face embeddings as numpy array or None if not found
        """
        if not self._is_connected():
            return self._load_embeddings_locally(person_id)
        
        try:
            embeddings_doc = self.face_embeddings_collection.find_one({"person_id": person_id})
            if embeddings_doc and "face_embeddings" in embeddings_doc:
                return np.array(embeddings_doc["face_embeddings"])
            return None
            
        except Exception as e:
            print(f"Error retrieving embeddings: {e}")
            return self._load_embeddings_locally(person_id)
    
    def update_person_info(self, person_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update specific fields in a person's record
        
        Args:
            person_id: Unique identifier for the person
            updates: Dictionary of fields to update
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._is_connected():
            return False
        
        try:
            updates["updated_at"] = datetime.utcnow()
            
            result = self.persons_collection.update_one(
                {"person_id": person_id},
                {"$set": updates}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            print(f"Error updating person record: {e}")
            return False
    
    def delete_person(self, person_id: str) -> bool:
        """
        Delete all records for a person (soft delete by marking as inactive)
        
        Args:
            person_id: Unique identifier for the person
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self._is_connected():
            return False
        
        try:
            # Soft delete by marking as inactive
            self.persons_collection.update_one(
                {"person_id": person_id},
                {"$set": {"status": "inactive", "deleted_at": datetime.utcnow()}}
            )
            
            return True
            
        except Exception as e:
            print(f"Error deleting person record: {e}")
            return False
    
    def get_all_persons(self, include_inactive: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get all person records
        
        Args:
            include_inactive: Whether to include inactive/deleted records
            limit: Maximum number of records to return
            
        Returns:
            List of person records
        """
        if not self._is_connected():
            return []
        
        try:
            query = {} if include_inactive else {"status": {"$ne": "inactive"}}
            
            results = []
            for person in self.persons_collection.find(query).limit(limit):
                results.append(person)
            
            return results
            
        except Exception as e:
            print(f"Error retrieving all persons: {e}")
            return []
    
    def _is_connected(self) -> bool:
        """Check if MongoDB connection is available"""
        return self.client is not None and self.db is not None
    
    def _save_locally(self, person_id: str, ocr_data: Dict[str, Any], 
                     face_embeddings: np.ndarray, face_image_path: Optional[str]) -> bool:
        """Fallback method to save data locally when MongoDB is not available"""
        try:
            # Create local storage directory
            os.makedirs("local_storage", exist_ok=True)
            
            # Save OCR data
            if ocr_data:
                with open(f"local_storage/{person_id}_ocr.json", "w") as f:
                    json.dump(ocr_data, f, indent=2, default=str)
            
            # Save embeddings
            if face_embeddings is not None:
                np.save(f"local_storage/{person_id}_embeddings.npy", face_embeddings)
            
            # Save metadata
            metadata = {
                "person_id": person_id,
                "created_at": datetime.utcnow().isoformat(),
                "face_image_path": face_image_path
            }
            with open(f"local_storage/{person_id}_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"Saved person data locally for {person_id}")
            return True
            
        except Exception as e:
            print(f"Error saving locally: {e}")
            return False
    
    def _load_locally(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Load person data from local storage"""
        try:
            result = {}
            
            # Load OCR data
            ocr_file = f"local_storage/{person_id}_ocr.json"
            if os.path.exists(ocr_file):
                with open(ocr_file, "r") as f:
                    result["document_data"] = json.load(f)
            
            # Load embeddings
            embeddings_file = f"local_storage/{person_id}_embeddings.npy"
            if os.path.exists(embeddings_file):
                result["face_embeddings"] = np.load(embeddings_file)
            
            # Load metadata
            metadata_file = f"local_storage/{person_id}_metadata.json"
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    result["person_info"] = json.load(f)
            
            return result if result else None
            
        except Exception as e:
            print(f"Error loading locally: {e}")
            return None
    
    def _load_embeddings_locally(self, person_id: str) -> Optional[np.ndarray]:
        """Load embeddings from local storage"""
        try:
            embeddings_file = f"local_storage/{person_id}_embeddings.npy"
            if os.path.exists(embeddings_file):
                return np.load(embeddings_file)
            return None
        except Exception as e:
            print(f"Error loading embeddings locally: {e}")
            return None
    
    def close_connection(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            print("MongoDB connection closed") 