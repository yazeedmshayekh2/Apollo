import torch
import numpy as np
from pymongo import MongoClient
from datetime import datetime
import uuid
import json
from typing import Dict, List, Optional, Any
import cv2
import os
import sqlite3
import bcrypt
from pathlib import Path

class PersonDatabase:
    """
    Enhanced MongoDB integration for storing OCR-extracted information 
    along with face embeddings for each person
    """
    
    def __init__(self, db_path="people.db"):
        """Initialize database connections for both SQLite and MongoDB"""
        self.db_path = db_path
        self._init_sqlite()
        self._init_mongodb()

    def _init_sqlite(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    phone_number TEXT,
                    document_type TEXT,
                    document_info TEXT,
                    face_embedding TEXT,
                    profile_image_path TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _init_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            self.mongo_client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
            self.mongo_client.server_info()
            print("Connected to MongoDB successfully")
            self.mongo_db = self.mongo_client['face_verification']
            self.face_features_collection = self.mongo_db['face_features']
            
            # Clean up any documents with null person_id
            self.face_features_collection.delete_many({"person_id": None})
            
            # Drop existing index if it exists
            try:
                self.face_features_collection.drop_index("person_id_1")
            except Exception:
                pass  # Index might not exist
            
            # Create index with partial filter to exclude null values
            self.face_features_collection.create_index(
                "person_id",
                unique=True,
                partialFilterExpression={"person_id": {"$type": "string"}}
            )
            print("MongoDB indexes created successfully")
        except Exception as e:
            print(f"Warning: Could not connect to MongoDB: {e}")
            self.mongo_client = None
            self.mongo_db = None
            self.face_features_collection = None

    def add_person(self, person_id, username, password, phone_number, document_type, document_info, face_embedding, profile_image_path):
        """Add a new person to both SQLite and MongoDB"""
        # Validate person_id
        if not person_id:
            raise ValueError("person_id cannot be null or empty")

        # Hash the password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Convert document info to string if it's a dict
        if isinstance(document_info, dict):
            document_info_str = json.dumps(document_info)
        else:
            document_info_str = document_info

        # Save to SQLite
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("""
                    INSERT INTO people (id, username, password_hash, phone_number, document_type, 
                                     document_info, profile_image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (person_id, username, password_hash, phone_number, document_type, 
                     document_info_str, profile_image_path))
                conn.commit()
            except sqlite3.IntegrityError:
                return False

        # Save face embedding to MongoDB
        if self.face_features_collection and face_embedding is not None:
            try:
                # Convert numpy array to list for MongoDB storage
                face_embedding_list = face_embedding.tolist() if isinstance(face_embedding, np.ndarray) else face_embedding
                
                # Store in MongoDB
                self.face_features_collection.update_one(
                    {"person_id": person_id},
                    {
                        "$set": {
                            "face_embedding": face_embedding_list,
                            "username": username,
                            "updated_at": datetime.utcnow()
                        }
                    },
                    upsert=True
                )
            except Exception as e:
                print(f"Warning: Could not save face embedding to MongoDB: {e}")
                # If MongoDB fails, store embedding in SQLite as fallback
                self._update_sqlite_embedding(person_id, face_embedding)
                
        return True

    def _update_sqlite_embedding(self, person_id, face_embedding):
        """Update face embedding in SQLite as fallback"""
        if face_embedding is not None:
            face_embedding_str = json.dumps(face_embedding.tolist()) if isinstance(face_embedding, np.ndarray) else json.dumps(face_embedding)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE people SET face_embedding = ? WHERE id = ?", 
                           (face_embedding_str, person_id))
                conn.commit()

    def get_all_face_embeddings(self):
        """Get all face embeddings, preferring MongoDB but falling back to SQLite"""
        embeddings = {}
        
        # Try MongoDB first
        if self.face_features_collection:
            try:
                cursor = self.face_features_collection.find({}, {"person_id": 1, "face_embedding": 1})
                for doc in cursor:
                    if "face_embedding" in doc:
                        embeddings[doc["person_id"]] = np.array(doc["face_embedding"])
                if embeddings:  # If we got embeddings from MongoDB, return them
                    return embeddings
            except Exception as e:
                print(f"Warning: Could not retrieve face embeddings from MongoDB: {e}")

        # Fallback to SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, face_embedding 
                FROM people 
                WHERE face_embedding IS NOT NULL AND is_active = 1
            """)
            results = cursor.fetchall()
            
            for person_id, embedding_str in results:
                if embedding_str:
                    embeddings[person_id] = np.array(json.loads(embedding_str))
        
        return embeddings

    def verify_password(self, username, password):
        """Verify password from SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT password_hash FROM people WHERE username = ? AND is_active = 1", (username,))
            result = cursor.fetchone()
            if result:
                stored_hash = result[0]
                return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
            return False

    def get_person_by_username(self, username):
        """Get person info from both SQLite and MongoDB"""
        # Get basic info from SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, username, phone_number, document_type, document_info, 
                       profile_image_path
                FROM people 
                WHERE username = ? AND is_active = 1
            """, (username,))
            result = cursor.fetchone()
            
            if not result:
                return None

            person_data = {
                'id': result[0],
                'username': result[1],
                'phone_number': result[2],
                'document_type': result[3],
                'document_info': json.loads(result[4]) if result[4] else None,
                'profile_image_path': result[5]
            }

            # Try to get face embedding from MongoDB
            if self.face_features_collection:
                try:
                    mongo_doc = self.face_features_collection.find_one({"person_id": person_data['id']})
                    if mongo_doc and "face_embedding" in mongo_doc:
                        person_data['face_embedding'] = np.array(mongo_doc["face_embedding"])
                except Exception as e:
                    print(f"Warning: Could not retrieve face embedding from MongoDB: {e}")

            return person_data

    def get_person(self, person_id):
        """Get person info from both SQLite and MongoDB"""
        # Get basic info from SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, username, phone_number, document_type, document_info, 
                       profile_image_path
                FROM people 
                WHERE id = ? AND is_active = 1
            """, (person_id,))
            result = cursor.fetchone()
            
            if not result:
                return None

            person_data = {
                'id': result[0],
                'username': result[1],
                'phone_number': result[2],
                'document_type': result[3],
                'document_info': json.loads(result[4]) if result[4] else None,
                'profile_image_path': result[5]
            }

            # Try to get face embedding from MongoDB
            if self.face_features_collection:
                try:
                    mongo_doc = self.face_features_collection.find_one({"person_id": person_id})
                    if mongo_doc and "face_embedding" in mongo_doc:
                        person_data['face_embedding'] = np.array(mongo_doc["face_embedding"])
                except Exception as e:
                    print(f"Warning: Could not retrieve face embedding from MongoDB: {e}")

            return person_data

    def username_exists(self, username):
        """Check if username exists in SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM people WHERE username = ? AND is_active = 1", (username,))
            return cursor.fetchone() is not None

    def close(self):
        """Close database connections"""
        if hasattr(self, 'mongo_client') and self.mongo_client:
            self.mongo_client.close()
            print("MongoDB connection closed")
    
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
        return self.mongo_client is not None and self.mongo_db is not None
    
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
        if self.mongo_client:
            self.mongo_client.close()
            print("MongoDB connection closed") 

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    phone_number TEXT,
                    document_type TEXT,
                    document_info TEXT,
                    face_embedding TEXT,
                    profile_image_path TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def add_person(self, person_id, username, password, phone_number, document_type, document_info, face_embedding, profile_image_path):
        # Hash the password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Convert face embedding to string for storage
        face_embedding_str = json.dumps(face_embedding.tolist()) if face_embedding is not None else None
        
        # Convert document info to string if it's a dict
        if isinstance(document_info, dict):
            document_info = json.dumps(document_info)

        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute("""
                    INSERT INTO people (id, username, password_hash, phone_number, document_type, 
                                     document_info, face_embedding, profile_image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (person_id, username, password_hash, phone_number, document_type, 
                     document_info, face_embedding_str, profile_image_path))
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def verify_password(self, username, password):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT password_hash FROM people WHERE username = ? AND is_active = 1", (username,))
            result = cursor.fetchone()
            if result:
                stored_hash = result[0]
                return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
            return False

    def get_person_by_username(self, username):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, username, phone_number, document_type, document_info, 
                       face_embedding, profile_image_path
                FROM people 
                WHERE username = ? AND is_active = 1
            """, (username,))
            result = cursor.fetchone()
            
            if result:
                person_data = {
                    'id': result[0],
                    'username': result[1],
                    'phone_number': result[2],
                    'document_type': result[3],
                    'document_info': json.loads(result[4]) if result[4] else None,
                    'face_embedding': np.array(json.loads(result[5])) if result[5] else None,
                    'profile_image_path': result[6]
                }
                return person_data
            return None

    def get_person(self, person_id):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, username, phone_number, document_type, document_info, 
                       face_embedding, profile_image_path
                FROM people 
                WHERE id = ? AND is_active = 1
            """, (person_id,))
            result = cursor.fetchone()
            
            if result:
                person_data = {
                    'id': result[0],
                    'username': result[1],
                    'phone_number': result[2],
                    'document_type': result[3],
                    'document_info': json.loads(result[4]) if result[4] else None,
                    'face_embedding': np.array(json.loads(result[5])) if result[5] else None,
                    'profile_image_path': result[6]
                }
                return person_data
            return None

    def get_all_face_embeddings(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT id, face_embedding 
                FROM people 
                WHERE face_embedding IS NOT NULL AND is_active = 1
            """)
            results = cursor.fetchall()
            
            embeddings = {}
            for person_id, embedding_str in results:
                if embedding_str:
                    embeddings[person_id] = np.array(json.loads(embedding_str))
            return embeddings

    def update_person(self, person_id, **kwargs):
        valid_fields = {'username', 'phone_number', 'document_type', 'document_info', 
                       'face_embedding', 'profile_image_path', 'is_active'}
        
        update_fields = []
        values = []
        
        for key, value in kwargs.items():
            if key in valid_fields:
                if key == 'face_embedding' and value is not None:
                    value = json.dumps(value.tolist())
                elif key == 'document_info' and isinstance(value, dict):
                    value = json.dumps(value)
                
                update_fields.append(f"{key} = ?")
                values.append(value)
        
        if not update_fields:
            return False
        
        values.append(person_id)
        query = f"""
            UPDATE people 
            SET {', '.join(update_fields)}
            WHERE id = ?
        """
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(query, values)
            conn.commit()
            return True

    def delete_person(self, person_id):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE people SET is_active = 0 WHERE id = ?", (person_id,))
            conn.commit()
            return True

    def username_exists(self, username):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT 1 FROM people WHERE username = ? AND is_active = 1", (username,))
            return cursor.fetchone() is not None 