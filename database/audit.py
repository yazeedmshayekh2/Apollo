"""
AuditLog class for tracking operations in the OCR application.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional
from pymongo.errors import PyMongoError

from .database import MongoDB
from utils.config import Config

logger = logging.getLogger(__name__)

class AuditLog:
    """Handles audit logging of operations in MongoDB."""
    
    def __init__(self):
        """Initialize the AuditLog with MongoDB connection."""
        self.db = MongoDB()
        self.collection_name = Config.get("MONGODB_AUDIT_COLLECTION", "audit_logs")
        self.collection = self.db.db[self.collection_name]
        
        # Create indexes
        self.collection.create_index("user_id")
        self.collection.create_index("timestamp")
        self.collection.create_index("operation")
        
    def log_operation(
        self, 
        operation: str, 
        user_id: str, 
        details: Dict[str, Any] = None, 
        status: str = "success"
    ) -> bool:
        """
        Log an operation to the audit log.
        
        Args:
            operation: Type of operation performed (e.g., "save_embedding", "ocr_extract")
            user_id: User who performed the operation
            details: Additional details about the operation
            status: Operation status (success/failure)
            
        Returns:
            bool: Success status
        """
        try:
            if details is None:
                details = {}
                
            # Create log entry
            log_entry = {
                "operation": operation,
                "user_id": user_id,
                "details": details,
                "status": status,
                "timestamp": datetime.datetime.utcnow()
            }
            
            # Insert log entry
            result = self.collection.insert_one(log_entry)
            
            return True
            
        except PyMongoError as e:
            logger.error(f"Error logging operation: {e}")
            return False
    
    def get_user_logs(
        self, 
        user_id: str, 
        operation: str = None, 
        start_time: datetime.datetime = None,
        end_time: datetime.datetime = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit logs for a user.
        
        Args:
            user_id: User ID to retrieve logs for
            operation: Filter by operation type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of logs to return
            
        Returns:
            List: List of audit log entries
        """
        try:
            # Build query
            query = {"user_id": user_id}
            
            if operation:
                query["operation"] = operation
                
            if start_time or end_time:
                time_query = {}
                if start_time:
                    time_query["$gte"] = start_time
                if end_time:
                    time_query["$lte"] = end_time
                if time_query:
                    query["timestamp"] = time_query
            
            # Execute query
            cursor = self.collection.find(query).sort("timestamp", -1).limit(limit)
            return list(cursor)
            
        except PyMongoError as e:
            logger.error(f"Error retrieving audit logs: {e}")
            return []
