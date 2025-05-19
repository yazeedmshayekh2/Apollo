"""
Database module for the OCR application.
Provides interfaces for MongoDB storage of user data and embeddings.
"""

from .database import MongoDB
from .images import ImageStore
from .audit import AuditLog
from .ocr_store import OCRStore

# Create singleton instances
db = MongoDB()
image_store = ImageStore()
audit_log = AuditLog()
ocr_store = OCRStore()

# Export components
__all__ = ['db', 'image_store', 'audit_log', 'ocr_store', 'MongoDB', 'ImageStore', 'AuditLog', 'OCRStore']
