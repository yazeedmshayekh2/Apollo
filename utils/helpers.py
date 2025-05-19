"""
Helper functions for the Vehicle Registration Card OCR System.
"""

import os
import json
import logging
import uuid
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import hashlib
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)

def generate_unique_id(prefix: str = "ocr") -> str:
    """
    Generate a unique ID for an OCR request or result.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        A unique ID string
    """
    unique_id = f"{prefix}_{uuid.uuid4().hex}"
    return unique_id

def enhance_image(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Enhance an image for better OCR processing.
    Applies contrast enhancement and sharpening.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the enhanced image (if None, modifies the original path)
        
    Returns:
        Path to the enhanced image
    """
    if output_path is None:
        # Create a new path by adding "_enhanced" before the extension
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_enhanced{ext}"
    
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)  # Increase contrast by 50%
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)  # Increase sharpness by 50%
        
        # Save the enhanced image
        img.save(output_path)
        
        logger.info(f"Enhanced image saved to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error enhancing image: {e}")
        return image_path  # Return original path on error

def save_json_result(result: Dict[str, Any], output_dir: str, file_name: Optional[str] = None) -> str:
    """
    Save OCR result as a JSON file.
    
    Args:
        result: The OCR result dictionary
        output_dir: Directory to save the file
        file_name: Optional file name (default: generated from timestamp)
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file name if not provided
    if file_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"ocr_result_{timestamp}.json"
    
    # Ensure .json extension
    if not file_name.endswith('.json'):
        file_name += '.json'
    
    # Build full path
    file_path = os.path.join(output_dir, file_name)
    
    # Write the result to file
    try:
        with open(file_path, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"Result saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving result: {e}")
        return ""

def calculate_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hex digest of the hash
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read the file in chunks
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash: {e}")
        return ""

def validate_field(field_name: str, value: Any, expected_type: type) -> Tuple[bool, str]:
    """
    Validate a field in the OCR result.
    
    Args:
        field_name: Name of the field
        value: Value to validate
        expected_type: Expected type of the value
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(value, expected_type):
        return False, f"Field '{field_name}' has incorrect type. Expected {expected_type.__name__}, got {type(value).__name__}"
    return True, ""

def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary with file information
    """
    try:
        stats = os.stat(file_path)
        return {
            "path": file_path,
            "size": stats.st_size,
            "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
            "hash": calculate_hash(file_path)
        }
    except Exception as e:
        logger.error(f"Error getting file info: {e}")
        return {"path": file_path, "error": str(e)}
