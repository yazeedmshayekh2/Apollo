"""
API routes for the Document OCR system.
Provides endpoints for uploading and processing various document types.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Query, APIRouter
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.staticfiles import StaticFiles
import json
import re
import torch

from core.qwen_service import QwenService
from core.extractor import VehicleDataExtractor
from core.registry import ModelRegistry, get_registry
from core.document_detector import DocumentTypeDetector
from utils.config import Config
from .upload import UploadHandler
from .validation import ResultValidator
from .webhooks import WebhookNotifier

# Set up logging
logging.basicConfig(level=Config.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Document OCR API",
    description="API for extracting information from document images including vehicle registration cards and ID cards",
    version="1.0.0",
)

# Define API router with prefix
api_router = APIRouter(prefix="/api")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the absolute path to the frontend directory
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
public_dir = os.path.join(frontend_dir, "public")
src_dir = os.path.join(frontend_dir, "src")

# Debug log the directory paths
logger.info(f"Frontend directory: {frontend_dir}")
logger.info(f"Public directory: {public_dir}")
logger.info(f"Source directory: {src_dir}")

# Check if directories exist
if not os.path.exists(frontend_dir):
    logger.error(f"Frontend directory not found: {frontend_dir}")
if not os.path.exists(public_dir):
    logger.error(f"Public directory not found: {public_dir}")
if not os.path.exists(src_dir):
    logger.error(f"Source directory not found: {src_dir}")

# Initialize services
model_registry = get_registry()
# Create QwenService with lazy loading - only initialize when needed
qwen_service = None
document_detector = None

# Initialize extractors, validators and webhooks
data_extractor = VehicleDataExtractor()
result_validator = ResultValidator(min_confidence=Config.get("MIN_CONFIDENCE", 0.7))
webhook_notifier = WebhookNotifier()

# Helper function to get or initialize QwenService
def get_qwen_service():
    global qwen_service
    if qwen_service is None:
        # Clear CUDA cache before initializing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        # Use the unsloth model specified in the registry
        model_config = model_registry.get_active_model_config()
        default_model = Config.get("MODEL_NAME", "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit")
        model_name = model_config.get("model_name", default_model)
        
        logger.info(f"Initializing QwenService with model: {model_name}")
        qwen_service = QwenService(model_id=model_config.get("model_id"))
        
        # Initialize document detector and share the QwenService
        global document_detector
        document_detector = DocumentTypeDetector()
        document_detector.set_qwen_service(qwen_service)
        
    return qwen_service

# Helper function to get or initialize DocumentDetector
def get_document_detector():
    global document_detector
    if document_detector is None:
        # Get QwenService first to ensure it's initialized
        service = get_qwen_service()
        document_detector = DocumentTypeDetector()
        document_detector.set_qwen_service(service)
    return document_detector

# Define request/response models
class ProcessingOptions(BaseModel):
    """Options for processing documents."""
    output_format: str = Field("json", description="Output format (json, yaml, csv)")

class ProcessingResult(BaseModel):
    """Result of processing a document."""
    job_id: str = Field(..., description="Unique job ID")
    status: str = Field(..., description="Processing status")
    document_type: str = Field("", description="Detected document type")
    additional_fields: dict = Field({}, description="Additional extracted fields")
    raw_model_response: str = Field("", description="Raw response from the VLM")
    extracted_json_str: str = Field("", description="Extracted JSON string without parsing")
    extracted_json: dict = Field({}, description="Extracted JSON if parsing succeeded")
    metadata: dict = Field({}, description="Processing metadata")

class JobStatus(BaseModel):
    """Status of a processing job."""
    job_id: str = Field(..., description="Unique job ID")
    status: str = Field(..., description="Processing status")
    created_at: str = Field(..., description="Job creation timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    document_type: Optional[str] = Field(None, description="Detected document type")

# In-memory job storage (replace with database in production)
jobs = {}

async def process_document_task(
    job_id: str,
    front_image_path: str,
    back_image_path: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """Background task to process a document and save to MongoDB if user_id is provided."""
    try:
        # Record start time
        start_time = datetime.now()
        
        # Update job status and notify
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["processing_stage"] = "starting"
        jobs[job_id]["processing_message"] = "Starting document processing..."
        await webhook_notifier.notify_job_status(job_id, "processing")
        
        # Create output directory
        output_dir = UploadHandler.create_job_directory(job_id)
        
        # Process the document with the Qwen service
        logger.info(f"Processing document: job_id={job_id}")
        
        # Update processing stage for frontend
        jobs[job_id]["processing_stage"] = "detecting_type"
        jobs[job_id]["processing_message"] = "Detecting document type..."
        await webhook_notifier.notify_job_status(job_id, "processing")
        
        # Get QwenService instance
        service = get_qwen_service()
        
        # Get document detector
        detector = get_document_detector()
        
        # Detect document type first
        document_type = detector.detect_document_type(front_image_path)
        logger.info(f"Detected document type: {document_type}")
        
        # Update job with detected document type
        jobs[job_id]["document_type"] = document_type
        jobs[job_id]["processing_stage"] = "processing"
        jobs[job_id]["processing_message"] = f"Processing {document_type.replace('_', ' ')}..."
        await webhook_notifier.notify_job_status(job_id, "processing")
        
        # Process images with QwenService
        # Use a smaller token limit to reduce memory usage
        max_tokens = Config.get("MAX_TOKENS", 256)
        result = service.extract_info(
            front_image_path, 
            back_image_path, 
            max_tokens=max_tokens
        )
        logger.info(f"Document processing complete")
        
        # Clear CUDA cache after processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        # Check for error in the result
        if "error" in result:
            raise ValueError(f"Error extracting information: {result['error']}")
        
        # Update processing stage for frontend
        jobs[job_id]["processing_stage"] = "finalizing"
        jobs[job_id]["processing_message"] = "Finalizing results..."
        await webhook_notifier.notify_job_status(job_id, "processing")
        
        # Add document type to job info
        document_type = result.get("document_type", document_type)
        jobs[job_id]["document_type"] = document_type
        
        # Add raw response for debugging
        result["raw_model_response"] = result.get("raw_model_response", "")
        # Use ensure_ascii=False to preserve Unicode characters when converting to JSON string
        result["extracted_json_str"] = json.dumps(result, indent=2, ensure_ascii=False)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Ensure metadata exists and add additional info
        if "metadata" not in result:
            result["metadata"] = {}
            
        active_model = model_registry.get_active_model_config()
        result["metadata"].update({
            "processing_time": processing_time,
            "model_id": active_model.get("model_id"),
            "model_name": active_model.get("model_name"),
            "model_version": active_model.get("version"),
            "front_image_processed": True,
            "back_image_processed": back_image_path is not None,
            "document_type": document_type,
            "confidence_score": 0.8,  # Placeholder
            "processing_device": "cuda" if torch.cuda.is_available() else "cpu",
            "processing_steps": [
                {"stage": "document_type_detection", "success": True, "result": document_type},
                {"stage": "processing", "success": True},
                {"stage": "extraction", "success": True}
            ]
        })
        
        # Validate the results
        validation_result = result_validator.validate(result)
        result["validation"] = validation_result
        
        # Set job status based on validation result
        completion_status = "completed" if validation_result.get("valid", False) else "completed_with_warnings"
        
        # Save the extracted information as JSON in the output directory
        output_file = os.path.join(output_dir, f"{job_id}_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save to MongoDB if user_id is provided
        if user_id:
            try:
                # Import database modules
                from database import ocr_store, db, audit_log, image_store
                
                # Generate document ID
                document_id = f"{user_id}_{job_id}"
                
                # Create embedding from extracted text
                embedding = None
                try:
                    # Create an embedding from the extracted text using a simple method
                    # This can be replaced with more sophisticated embedding generation
                    import numpy as np
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    
                    # Get raw text from result
                    raw_text = result.get("raw_model_response", "")
                    
                    # Create TF-IDF vectorizer with limited features
                    vectorizer = TfidfVectorizer(max_features=100)
                    
                    # Generate embedding
                    text_array = vectorizer.fit_transform([raw_text]).toarray()
                    embedding = text_array[0]
                    
                    logger.info(f"Generated embedding with shape {embedding.shape}")
                except Exception as e:
                    logger.warning(f"Error generating embedding: {e}")
                
                # Prepare extracted data
                extracted_data = result.get("extracted_json", {})
                if not extracted_data and "extracted_json_str" in result:
                    try:
                        extracted_data = json.loads(result["extracted_json_str"])
                    except:
                        extracted_data = {}
                
                # Save OCR data with embedding
                ocr_store.save_ocr_data(
                    user_id=user_id,
                    document_id=document_id,
                    extracted_data=extracted_data,
                    embedding=embedding,
                    document_type=document_type,
                    job_id=job_id,
                    raw_text=result.get("raw_model_response", ""),
                    metadata=result.get("metadata", {})
                )
                
                # Log the operation
                audit_log.log_operation(
                    operation="ocr_extract",
                    user_id=user_id,
                    details={"job_id": job_id, "document_type": document_type}
                )
                
                # Save image metadata
                if front_image_path:
                    image_store.save_image_metadata(
                        image_id=f"{document_id}_front",
                        user_id=user_id,
                        job_id=job_id,
                        file_path=front_image_path,
                        file_name=os.path.basename(front_image_path),
                        document_type=document_type
                    )
                
                if back_image_path:
                    image_store.save_image_metadata(
                        image_id=f"{document_id}_back",
                        user_id=user_id,
                        job_id=job_id,
                        file_path=back_image_path,
                        file_name=os.path.basename(back_image_path),
                        document_type=document_type
                    )
                
                # Save basic user data if not exists
                db.save_user_data(user_id, {
                    "last_document_processed": datetime.now().isoformat(),
                    "last_document_type": document_type,
                    "last_job_id": job_id
                })
                
                logger.info(f"Saved OCR data to MongoDB for user_id={user_id}")
                
            except Exception as e:
                logger.error(f"Error saving to MongoDB: {e}", exc_info=True)
        
        # Update job status and notify
        jobs[job_id].update({
            "status": completion_status,
            "result": result,
            "completed_at": datetime.now().isoformat(),
            "document_type": document_type,
            "user_id": user_id  # Store user_id with job if provided
        })
        
        await webhook_notifier.notify_job_status(job_id, completion_status)
        
        logger.info(f"Job {job_id} completed in {processing_time:.2f}s with status: {completion_status}")
        
    except Exception as e:
        # Log the error
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
        
        # Update job status and notify
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat()
        })
        
        await webhook_notifier.notify_job_status(job_id, "failed")

@api_router.get("/")
async def root():
    """Root endpoint that returns basic API information."""
    return {
        "message": "Document OCR API is running",
        "docs": "/docs",
        "version": "1.0.0",
        "status": "operational"
    }

@api_router.post("/process", response_model=JobStatus)
async def process_document_endpoint(
    background_tasks: BackgroundTasks,
    front: UploadFile = File(..., description="Front side of the document"),
    back: Optional[UploadFile] = File(None, description="Back side of the document (optional)"),
    output_format: str = Form("json", description="Output format (json, yaml, csv)"),
    user_id: str = Form(None, description="User ID for storing extracted data"),
):
    """
    Process a document (vehicle registration card or ID card).
    
    This endpoint accepts front and optionally back images of the document,
    processes them using OCR, and extracts structured information.
    The extracted data can be associated with a user ID for storage in MongoDB.
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job entry
        created_at = datetime.now().isoformat()
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "created_at": created_at,
            "completed_at": None,
            "document_type": None,
            "processing_stage": "uploading",
            "processing_message": "Uploading document..."
        }
        
        # Save the uploaded images
        handler = UploadHandler(job_id)
        
        # Save front side
        front_image_path = await handler.save_file(front, "front")
        logger.info(f"Saved front image: {front_image_path}")
        
        # Save back side if provided
        back_image_path = None
        if back:
            back_image_path = await handler.save_file(back, "back")
            logger.info(f"Saved back image: {back_image_path}")
        
        # Schedule the background task to process the images
        background_tasks.add_task(
            process_document_task,
            job_id=job_id,
            front_image_path=front_image_path,
            back_image_path=back_image_path,
            user_id=user_id
        )
        
        return {
            "job_id": job_id,
            "status": "pending",
            "created_at": created_at,
            "document_type": None  # Will be updated during processing
        }
        
    except Exception as e:
        logger.error(f"Error starting processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error starting processing: {str(e)}")

@api_router.get("/jobs/{job_id}", response_model=Union[JobStatus, ProcessingResult])
async def get_job_status(
    job_id: str,
    include_result: bool = Query(True, description="Whether to include processing result"),
):
    """
    Get the status of a processing job.
    
    Retrieve the current status of a registration card processing job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "completed" and include_result:
        # Return full result
        return {
            "job_id": job_id,
            "status": job["status"],
            **job["result"]
        }
    else:
        # Return status only
        return JobStatus(
            job_id=job_id,
            status=job["status"],
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            document_type=job.get("document_type"),
        )

@api_router.get("/jobs", response_model=List[JobStatus])
async def list_jobs(
    limit: int = Query(10, description="Maximum number of jobs to return"),
    offset: int = Query(0, description="Offset for pagination"),
):
    """
    List processing jobs.
    
    Retrieve a list of registration card processing jobs.
    """
    job_list = [
        JobStatus(
            job_id=job_id,
            status=job["status"],
            created_at=job["created_at"],
            completed_at=job.get("completed_at"),
            document_type=job.get("document_type"),
        )
        for job_id, job in list(jobs.items())[offset:offset+limit]
    ]
    
    return job_list

@api_router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """
    Delete a job and its associated files.
    
    Remove a job from the system and clean up any associated files.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    # Clean up associated files
    files_to_clean = []
    if "front_image_path" in job and job["front_image_path"]:
        files_to_clean.append(job["front_image_path"])
    if "back_image_path" in job and job["back_image_path"]:
        files_to_clean.append(job["back_image_path"])
    
    UploadHandler.cleanup_files(files_to_clean)
    
    # Remove the job
    del jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

# Model management endpoints
@api_router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    status: Optional[str] = Query(None, description="Filter by status (development, staging, production, deprecated)"),
    limit: int = Query(10, description="Maximum number of models to return")
):
    """
    List available models in the registry.
    
    Retrieve information about available OCR models.
    """
    return model_registry.list_models(status=status, limit=limit)

@api_router.get("/models/active", response_model=Dict[str, Any])
async def get_active_model():
    """
    Get information about the currently active model.
    
    Retrieve details about the model currently used for OCR processing.
    """
    active_model = model_registry.get_active_model_config()
    if not active_model:
        raise HTTPException(status_code=404, detail="No active model found")
    return active_model

@api_router.put("/models/active/{model_id}")
async def set_active_model(model_id: str):
    """
    Set the active model for OCR processing.
    
    Change which model is used for processing registration cards.
    """
    success = model_registry.set_active_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model with ID {model_id} not found")
    
    # Reinitialize the OCR service with the new model
    global qwen_service
    qwen_service = None  # Force recreation with new model on next use
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    return {"message": f"Active model set to {model_id}"}

# Health check endpoint
@api_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "version": "1.0.0",
        "active_jobs": len(jobs),
        "active_model": model_registry.get_active_model_config().get("model_id"),
        "services": {
            "qwen_service": "available",
            "validator": "available",
            "webhooks": "available",
            "model_registry": "available"
        }
    }

@api_router.get("/jobs/{job_id}/json", response_class=JSONResponse)
async def get_job_raw_json(job_id: str):
    """
    Get the raw JSON result of a job.
    
    Retrieve the raw JSON result of a job.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if "result" not in job:
        raise HTTPException(status_code=404, detail="Job result not found")
    
    return job["result"]

@api_router.get("/users/{user_id}/documents")
async def get_user_documents(
    user_id: str,
    document_type: Optional[str] = Query(None, description="Filter by document type"),
    limit: int = Query(10, description="Maximum number of documents to return"),
    offset: int = Query(0, description="Offset for pagination"),
):
    """
    Get OCR documents for a user.
    
    Retrieve documents processed for a specific user.
    """
    try:
        from database import ocr_store
        
        # Get documents from MongoDB
        documents = ocr_store.get_user_documents(
            user_id=user_id,
            document_type=document_type,
            include_embeddings=False,
            limit=limit,
            offset=offset
        )
        
        # Convert ObjectId to str for JSON serialization
        for doc in documents:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
        
        return {"documents": documents, "count": len(documents)}
        
    except Exception as e:
        logger.error(f"Error retrieving user documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving user documents: {str(e)}")

@api_router.get("/users/{user_id}/documents/{document_id}")
async def get_user_document(
    user_id: str,
    document_id: str,
    include_embedding: bool = Query(False, description="Whether to include the embedding vector")
):
    """
    Get a specific OCR document for a user.
    
    Retrieve a specific document processed for a user.
    """
    try:
        from database import ocr_store
        
        # Get document from MongoDB
        document = ocr_store.get_document(
            document_id=document_id,
            include_embedding=include_embedding
        )
        
        # Check if document exists and belongs to the user
        if not document or document.get("user_id") != user_id:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Convert ObjectId to str for JSON serialization
        if "_id" in document:
            document["_id"] = str(document["_id"])
            
        # If embedding is included, convert to list for JSON serialization
        if include_embedding and "embedding" in document and document["embedding"] is not None:
            document["embedding"] = document["embedding"].tolist()
        
        return document
        
    except Exception as e:
        logger.error(f"Error retrieving document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

# Register the API router
app.include_router(api_router)

# THEN mount the static directories (after API routes)
try:
    # Mount the src directory first (more specific path)
    app.mount("/src", StaticFiles(directory=src_dir), name="src")
    
    # Mount assets directory
    assets_dir = os.path.join(public_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    
    # Mount the public directory for root paths
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="public")
    
    logger.info("Static directories mounted successfully")
except Exception as e:
    logger.error(f"Error mounting static directories: {e}", exc_info=True)
