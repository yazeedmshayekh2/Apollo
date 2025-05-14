"""
API routes for the Vehicle Registration Card OCR system.
Provides endpoints for uploading and processing registration cards.
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
from utils.config import Config
from .upload import UploadHandler
from .validation import ResultValidator
from .webhooks import WebhookNotifier

# Set up logging
logging.basicConfig(level=Config.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vehicle Registration Card OCR API",
    description="API for extracting information from vehicle registration cards",
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
        qwen_service = QwenService()
    return qwen_service

# Define request/response models
class ProcessingOptions(BaseModel):
    """Options for processing registration cards."""
    output_format: str = Field("json", description="Output format (json, yaml, csv)")

class ProcessingResult(BaseModel):
    """Result of processing a registration card."""
    job_id: str = Field(..., description="Unique job ID")
    status: str = Field(..., description="Processing status")
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

# In-memory job storage (replace with database in production)
jobs = {}

async def process_registration_card_task(
    job_id: str,
    front_image_path: str,
    back_image_path: Optional[str] = None,
) -> None:
    """Background task to process a registration card."""
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
        
        # Process the registration card directly with the Qwen service
        logger.info(f"Processing registration card: job_id={job_id}")
        
        # Update processing stage for frontend
        jobs[job_id]["processing_stage"] = "processing"
        jobs[job_id]["processing_message"] = "Processing registration card..."
        await webhook_notifier.notify_job_status(job_id, "processing")
        
        # Get QwenService instance
        service = get_qwen_service()
        
        # Process images with QwenService - gets both front and back at once
        # Use a smaller token limit to reduce memory usage
        max_tokens = Config.get("MAX_TOKENS", 256)
        result = service.extract_info(
            front_image_path, 
            back_image_path, 
            max_tokens=max_tokens
        )
        logger.info(f"Registration card processing complete")
        
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
            "confidence_score": 0.8,  # Placeholder
            "processing_device": "cuda" if torch.cuda.is_available() else "cpu",
            "processing_steps": [
                {"stage": "processing", "success": True},
                {"stage": "extraction", "success": True},
                {"stage": "formatting", "success": True}
            ]
        })
        
        # Validate the results
        validated_result = result_validator.validate_result(result)
        
        # Update job status
        jobs[job_id].update({
            "status": "completed",
            "result": validated_result,
            "completed_at": datetime.now().isoformat(),
        })
        
        # Send webhook notification
        await webhook_notifier.notify_job_completed(job_id, validated_result)
        
        # Log completion
        logger.info(f"Job {job_id} completed successfully in {processing_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {e}", exc_info=True)
        
        # Update job status
        jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now().isoformat(),
        })
        
        # Log model failure
        try:
            active_model = model_registry.get_active_model_config()
            model_registry.log_model_performance(
                model_id=active_model.get("model_id"),
                metrics={
                    "success": False,
                    "error_type": type(e).__name__
                },
                sample_id=job_id
            )
        except Exception as log_error:
            logger.error(f"Error logging model failure: {log_error}")
        
        # Send webhook notification
        await webhook_notifier.notify_job_failed(job_id, str(e))

# API endpoints
@api_router.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Vehicle Registration Card OCR API"}

@api_router.post("/process", response_model=JobStatus)
async def process_registration_card_endpoint(
    background_tasks: BackgroundTasks,
    front: UploadFile = File(..., description="Front side of registration card"),
    back: Optional[UploadFile] = File(None, description="Back side of registration card"),
    output_format: str = Form("json", description="Output format (json, yaml, csv)"),
):
    """
    Process a vehicle registration card.
    
    Upload the front and (optionally) back sides of a vehicle registration card
    to extract structured information. Images are processed in their raw form without enhancement.
    """
    try:
        # Validate and save uploaded files
        front_image_path = await UploadHandler.save_file(front)
        back_image_path = None
        
        if back:
            back_image_path = await UploadHandler.save_file(back)
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Create job record
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            "front_image_path": front_image_path,
            "back_image_path": back_image_path,
            "output_format": output_format,
            "completed_at": None,
        }
        
        # Notify about job creation via webhook
        background_tasks.add_task(
            webhook_notifier.notify_job_status,
            job_id=job_id,
            status="queued"
        )
        
        # Add task to background queue
        background_tasks.add_task(
            process_registration_card_task,
            job_id=job_id,
            front_image_path=front_image_path,
            back_image_path=back_image_path,
        )
        
        logger.info(f"Job {job_id} created and queued for processing")
        
        return JobStatus(
            job_id=job_id,
            status="queued",
            created_at=jobs[job_id]["created_at"],
        )
        
    except Exception as e:
        logger.error(f"Error in process endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

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
    Get only the extracted JSON from a processing job.
    
    This endpoint returns only the raw extracted JSON from the VLM,
    either as parsed JSON if parsing succeeded or as a string if parsing failed.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job processing not completed")
    
    # If we have valid parsed JSON, return it directly
    if "result" in job and "extracted_json" in job["result"] and job["result"]["extracted_json"]:
        # Return with proper JSON encoding for Unicode characters
        return JSONResponse(
            content=job["result"]["extracted_json"],
            media_type="application/json; charset=utf-8"
        )
    
    # Otherwise, return the JSON string (client will need to handle this case)
    if "result" in job and "extracted_json_str" in job["result"]:
        return JSONResponse(
            content={"raw_json_string": job["result"]["extracted_json_str"]},
            media_type="application/json; charset=utf-8"
        )
    
    # Fallback to raw response if nothing else is available
    if "result" in job and "raw_model_response" in job["result"]:
        return JSONResponse(
            content={"raw_response": job["result"]["raw_model_response"]},
            media_type="application/json; charset=utf-8"
        )
    
    # If nothing is available
    raise HTTPException(status_code=404, detail="No JSON data found for this job")

# ADD API ROUTER FIRST - this is important for routing priority
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
