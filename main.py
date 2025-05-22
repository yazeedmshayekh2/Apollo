import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional
import json
import pandas as pd
from pathlib import Path

from ocr.document_detector import detect_document_type
from ocr.processor import process_document, process_both_sides

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(title="Qatar Document OCR")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def get_index():
    return FileResponse("templates/index.html")

@app.post("/api/detect-document")
async def detect_document(file: UploadFile = File(...)):
    """
    Detect document type from uploaded image
    """
    # Save uploaded file temporarily
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Detect document type
    doc_type, side = detect_document_type(file_path)
    
    return {"documentType": doc_type, "side": side, "filePath": file_path}

@app.post("/api/process-document")
async def process_document_api(
    file: UploadFile = File(...),
    document_type: str = Form(...),
    side: str = Form(...),
    output_format: str = Form("json")
):
    """
    Process document with OCR based on detected type and side
    """
    # Save uploaded file temporarily
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Process document
    result = process_document(file_path, document_type, side)
    
    # Return result in requested format
    if output_format == "csv":
        # Convert result to CSV
        df = pd.DataFrame([result])
        csv_path = f"uploads/{os.path.splitext(file.filename)[0]}.csv"
        df.to_csv(csv_path, index=False)
        return FileResponse(csv_path, media_type="text/csv", filename=f"{document_type}_{side}.csv")
    else:
        return JSONResponse(content=result)

@app.post("/api/process-document-both-sides")
async def process_document_both_sides_api(
    front_file: UploadFile = File(...),
    back_file: UploadFile = File(...),
    document_type: str = Form(...),
    output_format: str = Form("json")
):
    """
    Process both sides of a document and combine the results
    """
    # Save uploaded files temporarily
    front_path = f"uploads/front_{front_file.filename}"
    back_path = f"uploads/back_{back_file.filename}"
    
    with open(front_path, "wb") as f:
        f.write(await front_file.read())
    
    with open(back_path, "wb") as f:
        f.write(await back_file.read())
    
    # Verify that both images are of the same document type and correct sides
    front_detected_type, front_detected_side = detect_document_type(front_path)
    back_detected_type, back_detected_side = detect_document_type(back_path)
    
    # Check for potential errors
    errors = []
    
    # 1. Check if documents are of different types
    if front_detected_type != back_detected_type and front_detected_type != "unknown" and back_detected_type != "unknown":
        errors.append(f"The uploaded images appear to be different document types: Front is {front_detected_type}, Back is {back_detected_type}")
    
    # 2. Check if the sides are correctly provided
    if front_detected_side == "back" and back_detected_side == "front":
        errors.append("The images appear to be in the wrong order. The front image looks like a back side, and the back image looks like a front side.")
    
    # 3. Check if specified document type matches detected type
    if front_detected_type != "unknown" and front_detected_type != document_type:
        errors.append(f"The detected document type ({front_detected_type}) doesn't match the selected type ({document_type})")
    
    # 4. Provide a helpful message if both document detection and side detection failed
    if front_detected_type == "unknown" and back_detected_type == "unknown":
        errors.append("Could not automatically identify the document types. Please make sure the images are clear and contain the expected document information.")
    
    # Return errors if any were found
    if errors:
        return JSONResponse(
            status_code=400,
            content={"errors": errors}
        )
    
    # If all validations pass, process both sides using our unified processing
    combined_result = process_both_sides(front_path, back_path, document_type)
    
    # Return result in requested format
    if output_format == "csv":
        # Flatten the nested structure for CSV
        flat_result = {}
        
        # Function to recursively flatten nested dictionaries
        def flatten_dict(nested_dict, prefix=""):
            for key, value in nested_dict.items():
                if isinstance(value, dict):
                    # Recursively flatten nested dictionaries
                    flatten_dict(value, prefix + key + "_")
                else:
                    # Add leaf values to the flat result
                    flat_result[prefix + key] = value
        
        # Flatten the combined result
        flatten_dict(combined_result)
        
        # Convert to CSV
        df = pd.DataFrame([flat_result])
        csv_path = f"uploads/{document_type}_complete.csv"
        df.to_csv(csv_path, index=False)
        return FileResponse(csv_path, media_type="text/csv", filename=f"{document_type}_complete.csv")
    else:
        return JSONResponse(content=combined_result)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 