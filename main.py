import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
import os
import cv2
import numpy as np
from typing import Optional, List
import json
import pandas as pd
from pathlib import Path
import uuid
import shutil
import jwt
from datetime import datetime, timedelta

from ocr.document_detector import detect_document_type
from ocr.processor import process_document, process_both_sides
from face_verification import FaceVerification
from person_database import PersonDatabase

# Configuration
SECRET_KEY = "your-secret-key-here"  # In production, use a secure secret key
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app = FastAPI(title="Qatar Document OCR with Face Verification")

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

# Initialize systems
face_verifier = None
person_db = None

try:
    print("Initializing Face Verification System...")
    face_verifier = FaceVerification()
    print("Face Verification System initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize Face Verification System: {e}")

try:
    print("Initializing Person Database...")
    person_db = PersonDatabase()
    print("Person Database initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize Person Database: {e}")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    user = person_db.get_person_by_username(username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@app.get("/")
async def get_index():
    return FileResponse("templates/login.html")

@app.get("/register")
async def get_register():
    return FileResponse("templates/register.html")

@app.get("/login")
async def get_login():
    return FileResponse("templates/login.html")

@app.get("/profile")
async def get_profile():
    return FileResponse("templates/profile.html")

@app.post("/api/register")
async def register(
    front_file: UploadFile = File(...),
    back_file: Optional[UploadFile] = File(None),
    document_type: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
    phone_number: str = Form(...)
):
    if person_db.username_exists(username):
        raise HTTPException(status_code=400, detail="Username already registered")

    # Save uploaded files
    front_path = f"uploads/{uuid.uuid4()}.jpg"
    with open(front_path, "wb") as f:
        f.write(await front_file.read())

    back_path = None
    if back_file:
        back_path = f"uploads/{uuid.uuid4()}.jpg"
        with open(back_path, "wb") as f:
            f.write(await back_file.read())

    try:
        # Process document with OCR
        if back_path:
            ocr_result = process_both_sides(front_path, back_path, document_type)
        else:
            ocr_result = process_document(front_path, document_type, "front")

        # Extract face from ID
        face = face_verifier.detect_face(front_path)
        if face is None:
            raise HTTPException(status_code=400, detail="No face detected in document image")

        # Save face image as profile picture
        profile_image_path = f"static/profile_images/{uuid.uuid4()}.jpg"
        os.makedirs(os.path.dirname(profile_image_path), exist_ok=True)
        cv2.imwrite(profile_image_path, face)

        # Get face embedding
        face_embedding = face_verifier.extract_features(face)
        if face_embedding is None:
            raise HTTPException(status_code=400, detail="Could not extract face features")

        # Generate person ID (use ID from document if available)
        person_id = None
        if ocr_result and isinstance(ocr_result, dict):
            if "personal_info" in ocr_result and "id_number" in ocr_result["personal_info"]:
                person_id = ocr_result["personal_info"]["id_number"]
            elif "id_number" in ocr_result:
                person_id = ocr_result["id_number"]

        if not person_id:
            person_id = str(uuid.uuid4())

        # Add person to database
        success = person_db.add_person(
            person_id=person_id,
            username=username,
            password=password,
            phone_number=phone_number,
            document_type=document_type,
            document_info=ocr_result,
            face_embedding=face_embedding,
            profile_image_path=profile_image_path
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to register user")

        # Create access token
        access_token = create_access_token({"sub": username})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "profile_image_url": f"/{profile_image_path}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        if os.path.exists(front_path):
            os.remove(front_path)
        if back_path and os.path.exists(back_path):
            os.remove(back_path)

@app.post("/api/login")
async def login(face_image: UploadFile = File(...)):
    try:
        # Save uploaded face image
        temp_path = f"uploads/{uuid.uuid4()}.jpg"
        with open(temp_path, "wb") as f:
            f.write(await face_image.read())

        # Detect and get embedding for the login face
        login_face = face_verifier.detect_face(temp_path)
        if login_face is None:
            raise HTTPException(status_code=400, detail="No face detected in the image")

        login_embedding = face_verifier.extract_features(login_face)
        if login_embedding is None:
            raise HTTPException(status_code=400, detail="Could not extract face features")

        # Get all face embeddings from database
        stored_embeddings = person_db.get_all_face_embeddings()

        # Find the best match using cosine similarity
        best_match_id = None
        best_match_score = float('-inf')
        
        def cosine_similarity(a, b):
            # Ensure inputs are numpy arrays
            if not isinstance(a, np.ndarray):
                a = np.array(a)
            if not isinstance(b, np.ndarray):
                b = np.array(b)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0
            
            return dot_product / (norm_a * norm_b)
        
        for person_id, stored_embedding in stored_embeddings.items():
            similarity = cosine_similarity(login_embedding, stored_embedding)
            if similarity > best_match_score:
                best_match_score = similarity
                best_match_id = person_id

        # Check if we found a match above threshold (0.7 is a common threshold for cosine similarity)
        SIMILARITY_THRESHOLD = 0.7
        if best_match_score < SIMILARITY_THRESHOLD:
            raise HTTPException(status_code=401, detail="Face not recognized")

        # Get user information
        user = person_db.get_person(best_match_id)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")

        # Create access token
        access_token = create_access_token({"sub": user["username"]})
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/profile")
async def get_profile_data(user: dict = Depends(get_current_user)):
    try:
        # Extract relevant information from document_info
        doc_info = user.get('document_info', {})
        if not doc_info:
            doc_info = {}

        personal_info = doc_info.get('personal_info', {})
        if not personal_info:
            personal_info = {}

        profile_data = {
            "name": personal_info.get('name', 'N/A'),
            "id_number": personal_info.get('id_number', 'N/A'),
            "nationality": personal_info.get('nationality', 'N/A'),
            "date_of_birth": personal_info.get('date_of_birth', 'N/A'),
            "username": user.get('username', 'N/A'),
            "phone_number": user.get('phone_number', 'N/A'),
            "document_type": user.get('document_type', 'N/A'),
            "document_expiry": doc_info.get('expiry_date', 'N/A'),
            "profile_image_url": f"/{user.get('profile_image_path', '')}"
        }
        
        return profile_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/login")
    return response

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

@app.post("/api/process-document-with-face")
async def process_document_with_face(request: Request, 
                                   front_file: UploadFile = File(...),
                                   back_file: Optional[UploadFile] = File(None),
                                   document_type: str = Form(...)):
    """
    Enhanced endpoint that processes documents with OCR and extracts face embeddings,
    storing both in a unified person database
    """
    if face_verifier is None:
        raise HTTPException(status_code=500, detail="Face verification system not initialized")
    
    if person_db is None:
        raise HTTPException(status_code=500, detail="Person database not initialized")
    
    # Save uploaded files temporarily
    front_path = f"uploads/front_{uuid.uuid4()}.jpg"
    with open(front_path, "wb") as f:
        f.write(await front_file.read())
    
    back_path = None
    if back_file:
        back_path = f"uploads/back_{uuid.uuid4()}.jpg"
        with open(back_path, "wb") as f:
            f.write(await back_file.read())
    
    try:
        # Step 1: Process OCR
        print("Processing OCR...")
        ocr_result = None
        if back_path:
            ocr_result = process_both_sides(front_path, back_path, document_type)
        else:
            ocr_result = process_document(front_path, document_type, "front")
        
        # Step 2: Extract person ID from OCR result
        person_id = None
        if ocr_result:
            # Try to get ID from different possible fields
            if "personal_info" in ocr_result and "id_number" in ocr_result["personal_info"]:
                person_id = ocr_result["personal_info"]["id_number"]
            elif "id_number" in ocr_result:
                person_id = ocr_result["id_number"]
            elif "document_info" in ocr_result and "id_number" in ocr_result["document_info"]:
                person_id = ocr_result["document_info"]["id_number"]
        
        # Fallback to UUID if no ID found
        if not person_id:
            person_id = str(uuid.uuid4())
            print(f"No ID number found in OCR result, using generated ID: {person_id}")
        
        # Step 3: Detect and extract face
        print("Detecting face...")
        face = face_verifier.detect_face(front_path)
        if face is None:
            return {
                "success": False,
                "message": "No face detected in document image",
                "ocr_result": ocr_result,
                "person_id": person_id
            }
        
        # Save detected face
        face_image_path = f"uploads/face_{person_id}.jpg"
        cv2.imwrite(face_image_path, face)
        
        # Step 4: Extract face embeddings
        print("Extracting face embeddings...")
        face_embeddings = face_verifier.extract_features(face)
        if face_embeddings is None:
            return {
                "success": False,
                "message": "Could not extract face features",
                "ocr_result": ocr_result,
                "person_id": person_id,
                "face_detected": True
            }
        
        # Step 5: Prepare document image paths
        document_images = {"front": front_path}
        if back_path:
            document_images["back"] = back_path
        
        # Step 6: Save everything to the unified person database
        print(f"Saving person record for {person_id}...")
        success = person_db.save_person_with_embeddings(
            person_id=person_id,
            ocr_data=ocr_result,
            face_embeddings=face_embeddings,
            face_image_path=face_image_path,
            document_images=document_images
        )
        
        if success:
            return {
                "success": True,
                "message": f"Person record created successfully for {person_id}",
                "person_id": person_id,
                "ocr_result": ocr_result,
                "face_verification": {
                    "face_detected": True,
                    "features_extracted": True,
                    "embedding_method": "ResNet50" if len(face_embeddings) == 512 else "HOG",
                    "face_image_path": face_image_path
                }
            }
        else:
            return {
                "success": False,
                "message": "Failed to save person record to database",
                "person_id": person_id,
                "ocr_result": ocr_result
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    finally:
        # Clean up temporary files (keep face and document images)
        pass

@app.post("/api/verify-person")
async def verify_person(request: Request,
                       test_image: UploadFile = File(...),
                       person_id: str = Form(...)):
    """
    Verify a person by comparing a test image against stored face embeddings
    """
    if face_verifier is None:
        raise HTTPException(status_code=500, detail="Face verification system not initialized")
    
    if person_db is None:
        raise HTTPException(status_code=500, detail="Person database not initialized")
    
    # Save test image temporarily
    test_image_path = f"uploads/test_{uuid.uuid4()}.jpg"
    with open(test_image_path, "wb") as f:
        f.write(await test_image.read())
    
    try:
        # Get stored person data
        person_data = person_db.get_person_by_id(person_id)
        if not person_data:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found in database")
        
        # Get stored face embeddings
        stored_embeddings = person_data.get("face_embeddings")
        if stored_embeddings is None:
            raise HTTPException(status_code=404, detail=f"No face embeddings found for person {person_id}")
        
        # Detect face in test image
        test_face = face_verifier.detect_face(test_image_path)
        if test_face is None:
            return {
                "success": False,
                "message": "No face detected in test image",
                "person_id": person_id
            }
        
        # Extract features from test face
        test_embeddings = face_verifier.extract_features(test_face)
        if test_embeddings is None:
            return {
                "success": False,
                "message": "Could not extract features from test face",
                "person_id": person_id
            }
        
        # Calculate similarity
        def cosine_similarity(a, b):
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0
            return dot_product / (norm_a * norm_b)
        
        similarity_score = cosine_similarity(test_embeddings, stored_embeddings)
        threshold = 0.7
        is_match = similarity_score > threshold
        
        # Get person information for response
        person_info = person_data.get("person_info", {})
        document_data = person_data.get("document_data", {})
        
        # Convert MongoDB ObjectId to string and numpy types to Python types
        def clean_for_json(obj):
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return {k: clean_for_json(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif hasattr(obj, '__str__') and 'ObjectId' in str(type(obj)):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            else:
                return obj
        
        return {
            "success": True,
            "verification_result": {
                "is_match": bool(is_match),
                "similarity_score": float(similarity_score),
                "threshold": float(threshold),
                "person_id": person_id
            },
            "person_info": {
                "personal_info": clean_for_json(person_info.get("personal_info", {})),
                "document_info": clean_for_json(person_info.get("document_info", {})),
                "ocr_data": clean_for_json(document_data)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during verification: {str(e)}")
    
    finally:
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

@app.get("/api/person/{person_id}")
async def get_person(person_id: str):
    """
    Get complete person information by person ID
    """
    if person_db is None:
        raise HTTPException(status_code=500, detail="Person database not initialized")
    
    try:
        person_data = person_db.get_person_by_id(person_id)
        if not person_data:
            raise HTTPException(status_code=404, detail=f"Person {person_id} not found")
        
        # Convert MongoDB ObjectId to string and numpy types to Python types
        def clean_for_json(obj):
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return {k: clean_for_json(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif hasattr(obj, '__str__') and 'ObjectId' in str(type(obj)):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            else:
                return obj
        
        # Remove embeddings from response (too large for API response)
        response_data = {
            "person_info": clean_for_json(person_data.get("person_info", {})),
            "document_data": clean_for_json(person_data.get("document_data", {})),
            "has_face_embeddings": person_data.get("face_embeddings") is not None,
            "embedding_method": person_data.get("embedding_method")
        }
        
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving person: {str(e)}")

@app.get("/api/search/by-document/{document_number}")
async def search_by_document_number(document_number: str):
    """
    Search for a person by document/ID number
    """
    if person_db is None:
        raise HTTPException(status_code=500, detail="Person database not initialized")
    
    try:
        person_data = person_db.get_person_by_document_number(document_number)
        if not person_data:
            raise HTTPException(status_code=404, detail=f"No person found with document number {document_number}")
        
        # Convert MongoDB ObjectId to string and numpy types to Python types
        def clean_for_json(obj):
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return {k: clean_for_json(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif hasattr(obj, '__str__') and 'ObjectId' in str(type(obj)):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            else:
                return obj
        
        # Remove embeddings from response
        response_data = {
            "person_info": clean_for_json(person_data.get("person_info", {})),
            "document_data": clean_for_json(person_data.get("document_data", {})),
            "has_face_embeddings": person_data.get("face_embeddings") is not None,
            "embedding_method": person_data.get("embedding_method")
        }
        
        return response_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching by document number: {str(e)}")

@app.get("/api/search/by-name/{name}")
async def search_by_name(name: str, limit: int = 10):
    """
    Search for persons by name
    """
    if person_db is None:
        raise HTTPException(status_code=500, detail="Person database not initialized")
    
    try:
        results = person_db.search_persons_by_name(name, limit)
        
        # Convert MongoDB ObjectId to string and numpy types to Python types
        def clean_for_json(obj):
            if hasattr(obj, 'tolist'):  # numpy array
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return {k: clean_for_json(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif hasattr(obj, '__str__') and 'ObjectId' in str(type(obj)):
                return str(obj)
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            else:
                return obj
        
        # Remove embeddings from response
        response_results = []
        for person_data in results:
            response_data = {
                "person_info": clean_for_json(person_data.get("person_info", {})),
                "document_data": clean_for_json(person_data.get("document_data", {})),
                "has_face_embeddings": person_data.get("face_embeddings") is not None,
                "embedding_method": person_data.get("embedding_method")
            }
            response_results.append(response_data)
        
        return {"results": response_results, "count": len(response_results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching by name: {str(e)}")

@app.get("/api/persons")
async def get_all_persons(limit: int = 50, include_inactive: bool = False):
    """
    Get all person records
    """
    if person_db is None:
        raise HTTPException(status_code=500, detail="Person database not initialized")
    
    try:
        persons = person_db.get_all_persons(include_inactive=include_inactive, limit=limit)
        
        # Return summary information only
        summary_results = []
        for person in persons:
            summary = {
                "person_id": person.get("person_id"),
                "name": person.get("personal_info", {}).get("name"),
                "id_number": person.get("personal_info", {}).get("id_number"),
                "nationality": person.get("personal_info", {}).get("nationality"),
                "created_at": person.get("created_at"),
                "status": person.get("status", "active")
            }
            summary_results.append(summary)
        
        return {"results": summary_results, "count": len(summary_results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving persons: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 