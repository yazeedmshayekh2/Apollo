# Integrated OCR + Face Verification System with MongoDB

This enhanced system combines OCR document processing with face verification, storing both extracted information and face embeddings in a unified MongoDB database for each person.

## Features

### 🔍 **OCR Processing**
- Extract information from Qatar residency permits and vehicle registration cards
- Support for both front and back sides of documents
- Structured JSON output with personal information, document details, and metadata

### 👤 **Face Verification**
- Automatic face detection using YOLOv11L or OpenCV fallback
- Face feature extraction using ResNet-50 or HOG descriptors
- Cosine similarity-based face matching with configurable thresholds

### 🗄️ **Unified Database Storage**
- **Person Records**: Main collection with essential information
- **Document Records**: Detailed OCR data and document images
- **Face Embeddings**: Face feature vectors with metadata
- **Automatic Indexing**: Optimized for fast searches by ID, name, and document number

### 🔄 **Robust Fallback System**
- Local file storage when MongoDB is unavailable
- Graceful degradation with full functionality preservation

## Database Schema

### Collections Structure

```javascript
// persons collection
{
  "person_id": "28140001175",
  "created_at": ISODate("2024-01-15T10:30:00Z"),
  "updated_at": ISODate("2024-01-15T10:30:00Z"),
  "status": "active",
  "document_info": {
    "document_type": "Qatar Residency Permit",
    "issuing_authority": "State of Qatar"
  },
  "personal_info": {
    "id_number": "28140001175",
    "name": "Ahmed Hassan Al-Mahmoud",
    "arabic_name": "أحمد حسن المحمود",
    "nationality": "JORDAN",
    "date_of_birth": "15/03/1985",
    "expiry_date": "15/03/2027",
    "occupation": "Software Engineer"
  },
  "additional_info": {
    "sponsor": "Qatar Tech Company",
    "address": "Doha, Qatar"
  },
  "face_image_path": "uploads/face_28140001175.jpg",
  "document_images": {
    "front": "uploads/front_28140001175.jpg",
    "back": "uploads/back_28140001175.jpg"
  }
}

// documents collection
{
  "person_id": "28140001175",
  "document_id": "doc_uuid_here",
  "created_at": ISODate("2024-01-15T10:30:00Z"),
  "ocr_data": { /* Complete OCR result */ },
  "document_images": { /* Image paths */ }
}

// face_embeddings collection
{
  "person_id": "28140001175",
  "embedding_id": "emb_uuid_here",
  "created_at": ISODate("2024-01-15T10:30:00Z"),
  "face_embeddings": [0.1, 0.2, 0.3, ...], // 512-dim for ResNet50
  "embedding_method": "ResNet50",
  "face_image_path": "uploads/face_28140001175.jpg"
}
```

## API Endpoints

### 📄 **Document Processing with Face Extraction**

```http
POST /api/process-document-with-face
Content-Type: multipart/form-data

Parameters:
- front_file: Image file (required)
- back_file: Image file (optional)
- document_type: String (required) - "residency" or "vehicle"
```

**Response:**
```json
{
  "success": true,
  "message": "Person record created successfully for 28140001175",
  "person_id": "28140001175",
  "ocr_result": { /* OCR extracted data */ },
  "face_verification": {
    "face_detected": true,
    "features_extracted": true,
    "embedding_method": "ResNet50",
    "face_image_path": "uploads/face_28140001175.jpg"
  }
}
```

### 🔍 **Person Verification**

```http
POST /api/verify-person
Content-Type: multipart/form-data

Parameters:
- test_image: Image file (required)
- person_id: String (required)
```

**Response:**
```json
{
  "success": true,
  "verification_result": {
    "is_match": true,
    "similarity_score": 0.8542,
    "threshold": 0.7,
    "person_id": "28140001175"
  },
  "person_info": {
    "personal_info": { /* Personal details */ },
    "document_info": { /* Document details */ },
    "ocr_data": { /* Complete OCR data */ }
  }
}
```

### 📋 **Person Retrieval and Search**

```http
# Get person by ID
GET /api/person/{person_id}

# Search by document number
GET /api/search/by-document/{document_number}

# Search by name
GET /api/search/by-name/{name}?limit=10

# Get all persons
GET /api/persons?limit=50&include_inactive=false
```

## Python Usage Examples

### Basic Integration

```python
from person_database import PersonDatabase
from face_verification import FaceVerification
from ocr.processor import process_document

# Initialize systems
person_db = PersonDatabase()
face_verifier = FaceVerification()

# Process a document with face extraction
def process_id_card(front_image_path, back_image_path=None):
    # Step 1: Extract OCR data
    if back_image_path:
        ocr_data = process_both_sides(front_image_path, back_image_path, "residency")
    else:
        ocr_data = process_document(front_image_path, "residency", "front")
    
    # Step 2: Extract person ID
    person_id = ocr_data.get("personal_info", {}).get("id_number")
    if not person_id:
        person_id = str(uuid.uuid4())
    
    # Step 3: Process face
    face = face_verifier.detect_face(front_image_path)
    if face is None:
        return {"error": "No face detected"}
    
    face_embeddings = face_verifier.extract_features(face)
    if face_embeddings is None:
        return {"error": "Could not extract face features"}
    
    # Step 4: Save to database
    success = person_db.save_person_with_embeddings(
        person_id=person_id,
        ocr_data=ocr_data,
        face_embeddings=face_embeddings,
        face_image_path=f"uploads/face_{person_id}.jpg",
        document_images={"front": front_image_path, "back": back_image_path}
    )
    
    return {"success": success, "person_id": person_id}
```

### Face Verification

```python
def verify_person(test_image_path, person_id):
    # Get stored person data
    person_data = person_db.get_person_by_id(person_id)
    if not person_data:
        return {"error": "Person not found"}
    
    # Get stored embeddings
    stored_embeddings = person_data["face_embeddings"]
    
    # Process test image
    test_face = face_verifier.detect_face(test_image_path)
    test_embeddings = face_verifier.extract_features(test_face)
    
    # Calculate similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    similarity = cosine_similarity(test_embeddings, stored_embeddings)
    is_match = similarity > 0.7
    
    return {
        "is_match": is_match,
        "similarity_score": similarity,
        "person_info": person_data["person_info"]
    }
```

### Search Operations

```python
# Search by document number
person = person_db.get_person_by_document_number("28140001175")

# Search by name (supports partial matching)
results = person_db.search_persons_by_name("Ahmed", limit=10)

# Get all active persons
all_persons = person_db.get_all_persons(include_inactive=False, limit=100)

# Update person information
updates = {"personal_info.occupation": "Senior Engineer"}
success = person_db.update_person_info(person_id, updates)
```

## Installation and Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start MongoDB

```bash
# Ubuntu/Debian
sudo service mongod start

# Or using Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### 3. Run the Application

```bash
python main.py
```

### 4. Test the Integration

```bash
python test_integrated_system.py
```

## Configuration

### MongoDB Settings

```python
# Default configuration
person_db = PersonDatabase(
    mongo_uri='mongodb://localhost:27017/',
    database_name='person_verification'
)

# Custom configuration
person_db = PersonDatabase(
    mongo_uri='mongodb://username:password@host:port/',
    database_name='custom_db_name'
)
```

### Face Verification Thresholds

```python
# In face verification logic
threshold = 0.7  # Adjust based on your security requirements
# 0.6 - More permissive (higher false positives)
# 0.8 - More strict (higher false negatives)
```

## Performance Considerations

### Database Indexes
The system automatically creates indexes on:
- `person_id` (unique)
- `personal_info.id_number`
- `document_info.id_number`
- `personal_info.name`

### Memory Usage
- **ResNet-50 embeddings**: 512 dimensions × 4 bytes = 2KB per person
- **HOG embeddings**: ~3,780 dimensions × 4 bytes = 15KB per person
- **OCR data**: Typically 1-5KB per person

### Recommended Hardware
- **RAM**: 8GB+ (for model loading)
- **GPU**: CUDA-compatible (optional, for faster processing)
- **Storage**: SSD recommended for MongoDB

## Error Handling and Fallbacks

### MongoDB Unavailable
- Automatic fallback to local file storage
- Data saved in `local_storage/` directory
- Seamless operation continuation

### Face Detection Failures
- YOLOv11L → OpenCV Haar Cascade → Center crop fallback
- Graceful degradation with informative error messages

### OCR Processing Errors
- Robust JSON parsing with fallback to raw text
- Metadata preservation for debugging

## Security Considerations

### Data Privacy
- Face embeddings are mathematical representations, not images
- Original images can be optionally deleted after processing
- Soft delete functionality preserves audit trails

### Access Control
- Implement authentication middleware for production use
- Consider rate limiting for API endpoints
- Use HTTPS in production environments

## Monitoring and Maintenance

### Health Checks
```python
# Check system status
def health_check():
    return {
        "mongodb": person_db._is_connected(),
        "face_verifier": face_verifier is not None,
        "storage": os.path.exists("uploads/")
    }
```

### Database Maintenance
```python
# Clean up inactive records (optional)
person_db.get_all_persons(include_inactive=True)

# Backup embeddings
for person in person_db.get_all_persons():
    embeddings = person_db.get_face_embeddings(person["person_id"])
    # Save to backup location
```

## Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   ```bash
   # Check if MongoDB is running
   sudo service mongod status
   
   # Check connection
   mongo --eval "db.adminCommand('ping')"
   ```

2. **Face Detection Not Working**
   ```python
   # Check model files
   ls -la yolov11l-face.pt
   
   # Test with different images
   face = face_verifier.detect_face("test_image.jpg")
   ```

3. **OCR Processing Slow**
   ```python
   # Check GPU availability
   import torch
   print(torch.cuda.is_available())
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 