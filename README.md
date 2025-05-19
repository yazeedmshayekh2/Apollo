# Document OCR System with MongoDB & API Sharing

An advanced OCR system built on Qwen 2.5 Vision Language Model to extract structured information from documents like vehicle registration cards and ID cards, with MongoDB storage and API sharing capabilities.

## Features

- **High-Accuracy OCR**: Leverages Qwen 2.5 Vision model for superior text recognition performance
- **Multi-Document Support**: Processes vehicle registration cards and ID cards
- **Multi-Side Processing**: Handles both front and back sides of documents
- **Structured Data Extraction**: Outputs standardized JSON with document-specific fields
- **MongoDB Integration**: Stores extracted data and embeddings for future retrieval
- **Vector Embeddings**: Creates embeddings for similarity search and document comparison
- **API Sharing**: Easily expose your API to others via ngrok for testing and collaboration
- **User-Based Storage**: Associates documents with user IDs for organization
- **REST API**: Complete API for document upload, processing and retrieval
- **Web Interface**: Simple web UI for document upload and result viewing

## System Architecture

```
OCR System
│
├── API Layer
│   ├── Upload Service
│   ├── Validation Service
│   ├── REST Endpoints
│   ├── Ngrok Integration
│   └── Webhook Support
│
├── Core OCR Engine
│   ├── Qwen Vision Service
│   ├── Image Processor
│   ├── Data Extractor
│   ├── Document Type Detector
│   └── Model Registry
│
├── Database Layer
│   ├── MongoDB Connector
│   ├── OCR Data Store
│   ├── Image Metadata Store
│   ├── Vector Embeddings
│   └── Audit Logging
│
├── Frontend Layer
│   ├── Upload Interface
│   ├── Result Visualization
│   ├── Status Tracking
│   └── Document Management
│
└── Utilities
    ├── Configuration
    ├── Helpers
    ├── Validators
    └── Image Processing
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM (32GB+ recommended for production)
- MongoDB (local or remote instance)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/document-ocr-mongodb.git
   cd document-ocr-mongodb
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure MongoDB connection (in .env file or environment variables):
   ```
   MONGODB_URI=mongodb://localhost:27017
   MONGODB_DB=ocr_db
   ```

## Usage

### Running the API Server

Start the API server locally:

```bash
python -m uvicorn api.routes:app --host 0.0.0.0 --port 8000
```

Access the web interface at: http://localhost:8000/

### Sharing the API with Others

To expose your API via ngrok for others to test:

```bash
python ngrok_server.py
```

This will:
1. Start the API server locally
2. Create a secure public URL via ngrok
3. Generate comprehensive API documentation
4. Create a Postman collection for easy testing

Share the generated URL and documentation with your collaborators.

### Using the API

#### Process a Document

```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "front=@path/to/front_image.jpg" \
  -F "back=@path/to/back_image.jpg" \
  -F "user_id=user123" \
  -F "output_format=json"
```

#### Check Job Status

```bash
curl "http://localhost:8000/api/jobs/{job_id}"
```

#### Get User Documents

```bash
curl "http://localhost:8000/api/users/{user_id}/documents"
```

### Python Client Integration

```python
import requests

# Process a document
files = {
    'front': ('front.jpg', open('path/to/front.jpg', 'rb'), 'image/jpeg'),
    'back': ('back.jpg', open('path/to/back.jpg', 'rb'), 'image/jpeg')
}
data = {
    'user_id': 'user123',
    'output_format': 'json'
}
response = requests.post('http://localhost:8000/api/process', files=files, data=data)
job_id = response.json()['job_id']

# Check job status
response = requests.get(f'http://localhost:8000/api/jobs/{job_id}')
result = response.json()

# Get user documents
response = requests.get(f'http://localhost:8000/api/users/user123/documents')
documents = response.json()['documents']
```

## MongoDB Data Structure

The system uses the following collections in MongoDB:

- **users**: Basic user information
- **ocr_data**: Main collection for OCR results with embeddings
- **images**: Metadata about processed images
- **audit_log**: Operation tracking

Document structure example in `ocr_data`:

```json
{
  "user_id": "user123",
  "document_id": "user123_job456",
  "job_id": "job456",
  "document_type": "vehicle_registration",
  "extracted_data": {
    "vehicle_info": { ... },
    "owner_info": { ... },
    "registration_info": { ... }
  },
  "embedding": "<Binary data>",
  "embedding_dim": 100,
  "raw_text": "...",
  "created_at": "2023-10-15T14:30:00.000Z",
  "metadata": { ... }
}
```

## API Documentation

Full API documentation is available at `/docs` when the server is running.

When using the ngrok sharing feature, comprehensive documentation is automatically generated:
- `OCR_API_DOCUMENTATION.md`: API documentation in Markdown format
- `ocr_api_postman.json`: Ready-to-import Postman collection

## Development

### Running Tests

```bash
pytest tests/
```

### Testing the API directly

```bash
python test_api_endpoint.py
```

### Testing with MongoDB

```bash
python test_ocr_with_mongodb.py
```

## Docker Deployment

Build and run with Docker:

```bash
docker build -t document-ocr-mongodb .
docker run -p 8000:8000 \
  -e MONGODB_URI=mongodb://host.docker.internal:27017 \
  -v $(pwd)/sample_data:/app/sample_data \
  document-ocr-mongodb
```

## Model Information

This system uses the Qwen 2.5 Vision Language Model:

- **Model**: Qwen2.5-VL-7B-Instruct
- **Size**: 7 billion parameters
- **Context Window**: Supports up to 16K tokens
- **Visual Tokens**: 4-16384 tokens per image
- **Multimodal Understanding**: Processes both text and images together

## License

[MIT License](LICENSE)

## Contributors

- Your Name - Initial work
