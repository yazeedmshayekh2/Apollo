#!/usr/bin/env python3
"""
Script to expose the OCR API using ngrok.
This creates a public URL that can be shared with others for testing.
"""

import os
import sys
import subprocess
import time
import json
from pyngrok import ngrok, conf

# Configure ngrok
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
API_PORT = 8000
DEFAULT_REGION = "us"

def setup_ngrok():
    """Set up ngrok configuration."""   
    if NGROK_AUTH_TOKEN:
        print(f"Setting up ngrok with auth token...")
        conf.get_default().auth_token = NGROK_AUTH_TOKEN
    else:
        print("No NGROK_AUTH_TOKEN found in environment variables.")
        print("You can still use ngrok, but with limitations.")
        print("For higher rate limits, set your auth token with:")
        print("export NGROK_AUTH_TOKEN=your_token_here\n")

def start_api_server():
    """Start the FastAPI server in the background."""
    print(f"Starting FastAPI server on port {API_PORT}...")
    # Use uvicorn directly (assuming it's installed in the environment)
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.routes:app", f"--port={API_PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give the server a moment to start
    time.sleep(2)
    
    # Check if the process is still running
    if process.poll() is not None:
        print("Error starting the API server:")
        stdout, stderr = process.communicate()
        print(stderr)
        sys.exit(1)
    
    return process

def start_ngrok_tunnel(port):
    """Start an ngrok tunnel to the specified port."""
    print(f"Starting ngrok tunnel to port {port}...")
    
    # Create a tunnel to the API port
    public_url = ngrok.connect(port, bind_tls=True).public_url
    print(f"\n✓ Ngrok tunnel active!")
    print(f"✓ Public URL: {public_url}")
    return public_url

def create_api_docs(public_url):
    """Create a simple API documentation guide."""
    api_docs = {
        "api_name": "OCR Document Processing API",
        "base_url": public_url,
        "description": "API for OCR processing of document images, including vehicle registration cards and ID cards",
        "endpoints": [
            {
                "path": "/api/process",
                "method": "POST",
                "description": "Process a document image with OCR",
                "parameters": [
                    {
                        "name": "front",
                        "type": "file",
                        "required": True,
                        "description": "Front side of the document (image file)"
                    },
                    {
                        "name": "back",
                        "type": "file",
                        "required": False,
                        "description": "Back side of the document (image file, optional)"
                    },
                    {
                        "name": "output_format",
                        "type": "string",
                        "required": False,
                        "description": "Output format (json, yaml, csv)",
                        "default": "json"
                    },
                    {
                        "name": "user_id",
                        "type": "string",
                        "required": True,
                        "description": "User ID for storing the results"
                    }
                ],
                "response": {
                    "job_id": "Unique job identifier",
                    "status": "Job status (pending, processing, completed, failed)",
                    "created_at": "Timestamp when the job was created"
                }
            },
            {
                "path": "/api/jobs/{job_id}",
                "method": "GET",
                "description": "Get the status and results of a processing job",
                "parameters": [
                    {
                        "name": "job_id",
                        "type": "path",
                        "required": True,
                        "description": "Job ID from the process endpoint"
                    },
                    {
                        "name": "include_result",
                        "type": "query",
                        "required": False,
                        "description": "Whether to include the full result in the response",
                        "default": "true"
                    }
                ]
            },
            {
                "path": "/api/jobs/{job_id}/json",
                "method": "GET",
                "description": "Get the raw JSON result of a job",
                "parameters": [
                    {
                        "name": "job_id",
                        "type": "path",
                        "required": True,
                        "description": "Job ID from the process endpoint"
                    }
                ]
            },
            {
                "path": "/api/users/{user_id}/documents",
                "method": "GET",
                "description": "Get all documents processed for a user",
                "parameters": [
                    {
                        "name": "user_id",
                        "type": "path",
                        "required": True,
                        "description": "User ID"
                    },
                    {
                        "name": "document_type",
                        "type": "query",
                        "required": False,
                        "description": "Filter by document type"
                    },
                    {
                        "name": "limit",
                        "type": "query",
                        "required": False,
                        "description": "Maximum number of documents to return",
                        "default": "10"
                    },
                    {
                        "name": "offset",
                        "type": "query",
                        "required": False,
                        "description": "Offset for pagination",
                        "default": "0"
                    }
                ]
            },
            {
                "path": "/api/users/{user_id}/documents/{document_id}",
                "method": "GET",
                "description": "Get a specific document processed for a user",
                "parameters": [
                    {
                        "name": "user_id",
                        "type": "path",
                        "required": True,
                        "description": "User ID"
                    },
                    {
                        "name": "document_id",
                        "type": "path",
                        "required": True,
                        "description": "Document ID"
                    },
                    {
                        "name": "include_embedding",
                        "type": "query",
                        "required": False,
                        "description": "Whether to include the embedding vector",
                        "default": "false"
                    }
                ]
            }
        ],
        "example_workflow": [
            "1. Upload an image using POST /api/process with form-data containing front image and user_id",
            "2. Note the job_id from the response",
            "3. Poll GET /api/jobs/{job_id} until status is 'completed' or 'failed'",
            "4. If completed, examine the extracted data in the response",
            "5. Optionally, retrieve all user documents with GET /api/users/{user_id}/documents"
        ],
        "notes": [
            "The API processes document images and extracts structured information",
            "Vehicle registration cards and ID cards are the primary supported document types",
            "For testing, use any unique string as the user_id"
        ]
    }
    
    # Write API docs to file
    with open("api_docs.json", "w") as f:
        json.dump(api_docs, f, indent=2)
    
    # Write a markdown version for better readability
    with open("OCR_API_DOCUMENTATION.md", "w") as f:
        f.write(f"# {api_docs['api_name']}\n\n")
        f.write(f"Base URL: {api_docs['base_url']}\n\n")
        f.write(f"{api_docs['description']}\n\n")
        
        f.write("## Endpoints\n\n")
        for endpoint in api_docs['endpoints']:
            f.write(f"### {endpoint['method']} {endpoint['path']}\n\n")
            f.write(f"{endpoint['description']}\n\n")
            
            if endpoint.get('parameters'):
                f.write("**Parameters:**\n\n")
                for param in endpoint['parameters']:
                    required = "Required" if param['required'] else "Optional"
                    default = f" (default: {param['default']})" if 'default' in param else ""
                    f.write(f"- `{param['name']}` ({param['type']}): {param['description']} - {required}{default}\n")
                f.write("\n")
            
            if endpoint.get('response'):
                f.write("**Response:**\n\n")
                for key, desc in endpoint['response'].items():
                    f.write(f"- `{key}`: {desc}\n")
                f.write("\n")
        
        f.write("## Example Workflow\n\n")
        for step in api_docs['example_workflow']:
            f.write(f"{step}\n")
        f.write("\n")
        
        f.write("## Postman Collection\n\n")
        f.write("You can import the included `ocr_api_postman.json` file into Postman to quickly test the API.\n\n")
        
        f.write("## Notes\n\n")
        for note in api_docs['notes']:
            f.write(f"- {note}\n")
    
    print(f"\n✓ API documentation created:")
    print(f"  - api_docs.json (JSON format)")
    print(f"  - OCR_API_DOCUMENTATION.md (Markdown format)")
    
    return api_docs

def create_postman_collection(api_docs):
    """Create a Postman collection for the API."""
    base_url = api_docs["base_url"]
    
    collection = {
        "info": {
            "name": "OCR Document API",
            "description": api_docs["description"],
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "item": [
            {
                "name": "Process Document",
                "request": {
                    "method": "POST",
                    "header": [],
                    "url": {
                        "raw": f"{base_url}/api/process",
                        "host": [base_url.replace("https://", "").replace("http://", "")],
                        "path": ["api", "process"]
                    },
                    "description": "Upload and process a document image",
                    "body": {
                        "mode": "formdata",
                        "formdata": [
                            {
                                "key": "front",
                                "type": "file",
                                "description": "Front side of the document"
                            },
                            {
                                "key": "back",
                                "type": "file",
                                "description": "Back side of the document (optional)"
                            },
                            {
                                "key": "output_format",
                                "value": "json",
                                "type": "text",
                                "description": "Output format"
                            },
                            {
                                "key": "user_id",
                                "value": "test_user_123",
                                "type": "text",
                                "description": "User ID for storing results"
                            }
                        ]
                    }
                }
            },
            {
                "name": "Check Job Status",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": f"{base_url}/api/jobs/{{job_id}}?include_result=true",
                        "host": [base_url.replace("https://", "").replace("http://", "")],
                        "path": ["api", "jobs", "{{job_id}}"],
                        "query": [
                            {
                                "key": "include_result",
                                "value": "true"
                            }
                        ]
                    },
                    "description": "Check the status of a processing job"
                }
            },
            {
                "name": "Get Raw Job Results",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": f"{base_url}/api/jobs/{{job_id}}/json",
                        "host": [base_url.replace("https://", "").replace("http://", "")],
                        "path": ["api", "jobs", "{{job_id}}", "json"]
                    },
                    "description": "Get the raw JSON result of a job"
                }
            },
            {
                "name": "List User Documents",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": f"{base_url}/api/users/{{user_id}}/documents?limit=10&offset=0",
                        "host": [base_url.replace("https://", "").replace("http://", "")],
                        "path": ["api", "users", "{{user_id}}", "documents"],
                        "query": [
                            {
                                "key": "limit",
                                "value": "10"
                            },
                            {
                                "key": "offset",
                                "value": "0"
                            },
                            {
                                "key": "document_type",
                                "value": "",
                                "disabled": True
                            }
                        ]
                    },
                    "description": "List all documents for a user"
                }
            },
            {
                "name": "Get Specific Document",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": f"{base_url}/api/users/{{user_id}}/documents/{{document_id}}?include_embedding=false",
                        "host": [base_url.replace("https://", "").replace("http://", "")],
                        "path": ["api", "users", "{{user_id}}", "documents", "{{document_id}}"],
                        "query": [
                            {
                                "key": "include_embedding",
                                "value": "false"
                            }
                        ]
                    },
                    "description": "Get a specific document for a user"
                }
            },
            {
                "name": "Health Check",
                "request": {
                    "method": "GET",
                    "header": [],
                    "url": {
                        "raw": f"{base_url}/api/health",
                        "host": [base_url.replace("https://", "").replace("http://", "")],
                        "path": ["api", "health"]
                    },
                    "description": "Check if the API is healthy"
                }
            }
        ],
        "variable": [
            {
                "key": "job_id",
                "value": "replace_with_job_id"
            },
            {
                "key": "user_id",
                "value": "test_user_123"
            },
            {
                "key": "document_id",
                "value": "replace_with_document_id"
            }
        ]
    }
    
    # Write Postman collection to file
    with open("ocr_api_postman.json", "w") as f:
        json.dump(collection, f, indent=2)
    
    print(f"  - ocr_api_postman.json (Postman collection)")

def main():
    """Main function to run the ngrok server."""
    print("=" * 80)
    print("OCR API Ngrok Server")
    print("=" * 80)
    
    # Set up ngrok
    setup_ngrok()
    
    # Start API server
    api_process = start_api_server()
    
    try:
        # Start ngrok tunnel
        public_url = start_ngrok_tunnel(API_PORT)
        
        # Create API documentation
        api_docs = create_api_docs(public_url)
        
        # Create Postman collection
        create_postman_collection(api_docs)
        
        print("\n" + "=" * 80)
        print(f"Share the following information with your friend:")
        print(f"1. Public URL: {public_url}")
        print(f"2. API Documentation (OCR_API_DOCUMENTATION.md)")
        print(f"3. Postman Collection (ocr_api_postman.json)")
        print("=" * 80)
        
        print("\nPress Ctrl+C to stop the server...")
        
        # Keep the tunnel open
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        
    finally:
        # Clean up
        print("Closing ngrok tunnel...")
        ngrok.disconnect(public_url)
        ngrok.kill()
        
        print("Stopping API server...")
        api_process.terminate()
        api_process.wait()
        
        print("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main() 