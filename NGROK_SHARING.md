# API Sharing with Ngrok

This project includes a seamless API sharing capability using ngrok, allowing you to expose your locally running OCR API to collaborators over the internet securely.

## What is Ngrok?

Ngrok is a tool that creates secure tunnels to expose local servers to the internet. It allows others to access your locally running API without complex network configuration or server deployment.

## Prerequisites

1. The OCR API project installed and configured
2. MongoDB running (if you want to store data)
3. Python 3.8+ with pyngrok installed (included in requirements.txt)

## Basic Usage

To share your API:

```bash
python ngrok_server.py
```

This script will:
1. Start the FastAPI server locally on port 8000
2. Create a secure ngrok tunnel to that port
3. Generate comprehensive API documentation
4. Create a Postman collection for easy testing

## Authentication (Optional)

For higher rate limits and additional features, you can sign up for a free ngrok account and use your authtoken:

```bash
export NGROK_AUTH_TOKEN=your_token_here
python ngrok_server.py
```

## Output Files

When you run the script, it generates:

1. **OCR_API_DOCUMENTATION.md**: Comprehensive API documentation in Markdown format
2. **ocr_api_postman.json**: A ready-to-import Postman collection
3. **api_docs.json**: API documentation in JSON format (for programmatic use)

## Sharing with Collaborators

Share the following with your collaborators:

1. The ngrok URL displayed in the console (e.g., https://abc123.ngrok-free.app)
2. The OCR_API_DOCUMENTATION.md file
3. The ocr_api_postman.json file

Your collaborator can then:
1. Import the Postman collection
2. Update the collection variables with the ngrok URL
3. Test the API directly without installing anything

## Testing the Shared API

### With Postman

1. Import ocr_api_postman.json in Postman
2. Update the collection variables (if needed)
3. Use the predefined requests

### With curl

```bash
# Health check
curl "https://abc123.ngrok-free.app/api/health"

# Process a document
curl -X POST "https://abc123.ngrok-free.app/api/process" \
  -F "front=@path/to/image.jpg" \
  -F "user_id=test_user"
```

### With Python

```python
import requests

# Health check
response = requests.get("https://abc123.ngrok-free.app/api/health")
print(response.json())

# Process a document
files = {"front": open("path/to/image.jpg", "rb")}
data = {"user_id": "test_user"}
response = requests.post("https://abc123.ngrok-free.app/api/process", 
                         files=files, data=data)
print(response.json())
```

## Demo Script

The `test_ocr_api.py` script provides an easy way to test a shared API:

```bash
python test_ocr_api.py https://abc123.ngrok-free.app
```

## Security Considerations

1. Ngrok tunnels are secure, but you're exposing your API to the internet
2. Consider implementing API keys for production use
3. Be cautious with sensitive data
4. The free tier of ngrok has connection limitations

## Limitations

1. Ngrok free tier has connection and rate limits
2. The tunnel closes when you stop the script
3. The URL changes each time unless you have a paid ngrok account

## Troubleshooting

1. **"Address already in use" error**: API server is already running on port 8000
   ```bash
   pkill -f "uvicorn api.routes:app"
   ```

2. **"Tunnel session failed" error**: ngrok session limit exceeded
   - Restart your computer to reset the connection
   - Sign up for a free ngrok account and use your authtoken

3. **"Failed to connect to local port" error**: API server failed to start
   - Check the console for specific error messages
   - Make sure all dependencies are installed

## When to Use

- Development and testing
- Quick demonstrations
- Collaborating with remote team members
- Getting feedback on API functionality
- Testing from mobile devices 