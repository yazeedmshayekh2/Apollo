# Face Verification Testing Guide

This guide explains how to test the face verification system with two images to verify if they show the same person.

## Overview

The face verification system works in two steps:
1. **Registration**: Process an ID card image to extract OCR data and face embeddings
2. **Verification**: Compare a test image against the stored face embeddings

## Test Scripts Available

### 1. `test_full_face_verification.py` - Comprehensive Test Suite
This is the most complete test script with multiple modes:

```bash
python test_full_face_verification.py
```

**Features:**
- Interactive mode: You provide image paths
- Automated mode: Uses sample images
- Detailed analysis of results
- Similarity score interpretation
- Pass/fail determination

### 2. `test_two_images_example.py` - Simple Example
A straightforward script for basic testing:

```bash
python test_two_images_example.py
```

**Features:**
- Simple two-step process
- Clear result interpretation
- Good for quick tests

### 3. `test_api_with_yolov11n.py` - API Integration Test
Tests the complete API with sample images:

```bash
python test_api_with_yolov11n.py
```

## How to Test with Your Own Images

### Step 1: Prepare Your Images

You need two images:
1. **ID Card Image**: A clear photo of an ID card/document with a visible face
2. **Verification Image**: Another photo of the same person (or different person for negative testing)

**Image Requirements:**
- Clear, well-lit images
- Face should be visible and not obscured
- Supported formats: JPG, JPEG, PNG
- Recommended resolution: At least 300x300 pixels

### Step 2: Start the Server

Make sure your server is running:

```bash
python main.py
```

The server should be accessible at `http://localhost:8000`

### Step 3: Run the Test

#### Option A: Interactive Test (Recommended)

```bash
python test_full_face_verification.py
```

Choose option `1` for interactive mode, then:
1. Enter the path to your ID card image
2. Enter the path to your verification image
3. Specify if they should match (same person or not)

#### Option B: Simple Test

```bash
python test_two_images_example.py
```

Follow the prompts to enter your image paths.

#### Option C: Modify the Script

Edit `test_two_images_example.py` and change these lines:

```python
# Replace these paths with your images
id_card_image = "path/to/your/id_card.jpg"
verification_image = "path/to/your/verification_image.jpg"

test_two_images(id_card_image, verification_image)
```

## Understanding the Results

### Similarity Score
- **0.9 - 1.0**: Very high confidence match
- **0.7 - 0.9**: Good match (above threshold)
- **0.5 - 0.7**: Below threshold but similar
- **0.0 - 0.5**: Low similarity, likely different people

### Threshold
- Default threshold: **0.7**
- Scores above 0.7 = MATCH
- Scores below 0.7 = NO MATCH

### Example Output

```
✅ Verification completed!
   Result: ✅ MATCH
   Similarity Score: 0.8542
   Threshold: 0.7
   📊 High confidence - likely the same person
```

## Sample Test Scenarios

### Scenario 1: Same Person (Positive Test)
- Use the same image for both ID card and verification
- Expected result: MATCH with high similarity score (>0.9)

### Scenario 2: Different People (Negative Test)
- Use images of two different people
- Expected result: NO MATCH with low similarity score (<0.5)

### Scenario 3: Same Person, Different Photos
- Use an ID card photo and a different photo of the same person
- Expected result: MATCH with good similarity score (>0.7)

## Available Sample Images

The system includes sample images in the `sample_images/` directory:

- `front_iqama-1-front.jpg` - Qatar residency permit (front)
- `back_iqama-1-back.jpg` - Qatar residency permit (back)
- `id_card.jpg` - Extracted face from ID card
- `test_face.jpg` - Another face image for testing

## Troubleshooting

### Common Issues

1. **"No face detected"**
   - Ensure the face is clearly visible
   - Check image quality and lighting
   - Try a different image

2. **"Person not found"**
   - Make sure the ID card was processed successfully first
   - Check the person_id in the response

3. **Low similarity scores for same person**
   - Check image quality
   - Ensure good lighting in both images
   - Avoid heavily compressed images

4. **Server not running**
   - Start the server: `python main.py`
   - Check if port 8000 is available

### Tips for Better Results

- **Use clear, high-quality images**
- **Ensure good lighting** in both photos
- **Avoid shadows** on the face
- **Keep similar angles** when possible
- **Use uncompressed images** when available

## API Endpoints Used

The test scripts use these API endpoints:

1. `POST /api/process-document-with-face` - Register person from ID card
2. `POST /api/verify-person` - Verify person with test image
3. `GET /api/person/{person_id}` - Get person information

## Technical Details

- **Face Detection**: YOLOv11n-face-detection model
- **Feature Extraction**: ResNet-50 (512-dimensional embeddings)
- **Similarity Metric**: Cosine similarity
- **Database**: MongoDB with fallback to local files
- **Threshold**: 0.7 (configurable)

## Example Usage

```python
# Simple example
from test_two_images_example import test_two_images

# Test with your images
test_two_images("my_id_card.jpg", "my_photo.jpg")
```

For more advanced usage, see the source code of the test scripts. 