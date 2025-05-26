# Qatar Document OCR

A Python application for extracting information from Qatar residency permits (iqama) and vehicle registration cards using the Qwen2.5-VL-7B-Instruct-AWQ vision-language model.

## Features

- Automatic detection of document type (residency permit or vehicle registration)
- Process both front and back sides of documents simultaneously
- Intelligent validation and correction for incorrectly ordered/mismatched documents
- Unified document processing that combines information from both sides
- Specialized prompts for each document type and side
- Structured data output in both JSON and CSV formats
- User-friendly web interface with document preview

## Important Notes

- This project uses the quantized AWQ version of Qwen2.5-VL which requires specific package versions
- We use `torch.float16` for consistent dtype to avoid precision mismatches
- The model requires approximately 8GB of VRAM to run efficiently
- First-time loading will download the model weights (~5GB)

## Prerequisites

- Python 3.8+
- PyTorch
- CUDA-compatible GPU (recommended for faster processing)

## Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/qatar-document-ocr.git
cd qatar-document-ocr
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the application:

```bash
python main.py
```

2. Open your browser and navigate to `http://localhost:8000`.

3. Upload both front and back images of your document (Qatar residency permit or vehicle registration card).

4. Choose whether to auto-detect the document type or manually select it.

5. Submit the form to process both sides of the document together.

6. If the images are incorrectly ordered (front/back swapped), the system will detect this and offer to correct it.

7. View the combined, structured results and download them in JSON or CSV format.

### Command Line Testing

You can also test document processing from the command line:

```bash
python test_model.py path/to/document_image.jpg
```

## API Endpoints

The application provides the following API endpoints:

- `POST /api/detect-document`: Detect the document type and side from an uploaded image.
- `POST /api/process-document`: Process a single document side and extract information.
- `POST /api/process-document-both-sides`: Process both front and back sides of a document in a single request and return combined, structured data.

## Document Types Supported

1. Qatar Residency Permit (Iqama)
   - Front side: Extracts document info, ID number, full name (in English and Arabic), date of birth, nationality, occupation, and expiry date.
   - Back side: Extracts passport details, residency type, employer information, and other additional data.
   - Combined: Creates a comprehensive personal record with all extracted information organized into logical sections.

2. Vehicle Registration
   - Front side: Extracts plate number, owner details, registration dates, and vehicle identification.
   - Back side: Extracts vehicle specifications (make, model, year, etc.), technical details (chassis/engine numbers), and insurance information.
   - Combined: Creates a complete vehicle record with owner, registration, vehicle, and insurance information in a structured format.

## License

MIT 