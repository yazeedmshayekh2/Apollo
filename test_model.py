import os
import sys
import argparse
from ocr import detect_document_type, process_document

def main():
    parser = argparse.ArgumentParser(description='Test Qatar Document OCR model')
    parser.add_argument('image_path', help='Path to the document image')
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        sys.exit(1)
    
    # First detect document type
    print("Detecting document type...")
    doc_type, side = detect_document_type(args.image_path)
    print(f"Detected document type: {doc_type}, side: {side}")
    
    if doc_type == "unknown" or side == "unknown":
        print("Could not determine document type or side. Please provide a clearer image.")
        sys.exit(1)
        
    # Process the document
    print("\nProcessing document...")
    result = process_document(args.image_path, doc_type, side)
    
    # Print result
    print("\nExtracted information:")
    for key, value in result.items():
        print(f"{key}: {value}")
    
if __name__ == "__main__":
    main() 