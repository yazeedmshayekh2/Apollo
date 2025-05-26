from pathlib import Path
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from .qwen_vl_utils import process_vision_info, create_document_message

class DocumentDetector:
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = None
        self.model = None
        self.tokenizer = None
        
        # Document type detection keywords
        self.keywords = {
            "residency": ["state of qatar residency permit", "residency permit", "qatar residency", "إقامة", "رخصة إقامة"],
            "vehicle": ["vehicle registration", "vehicle information", "vehicle reg", "تسيير مركبة", "بيانات المركبة"]
        }
        
    def load_model(self):
        """Load the model only when needed"""
        if self.model is None:
            print("Loading Qwen2.5-VL model...")
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float16
            )
            print("Model loaded successfully")
        
    def detect_document_type(self, image_path):
        """Detect document type from image using Qwen2.5-VL model"""
        self.load_model()
        
        # Detection prompt
        detection_prompt = "What type of document is this? Is it a Qatar Residency Permit/Card (Iqama) or a Vehicle Registration/License? Also, is this the front or back side of the document? Please just reply with the document type (Residency or Vehicle) and the side (Front or Back)."
        
        # Create message format for Qwen2.5-VL
        messages = create_document_message(image_path, detection_prompt)
        
        # Process message for model input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=100)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # Parse response to determine document type and side
        doc_type = "unknown"
        side = "unknown"
        
        output = output.lower()
        if "residency" in output or "permit" in output or "iqama" in output:
            doc_type = "residency"
        elif "vehicle" in output or "registration" in output or "license" in output:
            doc_type = "vehicle"
            
        if "front" in output:
            side = "front"
        elif "back" in output:
            side = "back"
        
        return doc_type, side

# Create a singleton instance
detector = DocumentDetector()

def detect_document_type(image_path):
    """
    Detect document type from image path
    Returns: (document_type, side)
    where document_type is one of ["residency", "vehicle", "unknown"]
    and side is one of ["front", "back", "unknown"]
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    return detector.detect_document_type(str(image_path)) 