import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import json
from .qwen_vl_utils import process_vision_info, create_document_message

class DocumentProcessor:
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.processor = None
        self.model = None
        self.tokenizer = None
        
        # Specialized prompts for each document type and side
        self.prompts = {
            "residency": {
                "front": """
            Extract all information from this ID card or residency permit front side and return it as JSON.
            Output must be valid JSON only, no explanations.
            Output must be in the following format:
            {
              "document_info": {
                "document_type": "The type of document (e.g., 'Qatar Residency Permit', 'ID Card')",
                "issuing_authority": "Authority that issued the document (e.g., 'State of Qatar')"
              },
              "personal_info": {
                "id_number": "The ID or permit number (e.g., '28140001175')",
                "name": "Full name of the person as shown on the ID",
                "arabic_name": "Name in Arabic if present",
                "nationality": "Nationality of the ID holder (e.g., 'JORDAN', 'PALESTINE')",
                "date_of_birth": "Birth date in the format shown on the ID (e.g., '07/04/1981')",
                "expiry_date": "Expiration date of the ID (e.g., '07/07/2027')",
                "occupation": "Job title or occupation of the person"
              }
            }
            
            Make sure to preserve all original formatting of dates and numbers.
            Include all Arabic text exactly as shown, especially names.
            """,
                
                "back": """
                Extract all information from this ID card or residency permit back side and return it as JSON.
            Output must be valid JSON only, no explanations.
            Output must be in the following format:
            {
              "document_info": {
                "document_type": "The type of document (e.g., 'Qatar Residency Permit Back')",
                "issuing_authority": "Authority that issued the document if shown"
              },
              "additional_info": {
                "passport_number": "Passport number if shown",
                "passport_expiry": "Passport expiration date if shown",
                "residency_type": "Type of residency permit (e.g., 'Work', 'Family')",
                "employer": "Employer name if shown",
                "serial_number": "Serial number of the document if shown",
                "other_fields": {
                  "field_name": "field_value"
                }
              }
            }
            
            Make sure to preserve all original formatting of dates and numbers.
            Include all Arabic text exactly as shown.
            If any field is not visible or not present, omit it from the JSON.
            """
            },
            "vehicle": {
                "front": """
                Extract all shown information from this vehicle registration card front page and return them as JSON.
                Output must be valid JSON only, no explanations.
                Output must be in the following format:
                {
                    "document_info": {
                        "document_type": "Vehicle Registration Card",
                        "issuing_authority": "Authority that issued the document (e.g., 'State of Qatar', 'Ministry of Interior')"
                    },
                    "owner_info": {
                        "name": "Full name of the vehicle owner",
                        "name_arabic": "Owner name in Arabic if present",
                        "id": "Owner ID number",
                        "nationality": "Nationality of the owner"
                    },
                    "registration_info": {
                        "plate_number": "The vehicle registration/plate number",
                        "plate_type": "Type of license plate (e.g., Private/خصوصي)",
                        "registration_date": "Date when the registration was issued",
                        "expiry_date": "Date when the registration expires",
                        "renew_date": "Date when the registration needs to be renewed"
                    }
                }
                
                Make sure to preserve all original formatting of dates and numbers.
                Include all Arabic text exactly as shown, especially names.
                If any field is not visible or not present, omit it from the JSON.
                """,
                
                "back": """
                Extract all shown information from this vehicle registration card back page and return them as JSON.
                Output must be valid JSON only, no explanations.
                Output must be in the following format:
                {
                    "document_info": {
                        "document_type": "Vehicle Registration Card Back"
                    },
                    "vehicle_info": {
                        "make": "The vehicle manufacturer (e.g., Nissan, Toyota)",
                        "model": "The specific model of the vehicle",
                        "body_type": "The type or category of vehicle (e.g., Station Wagon/ستيشن واجن)",
                        "year_of_manufacture": "Year the vehicle was manufactured",
                        "country_of_manufacture": "Country where the vehicle was manufactured",
                        "cylinders": "Number of cylinders in the engine",
                        "seats": "Number of seats in the vehicle",
                        "weight": "Weight of the vehicle",
                        "color": "Color of the vehicle",
                        "chassis_number": "Vehicle identification/chassis number",
                        "engine_number": "Engine identification number"
                    },
                    "insurance_info": {
                        "insurance_company": "Name of the insurance company",
                        "policy_number": "Insurance policy number",
                        "expiry_date": "Date when the insurance expires"
                    }
                }
                
                Make sure to preserve all original formatting of dates and numbers.
                Include all Arabic text exactly as shown.
                If any field is not visible or not present, omit it from the JSON.
                """
            }
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
    
    def process_document(self, image_path, document_type, side):
        """Process document with specialized prompt based on type and side"""
        self.load_model()
        
        # Select appropriate prompt
        if document_type in self.prompts and side in self.prompts[document_type]:
            prompt = self.prompts[document_type][side]
        else:
            # Fallback to generic prompt
            prompt = """
            Extract all information from this document and return it as valid JSON.
            Be precise and extract all visible text, numbers, dates, and identifiers.
            Format the output as a structured JSON with appropriate field names.
            Include document type, issuing details, and all personal or vehicle information visible.
            Make sure to preserve all original formatting of dates and numbers.
            Include all Arabic text exactly as shown.
            """
        
        # Create message format for Qwen2.5-VL
        messages = create_document_message(image_path, prompt)
        
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
            generated_ids = self.model.generate(**inputs, max_new_tokens=1000)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # Try to find and extract JSON from the response
        try:
            # Find JSON by looking for opening/closing brackets
            start_idx = output.find('{')
            end_idx = output.rfind('}')
            
            if start_idx >= 0 and end_idx >= start_idx:
                json_str = output[start_idx:end_idx+1]
                result = json.loads(json_str)
            else:
                # If no JSON found, return the entire text as a single field
                result = {"raw_text": output}
                
        except json.JSONDecodeError:
            # If JSON parsing fails, return the entire text
            result = {"raw_text": output}
        
        # Add metadata
        result["document_type"] = document_type
        result["document_side"] = side
        
        return result

    def process_both_sides(self, front_image_path, back_image_path, document_type):
        """Process both sides of a document and merge the results intelligently"""
        # Process each side individually
        front_result = self.process_document(front_image_path, document_type, "front")
        back_result = self.process_document(back_image_path, document_type, "back")
        
        # Merge results based on document type
        if document_type == "residency":
            # For residency documents, create a comprehensive record
            merged_result = {
                "document_info": {},
                "personal_info": {},
                "additional_info": {}
            }
            
            # Extract document_info from front side (preferred) or back side
            if "document_info" in front_result:
                merged_result["document_info"] = front_result["document_info"]
            elif "document_info" in back_result:
                merged_result["document_info"] = back_result["document_info"]
            
            # Extract personal_info primarily from front side
            if "personal_info" in front_result:
                merged_result["personal_info"] = front_result["personal_info"]
            
            # Extract additional_info primarily from back side
            if "additional_info" in back_result:
                merged_result["additional_info"] = back_result["additional_info"]
            
            # Add any other fields that might have been detected
            for result in [front_result, back_result]:
                for key, value in result.items():
                    if key not in ["document_type", "document_side", "document_info", "personal_info", "additional_info"]:
                        merged_result[key] = value
            
        elif document_type == "vehicle":
            # For vehicle documents, create a comprehensive vehicle record
            merged_result = {
                "document_info": {},
                "vehicle_info": {},
                "owner_info": {},
                "registration_info": {},
                "insurance_info": {}
            }
            
            # Extract document_info (use front side as primary)
            if "document_info" in front_result:
                merged_result["document_info"] = front_result["document_info"]
            elif "document_info" in back_result:
                merged_result["document_info"] = back_result["document_info"]
            
            # Extract vehicle_info (primarily from back)
            if "vehicle_info" in back_result:
                merged_result["vehicle_info"] = back_result["vehicle_info"]
            
            # Extract owner_info (primarily from front)
            if "owner_info" in front_result:
                merged_result["owner_info"] = front_result["owner_info"]
            
            # Extract registration_info (primarily from front)
            if "registration_info" in front_result:
                merged_result["registration_info"] = front_result["registration_info"]
            
            # Extract insurance_info (primarily from back)
            if "insurance_info" in back_result:
                merged_result["insurance_info"] = back_result["insurance_info"]
            
            # Add any other fields that might have been detected
            for result in [front_result, back_result]:
                for key, value in result.items():
                    if key not in ["document_type", "document_side", "document_info", "vehicle_info", 
                                  "owner_info", "registration_info", "insurance_info"]:
                        merged_result[key] = value
                        
        else:
            # Generic merging for unknown document types
            merged_result = {
                "front_side": front_result,
                "back_side": back_result
            }
        
        # Add metadata
        merged_result["document_type"] = document_type
        
        return merged_result

# Create singleton processor
processor = DocumentProcessor()

def process_document(image_path, document_type, side):
    """
    Process document with OCR based on type and side
    Returns: Dictionary with extracted information
    """
    return processor.process_document(image_path, document_type, side)

def process_both_sides(front_image_path, back_image_path, document_type):
    """
    Process both sides of a document and return a merged result
    Returns: Dictionary with combined information from both sides
    """
    return processor.process_both_sides(front_image_path, back_image_path, document_type) 