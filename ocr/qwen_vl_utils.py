import os
import base64
from io import BytesIO
from PIL import Image

def process_vision_info(messages):
    """
    Extract images from messages
    Based on Qwen2.5-VL processing code
    """
    image_inputs = []

    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    image_info = content["image"]
                    
                    # Handle image paths, URLs, or base64
                    if isinstance(image_info, str):
                        if os.path.exists(image_info):  # Local file path
                            image_inputs.append(Image.open(image_info))
                        elif image_info.startswith("http"):  # URL
                            # URL handling would require additional HTTP client
                            raise NotImplementedError("URL image handling not implemented")
                        elif image_info.startswith("data:image"):  # base64
                            # Extract the base64 part
                            image_data = image_info.split(",")[1]
                            image = Image.open(BytesIO(base64.b64decode(image_data)))
                            image_inputs.append(image)
                    elif isinstance(image_info, Image.Image):
                        image_inputs.append(image_info)
    
    return image_inputs

def create_document_message(image_path, prompt_text):
    """
    Create a message with an image and prompt text
    For Qwen2.5-VL model input format
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    
    return messages 