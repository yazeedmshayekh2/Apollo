import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image

def process_vision_info(messages):
    """Minimal implementation to process image inputs"""
    image_inputs = []
    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    image_info = content["image"]
                    if isinstance(image_info, str):
                        image_inputs.append(Image.open(image_info))
                    elif isinstance(image_info, Image.Image):
                        image_inputs.append(image_info)
    return image_inputs

def main():
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct-AWQ", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
    
    # Replace with your image path
    image_path = "sample_images/test_image.jpg"
    
    try:
        # Try to create a test image if none exists
        try:
            import numpy as np
            test_image = Image.fromarray(np.ones((100, 100, 3), dtype=np.uint8) * 255)
            test_image.save(image_path)
            print(f"Created test image at {image_path}")
        except Exception as e:
            print(f"Could not create test image: {e}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": "Describe what you see in this image."},
                ],
            }
        ]

        # Preparation for inference
        print("Processing input...")
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            print("Using CUDA")
        else:
            print("CUDA not available, using CPU")

        # Generate output
        print("Generating response...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        print("\nOutput:")
        print(output_text[0])
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 