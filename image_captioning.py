import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import PeftModel
from settings import BLIP_FINETUNED_PATH, DEVICE


PEFT_MODEL_PATH = BLIP_FINETUNED_PATH


# Load the base BLIP model first
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
base_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load YOUR fine-tuned adapters on top
model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)
model.to(DEVICE)
model.eval()


def generate_caption(image_path):
    try:
        # Load and convert image
        image = Image.open(image_path).convert('RGB')

        # Preprocess
        inputs = processor(images=image, return_tensors="pt").to(DEVICE)

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100)

        # Decode
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    except FileNotFoundError:
        return "Error: Could not find the image file. Check the path!"
    except Exception as e:
        return f"Error: {e}"
