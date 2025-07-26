# backend/clip_utils.py
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

def image_to_clip_embedding(image_file):
    image = Image.open(image_file).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    return outputs.cpu().numpy()[0]
