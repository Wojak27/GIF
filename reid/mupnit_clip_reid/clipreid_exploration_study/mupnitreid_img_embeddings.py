# generate_embeddings.py
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import argparse
from tqdm import tqdm

def generate_embeddings(input_dir, output_dir, device='cuda'):
    os.makedirs(output_dir, exist_ok=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Process images in the root of input_dir if they are files
    for file in tqdm(os.listdir(input_dir), desc="Processing files in " + input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isdir(file_path):
            continue  # or process separately if desired
        if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        image = Image.open(file_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model.get_image_features(**inputs).cpu().squeeze(0)
        # Save with .pt extension regardless of original extension
        output_file = os.path.join(output_dir, os.path.splitext(file)[0] + '.pt')
        torch.save(embedding, output_file)


train_dir = '' # reid train/
test_dir = '' # reid test/
query_dir = '' # reid query/
gallery_train_dir = '' # reid gallery_train/
gallery_test_dir = '' # reid gallery_test/

output_dir = '' # output image_embeddings/

dirs = [train_dir, test_dir, query_dir, gallery_train_dir, gallery_test_dir]
os.makedirs(output_dir, exist_ok=True)
for d in dirs:
    generate_embeddings(d, os.path.join(output_dir,d.split("/")[-1]), "cuda")