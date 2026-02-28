#!/usr/bin/env python
import os, glob, torch, numpy as np
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
QUERY_DIR = "" # reid query dir
GALLERY_DIR = "" # reid gallery_test dir
TEXT_EMBED_DIR = "" # text embedding test dir
LOG_FILE = "" # log file to save results

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()

# --- Embedding functions ---
def get_image_embedding_pretrained(img_path):
    """Return image-only embedding using pretrained CLIP."""
    image = Image.open(img_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(image_input)
    emb = emb.float()
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy().squeeze()

def get_fused_query_embedding_pretrained(img_path):
    """
    For a query image: fuse the image embedding with its text embedding (if available).
    Gallery images remain as image-only.
    """
    img_emb = get_image_embedding_pretrained(img_path)
    pid = os.path.basename(img_path).split("_")[0]
    pattern = os.path.join(TEXT_EMBED_DIR, f"{pid}_*.pt")
    text_files = glob.glob(pattern)
    if text_files:
        text_emb = torch.load(text_files[0]).float().numpy().squeeze()
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        fused = (img_emb + text_emb) / 2.0
        fused = fused / (np.linalg.norm(fused) + 1e-8)
        return fused
    else:
        return img_emb

def parse_filename(filename):
    return os.path.basename(filename).split("_")[0]

def compute_metrics(query_embeddings, query_ids, gallery_embeddings, gallery_ids):
    num_queries = len(query_embeddings)
    ap_list = []
    cmc_acc = np.zeros(len(gallery_ids))
    for i in range(num_queries):
        q_emb = query_embeddings[i]
        sims = np.dot(gallery_embeddings, q_emb)
        sorted_idx = np.argsort(-sims)
        sorted_gallery_ids = np.array(gallery_ids)[sorted_idx]
        matches = (sorted_gallery_ids == query_ids[i]).astype(np.int32)
        if matches.sum() == 0:
            continue
        first_hit = np.where(matches == 1)[0][0]
        cmc = np.zeros(len(matches))
        cmc[first_hit:] = 1
        cmc_acc += cmc
        num_rel = matches.sum()
        ap = 0.0
        hit_count = 0
        for j, flag in enumerate(matches):
            if flag:
                hit_count += 1
                ap += hit_count / (j + 1)
        ap /= num_rel
        ap_list.append(ap)
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc / num_queries
    return mAP, cmc_avg

def evaluate_set(query_embedding_func, gallery_embedding_func):
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    query_embeddings = []
    query_ids = []
    for f in tqdm(query_files, desc="Processing query images"):
        query_embeddings.append(query_embedding_func(f))
        query_ids.append(parse_filename(f))
    gallery_embeddings = []
    gallery_ids = []
    for f in tqdm(gallery_files, desc="Processing gallery images"):
        gallery_embeddings.append(gallery_embedding_func(f))
        gallery_ids.append(parse_filename(f))
    query_embeddings = np.stack(query_embeddings)
    gallery_embeddings = np.stack(gallery_embeddings)
    return compute_metrics(query_embeddings, query_ids, gallery_embeddings, gallery_ids)

# --- Run evaluations ---
# Phase 1: Image-to-image matching (both query and gallery use image-only embeddings)
mAP_std, cmc_std = evaluate_set(get_image_embedding_pretrained, get_image_embedding_pretrained)
# Phase 2: Fused query (image+text) vs image-only gallery
mAP_fusion, cmc_fusion = evaluate_set(get_fused_query_embedding_pretrained, get_image_embedding_pretrained)

with open(LOG_FILE, "w") as f:
    f.write("Pretrained CLIP ReID Evaluation Results (New)\n")
    f.write("Phase 1 (Image-Image Matching):\n")
    f.write(f"mAP: {mAP_std:.4f}\n")
    f.write(f"Rank-1: {cmc_std[0]:.4f}, Rank-5: {cmc_std[4] if len(cmc_std) > 4 else 0:.4f}, "
            f"Rank-10: {cmc_std[9] if len(cmc_std) > 9 else 0:.4f}, Rank-20: {cmc_std[19] if len(cmc_std) > 19 else 0:.4f}\n\n")
    f.write("Phase 2 (Fused Query [Image+Text] vs Image-Only Gallery):\n")
    f.write(f"mAP: {mAP_fusion:.4f}\n")
    f.write(f"Rank-1: {cmc_fusion[0]:.4f}, Rank-5: {cmc_fusion[4] if len(cmc_fusion) > 4 else 0:.4f}, "
            f"Rank-10: {cmc_fusion[9] if len(cmc_fusion) > 9 else 0:.4f}, Rank-20: {cmc_fusion[19] if len(cmc_fusion) > 19 else 0:.4f}\n")
print(f"Pretrained CLIP ReID new results logged to {LOG_FILE}")
