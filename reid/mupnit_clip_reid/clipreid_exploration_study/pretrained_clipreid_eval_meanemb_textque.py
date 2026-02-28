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
    For a query image, fuse its image embedding with its text embedding (if available).
    If no text embedding is available, return the plain image embedding.
    """
    img_emb = get_image_embedding_pretrained(img_path)
    pid = parse_player_id(img_path)
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

def parse_player_id(filepath):
    """Extract player ID from the filename (assumed to be the first field)."""
    return os.path.basename(filepath).split("_")[0]

# --- Aggregation functions ---
def compute_mean_embeddings(file_list, embedding_func):
    """
    Group images by player id and compute the mean embedding for each player.
    Embeddings are normalized after averaging.
    """
    groups = {}
    for f in file_list:
        pid = parse_player_id(f)
        groups.setdefault(pid, []).append(f)
    mean_embeddings = {}
    for pid, files in groups.items():
        embeddings = []
        for f in files:
            emb = embedding_func(f)
            embeddings.append(emb)
        mean_emb = np.mean(embeddings, axis=0)
        # Normalize the mean embedding
        norm = np.linalg.norm(mean_emb) + 1e-8
        mean_embeddings[pid] = mean_emb / norm
    return mean_embeddings

# --- Metric computation ---
def compute_metrics(query_mean_embs, gallery_mean_embs):
    """
    Compute mAP and CMC metrics using the mean embeddings.
    query_mean_embs and gallery_mean_embs are dictionaries mapping player id to embedding.
    """
    query_ids = list(query_mean_embs.keys())
    query_embs = np.stack([query_mean_embs[pid] for pid in query_ids])
    gallery_ids = list(gallery_mean_embs.keys())
    gallery_embs = np.stack([gallery_mean_embs[pid] for pid in gallery_ids])
    
    num_queries = len(query_ids)
    ap_list = []
    cmc_acc = np.zeros(len(gallery_ids))
    for i in range(num_queries):
        q_emb = query_embs[i]
        sims = np.dot(gallery_embs, q_emb)
        sorted_idx = np.argsort(-sims)
        sorted_gallery_ids = np.array(gallery_ids)[sorted_idx]
        # A correct match is when the gallery player id equals the query player id.
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
    cmc_avg = cmc_acc / num_queries if num_queries > 0 else None
    return mAP, cmc_avg

# --- Evaluation using mean embeddings ---
def evaluate_mean_embeddings(query_emb_func, gallery_emb_func):
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    query_mean_embs = compute_mean_embeddings(query_files, query_emb_func)
    gallery_mean_embs = compute_mean_embeddings(gallery_files, gallery_emb_func)
    return compute_metrics(query_mean_embs, gallery_mean_embs)

# --- Run evaluations ---
# Phase 1: Mean image–image matching (both query and gallery use plain image embeddings)
mAP_phase1, cmc_phase1 = evaluate_mean_embeddings(get_image_embedding_pretrained, get_image_embedding_pretrained)
# Phase 2: Mean fusion matching (query uses fused image+text embedding, gallery uses plain image embedding)
mAP_phase2, cmc_phase2 = evaluate_mean_embeddings(get_fused_query_embedding_pretrained, get_image_embedding_pretrained)

with open(LOG_FILE, "w") as f:
    f.write("Pretrained CLIP ReID Evaluation (3A - Mean Embeddings at Player-level)\n")
    f.write("Phase 1 (Mean Image-Image Matching):\n")
    f.write(f"mAP: {mAP_phase1:.4f}\n")
    f.write(f"Rank-1: {cmc_phase1[0]:.4f}, Rank-5: {cmc_phase1[4] if len(cmc_phase1) > 4 else 0:.4f}, "
            f"Rank-10: {cmc_phase1[9] if len(cmc_phase1) > 9 else 0:.4f}, Rank-20: {cmc_phase1[19] if len(cmc_phase1) > 19 else 0:.4f}\n\n")
    f.write("Phase 2 (Mean Fused Query [Image+Text] vs. Mean Image-Only Gallery):\n")
    f.write(f"mAP: {mAP_phase2:.4f}\n")
    f.write(f"Rank-1: {cmc_phase2[0]:.4f}, Rank-5: {cmc_phase2[4] if len(cmc_phase2) > 4 else 0:.4f}, "
            f"Rank-10: {cmc_phase2[9] if len(cmc_phase2) > 9 else 0:.4f}, Rank-20: {cmc_phase2[19] if len(cmc_phase2) > 19 else 0:.4f}\n")
print(f"Pretrained CLIP ReID (3A - Mean Embeddings) results logged to {LOG_FILE}")
