#!/usr/bin/env python
import os, glob, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
QUERY_DIR = "" # reid query dir
GALLERY_DIR = "" # reid gallery_test dir
TEXT_EMBED_DIR = "" # text embedding test dir
LOG_FILE = "" # log file to save results
PROJ_WEIGHT_PATH = "" # path to the trained projection model weights

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()

# --- Define the projection network (same as training) ---
class ProjectionModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
    def forward(self, x):
        return self.net(x)

proj_model = ProjectionModel().to(device)
proj_model.load_state_dict(torch.load(PROJ_WEIGHT_PATH, map_location=device))
proj_model.eval()

# --- Embedding functions ---
def get_image_embedding_trained(img_path):
    """Return the image embedding using CLIP and then project it."""
    image = Image.open(img_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        base_emb = clip_model.encode_image(image_input)
    base_emb = base_emb.float()
    base_emb = F.normalize(base_emb, p=2, dim=-1)
    with torch.no_grad():
        proj_emb = proj_model(base_emb)
    proj_emb = F.normalize(proj_emb, p=2, dim=-1)
    return proj_emb.cpu().numpy().squeeze()

def get_fused_query_embedding_trained(img_path):
    """
    For a query image, fuse its trained image embedding with its text embedding (if available).
    If no text is available, return the plain projected image embedding.
    """
    img_emb = get_image_embedding_trained(img_path)
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
    """Extract player ID from the filename."""
    return os.path.basename(filepath).split("_")[0]

# --- Aggregation functions (same as in 3A) ---
def compute_mean_embeddings(file_list, embedding_func):
    """
    Group images by player id and compute the mean embedding for each player.
    The resulting mean embedding is normalized.
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
        norm = np.linalg.norm(mean_emb) + 1e-8
        mean_embeddings[pid] = mean_emb / norm
    return mean_embeddings

# --- Metric computation (same as in 3A) ---
def compute_metrics(query_mean_embs, gallery_mean_embs):
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

def evaluate_mean_embeddings(query_emb_func, gallery_emb_func):
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    query_mean_embs = compute_mean_embeddings(query_files, query_emb_func)
    gallery_mean_embs = compute_mean_embeddings(gallery_files, gallery_emb_func)
    return compute_metrics(query_mean_embs, gallery_mean_embs)

# --- Run evaluations ---
# Phase 1: Mean image–image matching (both query and gallery use plain trained image embeddings)
mAP_phase1, cmc_phase1 = evaluate_mean_embeddings(get_image_embedding_trained, get_image_embedding_trained)
# Phase 2: Mean fusion matching (query uses fused image+text embedding, gallery uses plain trained image embeddings)
mAP_phase2, cmc_phase2 = evaluate_mean_embeddings(get_fused_query_embedding_trained, get_image_embedding_trained)

with open(LOG_FILE, "w") as f:
    f.write("Trained Projection CLIP ReID Evaluation (3B - Mean Embeddings at Player-level)\n")
    f.write("Phase 1 (Mean Image-Image Matching):\n")
    f.write(f"mAP: {mAP_phase1:.4f}\n")
    f.write(f"Rank-1: {cmc_phase1[0]:.4f}, Rank-5: {cmc_phase1[4] if len(cmc_phase1) > 4 else 0:.4f}, "
            f"Rank-10: {cmc_phase1[9] if len(cmc_phase1) > 9 else 0:.4f}, Rank-20: {cmc_phase1[19] if len(cmc_phase1) > 19 else 0:.4f}\n\n")
    f.write("Phase 2 (Mean Fused Query [Image+Text] vs. Mean Image-Only Gallery):\n")
    f.write(f"mAP: {mAP_phase2:.4f}\n")
    f.write(f"Rank-1: {cmc_phase2[0]:.4f}, Rank-5: {cmc_phase2[4] if len(cmc_phase2) > 4 else 0:.4f}, "
            f"Rank-10: {cmc_phase2[9] if len(cmc_phase2) > 9 else 0:.4f}, Rank-20: {cmc_phase2[19] if len(cmc_phase2) > 19 else 0:.4f}\n")
print(f"Trained Projection CLIP ReID (3B - Mean Embeddings) results logged to {LOG_FILE}")
