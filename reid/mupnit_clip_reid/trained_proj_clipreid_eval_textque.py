#!/usr/bin/env python
import os, glob, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
QUERY_DIR = ""  # reid query dir
GALLERY_DIR = ""  # reid gallery_test dir
TEXT_EMBED_DIR = ""  # text embedding test dir
LOG_FILE = ""  # log file to save results
PROJ_WEIGHT_PATH = ""  # path to the trained projection model weights

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
    For a query image: fuse the trained image embedding with its text embedding (if available).
    Gallery images remain as image-only (trained).
    """
    img_emb = get_image_embedding_trained(img_path)
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
# Phase 1: Image-to-image matching using the trained projection model
mAP_std, cmc_std = evaluate_set(get_image_embedding_trained, get_image_embedding_trained)
# Phase 2: Fused query (image+text) vs image-only gallery (trained projection for image embeddings)
mAP_fusion, cmc_fusion = evaluate_set(get_fused_query_embedding_trained, get_image_embedding_trained)

with open(LOG_FILE, "w") as f:
    f.write("Trained Projection CLIP ReID Evaluation Results (New)\n")
    f.write("Phase 1 (Image-Image Matching):\n")
    f.write(f"mAP: {mAP_std:.4f}\n")
    f.write(f"Rank-1: {cmc_std[0]:.4f}, Rank-5: {cmc_std[4] if len(cmc_std) > 4 else 0:.4f}, "
            f"Rank-10: {cmc_std[9] if len(cmc_std) > 9 else 0:.4f}, Rank-20: {cmc_std[19] if len(cmc_std) > 19 else 0:.4f}\n\n")
    f.write("Phase 2 (Fused Query [Image+Text] vs Image-Only Gallery):\n")
    f.write(f"mAP: {mAP_fusion:.4f}\n")
    f.write(f"Rank-1: {cmc_fusion[0]:.4f}, Rank-5: {cmc_fusion[4] if len(cmc_fusion) > 4 else 0:.4f}, "
            f"Rank-10: {cmc_fusion[9] if len(cmc_fusion) > 9 else 0:.4f}, Rank-20: {cmc_fusion[19] if len(cmc_fusion) > 19 else 0:.4f}\n")
print(f"Trained projection CLIP ReID new results logged to {LOG_FILE}")
