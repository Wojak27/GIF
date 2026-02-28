#!/usr/bin/env python
import os, glob, torch, numpy as np
from tqdm import tqdm
import torch.nn as nn, torch.nn.functional as F
import clip
from PIL import Image
import logging

# ---------------- Logging Setup ----------------
LOG_DIR = "" # directory to save logs
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, "clip_reid_evaluation.log")

# Set up logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# ---------------- Configuration ----------------
QUERY_DIR = "" # reid query dir
GALLERY_DIR = "" # reid gallery_test dir
TEXT_EMBED_DIR = "" # text embedding test dir
PROJ_WEIGHT_PATH = "" # path to the trained projection model weights

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Load CLIP and Projection Model ----------------
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()

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

# ---------------- Helper Functions ----------------

def parse_filename(filepath):
    # Assumes the player's identity is the first part of the filename (separated by '_')
    return os.path.basename(filepath).split("_")[0]

def get_image_embedding_trained(img_path):
    """
    Returns the projected (trained) image embedding.
    """
    image = Image.open(img_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        base_emb = clip_model.encode_image(image_input)
    base_emb = F.normalize(base_emb.float(), p=2, dim=-1)
    with torch.no_grad():
        proj_emb = proj_model(base_emb)
    proj_emb = F.normalize(proj_emb, p=2, dim=-1)
    return proj_emb.cpu().numpy().squeeze()

def get_image_raw_embedding(img_path):
    """
    Returns the plain CLIP image embedding (before projection).
    """
    image = Image.open(img_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        base_emb = clip_model.encode_image(image_input)
    base_emb = F.normalize(base_emb.float(), p=2, dim=-1)
    return base_emb.cpu().numpy().squeeze()

def get_query_text_embedding_trained(query_img_path):
    """
    Loads the plain text embedding for a query image.
    (Assumes the saved text embeddings are plain CLIP embeddings.)
    """
    pid = os.path.basename(query_img_path).split("_")[0]
    pattern = os.path.join(TEXT_EMBED_DIR, f"{pid}_*.pt")
    text_files = glob.glob(pattern)
    if text_files:
        text_emb = torch.load(text_files[0]).float().numpy().squeeze()
        text_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        return text_emb
    else:
        return None

def compute_metrics_from_similarity(similarity_vector, query_id, gallery_ids):
    """
    Computes AP and CMC given a pre-computed similarity vector.
    """
    sorted_idx = np.argsort(-similarity_vector)
    sorted_gallery_ids = np.array(gallery_ids)[sorted_idx]
    matches = (sorted_gallery_ids == query_id).astype(np.int32)
    if matches.sum() == 0:
        return None, None
    first_hit = np.where(matches == 1)[0][0]
    cmc = np.zeros(len(matches))
    cmc[first_hit:] = 1
    num_rel = matches.sum()
    ap = 0.0
    hit_count = 0
    for j, flag in enumerate(matches):
        if flag:
            hit_count += 1
            ap += hit_count / (j + 1)
    ap /= num_rel
    return ap, cmc

# ---------------- Evaluation: Late Fusion with Varying Alpha ----------------

def evaluate_late_fusion():
    # Load gallery images
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    gallery_ids = [parse_filename(f) for f in gallery_files]
    gallery_proj_embeddings = []
    gallery_raw_embeddings = []
    for f in tqdm(gallery_files, desc="Processing gallery images"):
        gallery_proj_embeddings.append(get_image_embedding_trained(f))
        gallery_raw_embeddings.append(get_image_raw_embedding(f))
    gallery_proj_embeddings = np.stack(gallery_proj_embeddings)
    gallery_raw_embeddings = np.stack(gallery_raw_embeddings)
    
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    
    best_mAP = 0.0
    best_alpha = None
    best_cmc = None
    results = []

    # Try different values of alpha from 0 to 1 (inclusive) in steps of 0.1
    for alpha in np.linspace(0, 1, 20):
        ap_list = []
        cmc_acc = None
        num_valid_queries = 0

        for qf in tqdm(query_files, desc=f"Processing queries with alpha = {alpha:.1f}", leave=False):
            query_id = parse_filename(qf)
            
            # Compute image-image similarity using projected embeddings.
            query_img_emb = get_image_embedding_trained(qf)
            sim_image = np.dot(gallery_proj_embeddings, query_img_emb)
            
            # Compute text-image similarity using plain embeddings:
            query_text_emb = get_query_text_embedding_trained(qf)
            if query_text_emb is not None:
                sim_text = np.dot(gallery_raw_embeddings, query_text_emb)
                # Late fusion of similarities
                fusion_sim = alpha * sim_image + (1 - alpha) * sim_text
            else:
                fusion_sim = sim_image  # fallback if no text available
            
            result = compute_metrics_from_similarity(fusion_sim, query_id, gallery_ids)
            if result[0] is None:
                continue
            ap, cmc = result
            ap_list.append(ap)
            if cmc_acc is None:
                cmc_acc = cmc
            else:
                cmc_acc += cmc
            num_valid_queries += 1

        mAP = np.mean(ap_list) if ap_list else 0.0
        cmc_avg = cmc_acc / num_valid_queries if num_valid_queries > 0 else None
        
        results.append((alpha, mAP, cmc_avg))
        logging.info(f"Alpha {alpha:.1f} -- mAP: {mAP:.4f}")
        if cmc_avg is not None:
            rank1 = cmc_avg[0] if len(cmc_avg) > 0 else 0
            rank5 = cmc_avg[4] if len(cmc_avg) > 4 else 0
            rank10 = cmc_avg[9] if len(cmc_avg) > 9 else 0
            rank20 = cmc_avg[19] if len(cmc_avg) > 19 else 0
            logging.info(f"           Rank-1: {rank1:.4f}, Rank-5: {rank5:.4f}, Rank-10: {rank10:.4f}, Rank-20: {rank20:.4f}")
        
        if mAP > best_mAP:
            best_mAP = mAP
            best_alpha = alpha
            best_cmc = cmc_avg

    logging.info("\nOptimal Late Fusion ReID Evaluation Results:")
    logging.info(f"Best Alpha: {best_alpha:.1f}")
    logging.info(f"mAP: {best_mAP:.4f}")
    if best_cmc is not None:
        rank1 = best_cmc[0] if len(best_cmc) > 0 else 0
        rank5 = best_cmc[4] if len(best_cmc) > 4 else 0
        rank10 = best_cmc[9] if len(best_cmc) > 9 else 0
        rank20 = best_cmc[19] if len(best_cmc) > 19 else 0
        logging.info(f"Rank-1: {rank1:.4f}, Rank-5: {rank5:.4f}, Rank-10: {rank10:.4f}, Rank-20: {rank20:.4f}")
    else:
        logging.info("No valid queries processed.")

if __name__ == "__main__":
    evaluate_late_fusion()

