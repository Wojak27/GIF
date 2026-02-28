# 10 phases
#!/usr/bin/env python
import os, glob, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
QUERY_DIR = "" # reid query dir
GALLERY_DIR = "" # reid test dir
TEXT_EMBED_DIR = "" # text embedding test dir
LOG_FILE_MODIFIED = "" # log file to save results
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

# ------------------- Helper Functions -------------------

def get_image_embedding_trained(img_path):
    """
    Returns the projected (trained) image embedding.
    Used in phases 1-3.
    """
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

def get_image_raw_embedding(img_path):
    """
    Returns the raw CLIP image embedding (before projection).
    Used for fusion in phases 2 & 3.
    """
    image = Image.open(img_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        base_emb = clip_model.encode_image(image_input)
    base_emb = base_emb.float()
    base_emb = F.normalize(base_emb, p=2, dim=-1)
    return base_emb.cpu().numpy().squeeze()

def fuse_and_project(raw_emb, text_emb):
    """
    Fuses a raw image embedding with a text embedding (by addition and normalization),
    then projects the result using the projection network.
    """
    fused_raw = raw_emb + text_emb
    fused_raw = fused_raw / (np.linalg.norm(fused_raw) + 1e-8)
    fused_tensor = torch.tensor(fused_raw, device=device).unsqueeze(0).float()
    with torch.no_grad():
        fused_proj = proj_model(fused_tensor)
    fused_proj = F.normalize(fused_proj, p=2, dim=-1)
    return fused_proj.cpu().numpy().squeeze()

def get_query_text_embedding_trained(query_img_path):
    """
    Loads the text embedding for a query image.
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

def parse_filename(filename):
    return os.path.basename(filename).split("_")[0]

def compute_metrics_for_query(query_emb, gallery_embs, query_id, gallery_ids):
    sims = np.dot(gallery_embs, query_emb)
    sorted_idx = np.argsort(-sims)
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

def aggregate_embeddings(emb_list):
    """
    Aggregates a list of embeddings by summing and then L2-normalizing.
    """
    if not emb_list:
        return None
    agg = np.sum(np.stack(emb_list), axis=0)
    norm = np.linalg.norm(agg) + 1e-8
    return agg / norm

# ------------------- Phases 1-3: Per-image Evaluation -------------------

def evaluate_phase1():
    """
    Phase 1: Image-to-image matching (re-ID) using projected embeddings.
    """
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    gallery_ids = [parse_filename(f) for f in gallery_files]
    gallery_embeddings = []
    for f in tqdm(gallery_files, desc="Processing gallery images (Phase 1)"):
        gallery_embeddings.append(get_image_embedding_trained(f))
    gallery_embeddings = np.stack(gallery_embeddings)

    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    ap_list = []
    cmc_acc = None
    num_valid_queries = 0
    for qf in tqdm(query_files, desc="Processing query images (Phase 1)"):
        query_id = parse_filename(qf)
        query_emb = get_image_embedding_trained(qf)
        result = compute_metrics_for_query(query_emb, gallery_embeddings, query_id, gallery_ids)
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
    return mAP, cmc_avg

def evaluate_phase2():
    """
    Phase 2: For each query, if a text embedding is available, fuse the query text embedding
    with every gallery image's raw embedding before projection; otherwise, use the original gallery embeddings.
    Then perform re-ID with query projected embeddings.
    """
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    gallery_ids = [parse_filename(f) for f in gallery_files]
    gallery_raw_embeddings = []
    gallery_proj_embeddings = []
    for f in tqdm(gallery_files, desc="Processing gallery images (Phase 2)"):
        gallery_raw_embeddings.append(get_image_raw_embedding(f))
        gallery_proj_embeddings.append(get_image_embedding_trained(f))
    gallery_raw_embeddings = np.stack(gallery_raw_embeddings)
    gallery_proj_embeddings = np.stack(gallery_proj_embeddings)

    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    ap_list = []
    cmc_acc = None
    num_valid_queries = 0
    for qf in tqdm(query_files, desc="Processing query images (Phase 2)"):
        query_id = parse_filename(qf)
        query_img_emb = get_image_embedding_trained(qf)
        query_text_emb = get_query_text_embedding_trained(qf)
        if query_text_emb is not None:
            fused_gallery = []
            for raw_emb in gallery_raw_embeddings:
                fused_gallery.append(fuse_and_project(raw_emb, query_text_emb))
            fused_gallery = np.stack(fused_gallery)
        else:
            fused_gallery = gallery_proj_embeddings
        result = compute_metrics_for_query(query_img_emb, fused_gallery, query_id, gallery_ids)
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
    return mAP, cmc_avg

def evaluate_phase3():
    """
    Phase 3: For each query, fuse its text embedding with the gallery images of the same player 
    (using raw embeddings before projection) then project to obtain fused embeddings.
    For gallery images of other players, use the original projected embeddings.
    Then perform re-ID with the query's projected embedding.
    """
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    gallery_ids = [parse_filename(f) for f in gallery_files]
    gallery_raw_embeddings = []
    gallery_proj_embeddings = []
    for f in tqdm(gallery_files, desc="Processing gallery images (Phase 3)"):
        gallery_raw_embeddings.append(get_image_raw_embedding(f))
        gallery_proj_embeddings.append(get_image_embedding_trained(f))
    gallery_raw_embeddings = np.stack(gallery_raw_embeddings)
    gallery_proj_embeddings = np.stack(gallery_proj_embeddings)

    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    ap_list = []
    cmc_acc = None
    num_valid_queries = 0
    for qf in tqdm(query_files, desc="Processing query images (Phase 3)"):
        query_id = parse_filename(qf)
        query_img_emb = get_image_embedding_trained(qf)
        query_text_emb = get_query_text_embedding_trained(qf)
        fused_gallery = []
        for i, g_raw in enumerate(gallery_raw_embeddings):
            g_id = gallery_ids[i]
            if (g_id == query_id) and (query_text_emb is not None):
                fused_gallery.append(fuse_and_project(g_raw, query_text_emb))
            else:
                fused_gallery.append(gallery_proj_embeddings[i])
        fused_gallery = np.stack(fused_gallery)
        result = compute_metrics_for_query(query_img_emb, fused_gallery, query_id, gallery_ids)
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
    return mAP, cmc_avg

# ------------------- Phases 4-6: Aggregated (Per-player) Evaluation -------------------

def aggregate_player_embeddings(file_list, embedding_func):
    """
    Groups files by player ID and aggregates embeddings using embedding_func.
    Returns a dict mapping player id to aggregated embedding.
    """
    agg_dict = {}
    for f in file_list:
        pid = parse_filename(f)
        emb = embedding_func(f)
        if pid not in agg_dict:
            agg_dict[pid] = []
        agg_dict[pid].append(emb)
    # Aggregate (sum then normalize) for each player.
    for pid in agg_dict:
        agg_dict[pid] = aggregate_embeddings(agg_dict[pid])
    return agg_dict

def aggregate_player_texts(query_files):
    """
    Groups query files by player ID and aggregates available text embeddings.
    Returns a dict mapping player id to aggregated text embedding (if any).
    """
    text_dict = {}
    for f in query_files:
        pid = parse_filename(f)
        text_emb = get_query_text_embedding_trained(f)
        if text_emb is not None:
            if pid not in text_dict:
                text_dict[pid] = []
            text_dict[pid].append(text_emb)
    for pid in text_dict:
        text_dict[pid] = aggregate_embeddings(text_dict[pid])
    return text_dict

def evaluate_phase4():
    """
    Phase 4: Aggregated query vs. aggregated gallery using projected embeddings.
    For each player, aggregate all query image embeddings (projected) and gallery image embeddings (projected).
    Then perform matching across players.
    """
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    
    query_agg = aggregate_player_embeddings(query_files, get_image_embedding_trained)
    gallery_agg = aggregate_player_embeddings(gallery_files, get_image_embedding_trained)
    
    query_ids = sorted(query_agg.keys())
    gallery_ids = sorted(gallery_agg.keys())
    
    # Build list for gallery evaluation.
    gallery_embs = [gallery_agg[pid] for pid in gallery_ids]
    
    ap_list = []
    cmc_acc = None
    num_valid = 0
    for qid in query_ids:
        query_emb = query_agg[qid]
        # Compute similarity with each gallery player's aggregated embedding.
        sims = np.dot(np.stack(gallery_embs), query_emb)
        sorted_idx = np.argsort(-sims)
        sorted_gallery_ids = np.array(gallery_ids)[sorted_idx]
        matches = (sorted_gallery_ids == qid).astype(np.int32)
        if matches.sum() == 0:
            continue
        first_hit = np.where(matches==1)[0][0]
        cmc = np.zeros(len(matches))
        cmc[first_hit:] = 1
        num_rel = matches.sum()
        ap = 0.0
        hit_count = 0
        for j, flag in enumerate(matches):
            if flag:
                hit_count += 1
                ap += hit_count / (j+1)
        ap /= num_rel
        ap_list.append(ap)
        if cmc_acc is None:
            cmc_acc = cmc
        else:
            cmc_acc += cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc/num_valid if num_valid > 0 else None
    return mAP, cmc_avg

def evaluate_phase5():
    """
    Phase 5: Aggregated query vs. aggregated gallery after fusion.
    For each player, if aggregated query text is available, fuse the aggregated gallery raw embedding
    (aggregated from raw gallery images) with the aggregated query text embedding (then project).
    Otherwise, use the aggregated gallery projected embedding.
    """
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    
    query_agg = aggregate_player_embeddings(query_files, get_image_embedding_trained)
    gallery_proj_agg = aggregate_player_embeddings(gallery_files, get_image_embedding_trained)
    gallery_raw_agg = aggregate_player_embeddings(gallery_files, get_image_raw_embedding)
    query_text_agg = aggregate_player_texts(query_files)
    
    query_ids = sorted(query_agg.keys())
    gallery_ids = sorted(gallery_proj_agg.keys())
    
    # Build gallery fused embeddings for each player.
    fused_gallery = {}
    for pid in gallery_ids:
        if pid in query_text_agg:
            fused_gallery[pid] = fuse_and_project(gallery_raw_agg[pid], query_text_agg[pid])
        else:
            fused_gallery[pid] = gallery_proj_agg[pid]
    
    gallery_list = [fused_gallery[pid] for pid in gallery_ids]
    
    ap_list = []
    cmc_acc = None
    num_valid = 0
    for qid in query_ids:
        query_emb = query_agg[qid]
        sims = np.dot(np.stack(gallery_list), query_emb)
        sorted_idx = np.argsort(-sims)
        sorted_gallery_ids = np.array(gallery_ids)[sorted_idx]
        matches = (sorted_gallery_ids == qid).astype(np.int32)
        if matches.sum() == 0:
            continue
        first_hit = np.where(matches==1)[0][0]
        cmc = np.zeros(len(matches))
        cmc[first_hit:] = 1
        num_rel = matches.sum()
        ap = 0.0
        hit_count = 0
        for j, flag in enumerate(matches):
            if flag:
                hit_count += 1
                ap += hit_count / (j+1)
        ap /= num_rel
        ap_list.append(ap)
        if cmc_acc is None:
            cmc_acc = cmc
        else:
            cmc_acc += cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc/num_valid if num_valid > 0 else None
    return mAP, cmc_avg

def evaluate_phase6():
    """
    Phase 6: Aggregated query vs. mixed aggregated gallery.
    For each query player, in the gallery set use the fused aggregated embedding for that same player (if text exists)
    and the regular aggregated projected embeddings for other players.
    """
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    
    query_agg = aggregate_player_embeddings(query_files, get_image_embedding_trained)
    gallery_proj_agg = aggregate_player_embeddings(gallery_files, get_image_embedding_trained)
    gallery_raw_agg = aggregate_player_embeddings(gallery_files, get_image_raw_embedding)
    query_text_agg = aggregate_player_texts(query_files)
    
    query_ids = sorted(query_agg.keys())
    gallery_ids = sorted(gallery_proj_agg.keys())
    
    ap_list = []
    cmc_acc = None
    num_valid = 0
    for qid in query_ids:
        query_emb = query_agg[qid]
        mixed_gallery = {}
        for pid in gallery_ids:
            if pid == qid and (qid in query_text_agg):
                mixed_gallery[pid] = fuse_and_project(gallery_raw_agg[pid], query_text_agg[qid])
            else:
                mixed_gallery[pid] = gallery_proj_agg[pid]
        gallery_list = [mixed_gallery[pid] for pid in gallery_ids]
        sims = np.dot(np.stack(gallery_list), query_emb)
        sorted_idx = np.argsort(-sims)
        sorted_gallery_ids = np.array(gallery_ids)[sorted_idx]
        matches = (sorted_gallery_ids == qid).astype(np.int32)
        if matches.sum() == 0:
            continue
        first_hit = np.where(matches==1)[0][0]
        cmc = np.zeros(len(matches))
        cmc[first_hit:] = 1
        num_rel = matches.sum()
        ap = 0.0
        hit_count = 0
        for j, flag in enumerate(matches):
            if flag:
                hit_count += 1
                ap += hit_count / (j+1)
        ap /= num_rel
        ap_list.append(ap)
        if cmc_acc is None:
            cmc_acc = cmc
        else:
            cmc_acc += cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc/num_valid if num_valid > 0 else None
    return mAP, cmc_avg

# ------------------- New Phases: Fused Query and Aggregated Evaluations -------------------

def evaluate_phase7():
    """
    Phase 7: Fused Query vs Fused Gallery [all gallery images fused with query text].
    For each query, if a text embedding is available, fuse the query image (using its raw embedding)
    with its text embedding to obtain a fused query embedding, and fuse every gallery image's raw embedding
    with the same query text embedding. Otherwise, use the original projected embeddings.
    """
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    gallery_ids = [parse_filename(f) for f in gallery_files]
    gallery_raw_embeddings = []
    gallery_proj_embeddings = []
    for f in tqdm(gallery_files, desc="Processing gallery images (Phase 7)"):
        gallery_raw_embeddings.append(get_image_raw_embedding(f))
        gallery_proj_embeddings.append(get_image_embedding_trained(f))
    gallery_raw_embeddings = np.stack(gallery_raw_embeddings)
    gallery_proj_embeddings = np.stack(gallery_proj_embeddings)

    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    ap_list = []
    cmc_acc = None
    num_valid_queries = 0
    for qf in tqdm(query_files, desc="Processing query images (Phase 7)"):
        query_id = parse_filename(qf)
        query_text_emb = get_query_text_embedding_trained(qf)
        if query_text_emb is not None:
            # Fuse query image with its text embedding using raw embedding.
            query_raw_emb = get_image_raw_embedding(qf)
            fused_query_emb = fuse_and_project(query_raw_emb, query_text_emb)
            # Fuse every gallery image with the query text embedding.
            fused_gallery = []
            for raw_emb in gallery_raw_embeddings:
                fused_gallery.append(fuse_and_project(raw_emb, query_text_emb))
            fused_gallery = np.stack(fused_gallery)
        else:
            fused_query_emb = get_image_embedding_trained(qf)
            fused_gallery = gallery_proj_embeddings
        result = compute_metrics_for_query(fused_query_emb, fused_gallery, query_id, gallery_ids)
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
    return mAP, cmc_avg

def evaluate_phase8():
    """
    Phase 8: Fused Query vs Mixed Gallery [fused for same player only].
    For each query, if a text embedding is available, fuse the query image with its text embedding
    (using its raw embedding) to obtain a fused query embedding, and fuse only the gallery images
    that belong to the same player with the query text embedding. Otherwise, use the original projected embeddings.
    """
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    gallery_ids = [parse_filename(f) for f in gallery_files]
    gallery_raw_embeddings = []
    gallery_proj_embeddings = []
    for f in tqdm(gallery_files, desc="Processing gallery images (Phase 8)"):
        gallery_raw_embeddings.append(get_image_raw_embedding(f))
        gallery_proj_embeddings.append(get_image_embedding_trained(f))
    gallery_raw_embeddings = np.stack(gallery_raw_embeddings)
    gallery_proj_embeddings = np.stack(gallery_proj_embeddings)

    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    ap_list = []
    cmc_acc = None
    num_valid_queries = 0
    for qf in tqdm(query_files, desc="Processing query images (Phase 8)"):
        query_id = parse_filename(qf)
        query_text_emb = get_query_text_embedding_trained(qf)
        if query_text_emb is not None:
            # Fuse query image with its text embedding using raw embedding.
            query_raw_emb = get_image_raw_embedding(qf)
            fused_query_emb = fuse_and_project(query_raw_emb, query_text_emb)
            # Build mixed gallery: fuse gallery images for the same player only.
            fused_gallery = []
            for i, g_raw in enumerate(gallery_raw_embeddings):
                g_id = gallery_ids[i]
                if g_id == query_id:
                    fused_gallery.append(fuse_and_project(g_raw, query_text_emb))
                else:
                    fused_gallery.append(gallery_proj_embeddings[i])
            fused_gallery = np.stack(fused_gallery)
        else:
            fused_query_emb = get_image_embedding_trained(qf)
            fused_gallery = gallery_proj_embeddings
        result = compute_metrics_for_query(fused_query_emb, fused_gallery, query_id, gallery_ids)
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
    return mAP, cmc_avg

def evaluate_phase9():
    """
    Phase 9: Fused Aggregated Query vs Aggregated Gallery fused with aggregated query text.
    For each player, if an aggregated query text is available, fuse the player's aggregated query raw embedding
    with the aggregated query text embedding to obtain a fused aggregated query embedding, and similarly fuse
    the aggregated gallery raw embedding with the aggregated query text embedding. Otherwise, use the aggregated
    projected embeddings.
    Then perform matching across players.
    """
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    
    # For query aggregation, obtain both projected and raw embeddings.
    query_agg_proj = aggregate_player_embeddings(query_files, get_image_embedding_trained)
    query_agg_raw = aggregate_player_embeddings(query_files, get_image_raw_embedding)
    gallery_proj_agg = aggregate_player_embeddings(gallery_files, get_image_embedding_trained)
    gallery_raw_agg = aggregate_player_embeddings(gallery_files, get_image_raw_embedding)
    query_text_agg = aggregate_player_texts(query_files)
    
    fused_query_agg = {}
    fused_gallery = {}
    player_ids = sorted(query_agg_proj.keys())
    for pid in player_ids:
        if pid in query_text_agg:
            fused_query_agg[pid] = fuse_and_project(query_agg_raw[pid], query_text_agg[pid])
            fused_gallery[pid] = fuse_and_project(gallery_raw_agg[pid], query_text_agg[pid])
        else:
            fused_query_agg[pid] = query_agg_proj[pid]
            fused_gallery[pid] = gallery_proj_agg[pid]
    
    query_ids = sorted(fused_query_agg.keys())
    gallery_ids = sorted(fused_gallery.keys())
    gallery_embs = [fused_gallery[pid] for pid in gallery_ids]
    
    ap_list = []
    cmc_acc = None
    num_valid = 0
    for qid in query_ids:
        query_emb = fused_query_agg[qid]
        sims = np.dot(np.stack(gallery_embs), query_emb)
        sorted_idx = np.argsort(-sims)
        sorted_gallery_ids = np.array(gallery_ids)[sorted_idx]
        matches = (sorted_gallery_ids == qid).astype(np.int32)
        if matches.sum() == 0:
            continue
        first_hit = np.where(matches==1)[0][0]
        cmc = np.zeros(len(matches))
        cmc[first_hit:] = 1
        num_rel = matches.sum()
        ap = 0.0
        hit_count = 0
        for j, flag in enumerate(matches):
            if flag:
                hit_count += 1
                ap += hit_count / (j+1)
        ap /= num_rel
        ap_list.append(ap)
        if cmc_acc is None:
            cmc_acc = cmc
        else:
            cmc_acc += cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc/num_valid if num_valid > 0 else None
    return mAP, cmc_avg

def evaluate_phase10():
    """
    Phase 10: Fused Aggregated Query vs Mixed Aggregated Gallery.
    For each player, if an aggregated query text is available, fuse the player's aggregated query raw embedding
    with the aggregated query text embedding to obtain a fused aggregated query embedding; otherwise use the aggregated
    projected query. For the gallery, use the fused aggregated gallery for the same player (if text exists) and the
    projected aggregated gallery for other players. Then perform matching across players.
    """
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    
    query_agg_proj = aggregate_player_embeddings(query_files, get_image_embedding_trained)
    query_agg_raw = aggregate_player_embeddings(query_files, get_image_raw_embedding)
    gallery_proj_agg = aggregate_player_embeddings(gallery_files, get_image_embedding_trained)
    gallery_raw_agg = aggregate_player_embeddings(gallery_files, get_image_raw_embedding)
    query_text_agg = aggregate_player_texts(query_files)
    
    fused_query_agg = {}
    for pid in query_agg_proj:
        if pid in query_text_agg:
            fused_query_agg[pid] = fuse_and_project(query_agg_raw[pid], query_text_agg[pid])
        else:
            fused_query_agg[pid] = query_agg_proj[pid]
    
    gallery_fused_mixed = {}
    gallery_ids = sorted(gallery_proj_agg.keys())
    for pid in gallery_ids:
        if pid in query_text_agg:
            gallery_fused_mixed[pid] = fuse_and_project(gallery_raw_agg[pid], query_text_agg[pid])
        else:
            gallery_fused_mixed[pid] = gallery_proj_agg[pid]
    
    query_ids = sorted(fused_query_agg.keys())
    gallery_ids = sorted(gallery_fused_mixed.keys())
    gallery_embs = [gallery_fused_mixed[pid] for pid in gallery_ids]
    
    ap_list = []
    cmc_acc = None
    num_valid = 0
    for qid in query_ids:
        query_emb = fused_query_agg[qid]
        sims = np.dot(np.stack(gallery_embs), query_emb)
        sorted_idx = np.argsort(-sims)
        sorted_gallery_ids = np.array(gallery_ids)[sorted_idx]
        matches = (sorted_gallery_ids == qid).astype(np.int32)
        if matches.sum() == 0:
            continue
        first_hit = np.where(matches==1)[0][0]
        cmc = np.zeros(len(matches))
        cmc[first_hit:] = 1
        num_rel = matches.sum()
        ap = 0.0
        hit_count = 0
        for j, flag in enumerate(matches):
            if flag:
                hit_count += 1
                ap += hit_count / (j+1)
        ap /= num_rel
        ap_list.append(ap)
        if cmc_acc is None:
            cmc_acc = cmc
        else:
            cmc_acc += cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc/num_valid if num_valid > 0 else None
    return mAP, cmc_avg

# ------------------- Run All Evaluations -------------------

mAP_phase1, cmc_phase1 = evaluate_phase1()
mAP_phase2, cmc_phase2 = evaluate_phase2()
mAP_phase3, cmc_phase3 = evaluate_phase3()

mAP_phase4, cmc_phase4 = evaluate_phase4()
mAP_phase5, cmc_phase5 = evaluate_phase5()
mAP_phase6, cmc_phase6 = evaluate_phase6()

# New phases
mAP_phase7, cmc_phase7 = evaluate_phase7()
mAP_phase8, cmc_phase8 = evaluate_phase8()
mAP_phase9, cmc_phase9 = evaluate_phase9()
mAP_phase10, cmc_phase10 = evaluate_phase10()

# Create the directory for the log file if it doesn't exist
log_dir = os.path.dirname(LOG_FILE_MODIFIED)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

with open(LOG_FILE_MODIFIED, "w") as f:
    f.write("Trained Projection CLIP ReID Evaluation Results\n\n")
    
    f.write("Phase 1 (Image-Image Matching):\n")
    f.write(f"mAP: {mAP_phase1:.4f}\n")
    if cmc_phase1 is not None:
        f.write(f"Rank-1: {cmc_phase1[0]:.4f}, Rank-5: {cmc_phase1[4] if len(cmc_phase1)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase1[9] if len(cmc_phase1)>9 else 0:.4f}, Rank-20: {cmc_phase1[19] if len(cmc_phase1)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid queries processed in Phase 1.\n\n")
    
    f.write("Phase 2 (Query vs Fused Gallery [all gallery images fused with query text]):\n")
    f.write(f"mAP: {mAP_phase2:.4f}\n")
    if cmc_phase2 is not None:
        f.write(f"Rank-1: {cmc_phase2[0]:.4f}, Rank-5: {cmc_phase2[4] if len(cmc_phase2)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase2[9] if len(cmc_phase2)>9 else 0:.4f}, Rank-20: {cmc_phase2[19] if len(cmc_phase2)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid queries processed in Phase 2.\n\n")
    
    f.write("Phase 3 (Query vs Mixed Gallery [fused for same player only]):\n")
    f.write(f"mAP: {mAP_phase3:.4f}\n")
    if cmc_phase3 is not None:
        f.write(f"Rank-1: {cmc_phase3[0]:.4f}, Rank-5: {cmc_phase3[4] if len(cmc_phase3)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase3[9] if len(cmc_phase3)>9 else 0:.4f}, Rank-20: {cmc_phase3[19] if len(cmc_phase3)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid queries processed in Phase 3.\n\n")
    
    f.write("Phase 4 (Aggregated Query vs Aggregated Gallery, both projected):\n")
    f.write(f"mAP: {mAP_phase4:.4f}\n")
    if cmc_phase4 is not None:
        f.write(f"Rank-1: {cmc_phase4[0]:.4f}, Rank-5: {cmc_phase4[4] if len(cmc_phase4)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase4[9] if len(cmc_phase4)>9 else 0:.4f}, Rank-20: {cmc_phase4[19] if len(cmc_phase4)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid players processed in Phase 4.\n\n")
    
    f.write("Phase 5 (Aggregated Query vs Aggregated Gallery fused with aggregated query text):\n")
    f.write(f"mAP: {mAP_phase5:.4f}\n")
    if cmc_phase5 is not None:
        f.write(f"Rank-1: {cmc_phase5[0]:.4f}, Rank-5: {cmc_phase5[4] if len(cmc_phase5)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase5[9] if len(cmc_phase5)>9 else 0:.4f}, Rank-20: {cmc_phase5[19] if len(cmc_phase5)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid players processed in Phase 5.\n\n")
    
    f.write("Phase 6 (Aggregated Query vs Mixed Aggregated Gallery):\n")
    f.write(f"mAP: {mAP_phase6:.4f}\n")
    if cmc_phase6 is not None:
        f.write(f"Rank-1: {cmc_phase6[0]:.4f}, Rank-5: {cmc_phase6[4] if len(cmc_phase6)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase6[9] if len(cmc_phase6)>9 else 0:.4f}, Rank-20: {cmc_phase6[19] if len(cmc_phase6)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid players processed in Phase 6.\n\n")
    
    # New phases logging
    f.write("Phase 7 (Fused Query vs Fused Gallery):\n")
    f.write(f"mAP: {mAP_phase7:.4f}\n")
    if cmc_phase7 is not None:
        f.write(f"Rank-1: {cmc_phase7[0]:.4f}, Rank-5: {cmc_phase7[4] if len(cmc_phase7)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase7[9] if len(cmc_phase7)>9 else 0:.4f}, Rank-20: {cmc_phase7[19] if len(cmc_phase7)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid queries processed in Phase 7.\n\n")
    
    f.write("Phase 8 (Fused Query vs Mixed Gallery):\n")
    f.write(f"mAP: {mAP_phase8:.4f}\n")
    if cmc_phase8 is not None:
        f.write(f"Rank-1: {cmc_phase8[0]:.4f}, Rank-5: {cmc_phase8[4] if len(cmc_phase8)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase8[9] if len(cmc_phase8)>9 else 0:.4f}, Rank-20: {cmc_phase8[19] if len(cmc_phase8)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid queries processed in Phase 8.\n\n")
    
    f.write("Phase 9 (Fused Aggregated Query vs Aggregated Gallery fused with aggregated query text):\n")
    f.write(f"mAP: {mAP_phase9:.4f}\n")
    if cmc_phase9 is not None:
        f.write(f"Rank-1: {cmc_phase9[0]:.4f}, Rank-5: {cmc_phase9[4] if len(cmc_phase9)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase9[9] if len(cmc_phase9)>9 else 0:.4f}, Rank-20: {cmc_phase9[19] if len(cmc_phase9)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid players processed in Phase 9.\n\n")
    
    f.write("Phase 10 (Fused Aggregated Query vs Mixed Aggregated Gallery):\n")
    f.write(f"mAP: {mAP_phase10:.4f}\n")
    if cmc_phase10 is not None:
        f.write(f"Rank-1: {cmc_phase10[0]:.4f}, Rank-5: {cmc_phase10[4] if len(cmc_phase10)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase10[9] if len(cmc_phase10)>9 else 0:.4f}, Rank-20: {cmc_phase10[19] if len(cmc_phase10)>19 else 0:.4f}\n")
    else:
        f.write("No valid players processed in Phase 10.\n")
        
print(f"Trained Projection CLIP ReID results logged to {LOG_FILE_MODIFIED}")



