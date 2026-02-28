# 6 phases
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
LOG_FILE_MODIFIED = "" # log file to save results

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)
clip_model.eval()

# --- Embedding functions ---
def get_image_embedding_pretrained(img_path):
    """
    Returns the pretrained CLIP image embedding (normalized).
    """
    image = Image.open(img_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = clip_model.encode_image(image_input)
    emb = emb.float()
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy().squeeze()

def get_query_text_embedding_pretrained(query_img_path):
    """
    Loads a text embedding corresponding to a query image.
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

def fuse_gallery_with_query_text(gallery_emb, query_text_emb):
    """
    Fuses a gallery embedding with a query text embedding.
    Since all embeddings lie in the same CLIP space, a simple summation (with normalization) is used.
    """
    fused = gallery_emb + query_text_emb
    fused = fused / (np.linalg.norm(fused) + 1e-8)
    return fused

def parse_filename(filename):
    """
    Returns the player ID assumed to be the first token in the filename.
    """
    return os.path.basename(filename).split("_")[0]

def compute_metrics_for_query(query_emb, gallery_embeddings, query_id, gallery_ids):
    """
    Computes cosine similarity, sorts the gallery, and then computes AP and CMC.
    """
    sims = np.dot(gallery_embeddings, query_emb)
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

# --- Aggregation Helpers ---
def aggregate_embeddings(emb_list):
    """
    Aggregates a list of embeddings by summing and then L2-normalizing.
    """
    if not emb_list:
        return None
    agg = np.sum(np.stack(emb_list), axis=0)
    return agg / (np.linalg.norm(agg) + 1e-8)

def aggregate_player_embeddings(file_list, embedding_func):
    """
    Groups files by player ID and aggregates embeddings using embedding_func.
    Returns a dictionary mapping player ID to aggregated embedding.
    """
    agg_dict = {}
    for f in file_list:
        pid = parse_filename(f)
        emb = embedding_func(f)
        if pid not in agg_dict:
            agg_dict[pid] = []
        agg_dict[pid].append(emb)
    for pid in agg_dict:
        agg_dict[pid] = aggregate_embeddings(agg_dict[pid])
    return agg_dict

def aggregate_player_texts(query_files):
    """
    Groups query files by player ID and aggregates available text embeddings.
    Returns a dictionary mapping player ID to aggregated text embedding.
    """
    text_dict = {}
    for f in query_files:
        pid = parse_filename(f)
        text_emb = get_query_text_embedding_pretrained(f)
        if text_emb is not None:
            if pid not in text_dict:
                text_dict[pid] = []
            text_dict[pid].append(text_emb)
    for pid in text_dict:
        text_dict[pid] = aggregate_embeddings(text_dict[pid])
    return text_dict

# --- Phase 1-3: Per-Image Evaluation ---
def evaluate_phase1():
    """
    Phase 1: Image-to-image matching using plain pretrained CLIP embeddings.
    """
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    gallery_ids = [parse_filename(f) for f in gallery_files]
    gallery_embeddings = []
    for f in tqdm(gallery_files, desc="Processing gallery images (Phase 1)"):
        gallery_embeddings.append(get_image_embedding_pretrained(f))
    gallery_embeddings = np.stack(gallery_embeddings)
    
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    ap_list = []
    cmc_acc = None
    num_valid = 0
    for qf in tqdm(query_files, desc="Processing query images (Phase 1)"):
        query_id = parse_filename(qf)
        query_emb = get_image_embedding_pretrained(qf)
        result = compute_metrics_for_query(query_emb, gallery_embeddings, query_id, gallery_ids)
        if result[0] is None:
            continue
        ap, cmc = result
        ap_list.append(ap)
        cmc_acc = cmc if cmc_acc is None else cmc_acc + cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc / num_valid if num_valid > 0 else None
    return mAP, cmc_avg

def evaluate_phase2():
    """
    Phase 2: For each query, fuse its text embedding with every gallery embedding.
    If no text is available, use the plain gallery embeddings.
    """
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    gallery_ids = [parse_filename(f) for f in gallery_files]
    gallery_embeddings = []
    for f in tqdm(gallery_files, desc="Processing gallery images (Phase 2)"):
        gallery_embeddings.append(get_image_embedding_pretrained(f))
    gallery_embeddings = np.stack(gallery_embeddings)
    
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    ap_list = []
    cmc_acc = None
    num_valid = 0
    for qf in tqdm(query_files, desc="Processing query images (Phase 2)"):
        query_id = parse_filename(qf)
        query_emb = get_image_embedding_pretrained(qf)
        query_text_emb = get_query_text_embedding_pretrained(qf)
        if query_text_emb is None:
            fused_gallery = gallery_embeddings
        else:
            fused_gallery = []
            for g_emb in gallery_embeddings:
                fused_gallery.append(fuse_gallery_with_query_text(g_emb, query_text_emb))
            fused_gallery = np.stack(fused_gallery)
        result = compute_metrics_for_query(query_emb, fused_gallery, query_id, gallery_ids)
        if result[0] is None:
            continue
        ap, cmc = result
        ap_list.append(ap)
        cmc_acc = cmc if cmc_acc is None else cmc_acc + cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc / num_valid if num_valid > 0 else None
    return mAP, cmc_avg

def evaluate_phase3():
    """
    Phase 3: For each query, fuse its text embedding only with gallery embeddings from the same identity.
    Gallery images of other identities are used as is.
    """
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    gallery_ids = [parse_filename(f) for f in gallery_files]
    gallery_embeddings = []
    for f in tqdm(gallery_files, desc="Processing gallery images (Phase 3)"):
        gallery_embeddings.append(get_image_embedding_pretrained(f))
    gallery_embeddings = np.stack(gallery_embeddings)
    
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    ap_list = []
    cmc_acc = None
    num_valid = 0
    for qf in tqdm(query_files, desc="Processing query images (Phase 3)"):
        query_id = parse_filename(qf)
        query_emb = get_image_embedding_pretrained(qf)
        query_text_emb = get_query_text_embedding_pretrained(qf)
        fused_gallery = []
        for i, g_emb in enumerate(gallery_embeddings):
            if gallery_ids[i] == query_id and (query_text_emb is not None):
                fused_gallery.append(fuse_gallery_with_query_text(g_emb, query_text_emb))
            else:
                fused_gallery.append(g_emb)
        fused_gallery = np.stack(fused_gallery)
        result = compute_metrics_for_query(query_emb, fused_gallery, query_id, gallery_ids)
        if result[0] is None:
            continue
        ap, cmc = result
        ap_list.append(ap)
        cmc_acc = cmc if cmc_acc is None else cmc_acc + cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc / num_valid if num_valid > 0 else None
    return mAP, cmc_avg

# --- Phase 4-6: Aggregated (Per-Player) Evaluation ---
def evaluate_phase4():
    """
    Phase 4: Aggregated query vs. aggregated gallery matching.
    For each player, aggregate all query embeddings and all gallery embeddings (plain pretrained CLIP).
    """
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    
    query_agg = aggregate_player_embeddings(query_files, get_image_embedding_pretrained)
    gallery_agg = aggregate_player_embeddings(gallery_files, get_image_embedding_pretrained)
    
    query_ids = sorted(query_agg.keys())
    gallery_ids = sorted(gallery_agg.keys())
    gallery_list = [gallery_agg[pid] for pid in gallery_ids]
    
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
        first_hit = np.where(matches == 1)[0][0]
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
        cmc_acc = cmc if cmc_acc is None else cmc_acc + cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc/num_valid if num_valid > 0 else None
    return mAP, cmc_avg

def evaluate_phase5():
    """
    Phase 5: Aggregated query vs. aggregated gallery fused with aggregated query text.
    For each player, if an aggregated text embedding is available then fuse the aggregated gallery embedding with it.
    """
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    
    query_agg = aggregate_player_embeddings(query_files, get_image_embedding_pretrained)
    gallery_plain = aggregate_player_embeddings(gallery_files, get_image_embedding_pretrained)
    gallery_raw = aggregate_player_embeddings(gallery_files, get_image_embedding_pretrained)  # same as plain here
    query_text_agg = aggregate_player_texts(query_files)
    
    gallery_fused = {}
    for pid in gallery_plain:
        if pid in query_text_agg:
            gallery_fused[pid] = fuse_gallery_with_query_text(gallery_plain[pid], query_text_agg[pid])
        else:
            gallery_fused[pid] = gallery_plain[pid]
    
    query_ids = sorted(query_agg.keys())
    gallery_ids = sorted(gallery_fused.keys())
    gallery_list = [gallery_fused[pid] for pid in gallery_ids]
    
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
        first_hit = np.where(matches == 1)[0][0]
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
        cmc_acc = cmc if cmc_acc is None else cmc_acc + cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc/num_valid if num_valid > 0 else None
    return mAP, cmc_avg

def evaluate_phase6():
    """
    Phase 6: Aggregated query vs. mixed aggregated gallery.
    For each player, use the fused aggregated gallery embedding for that same player (if aggregated text exists),
    and for other players use the plain aggregated gallery embedding.
    """
    query_files = glob.glob(os.path.join(QUERY_DIR, "*.jpg"))
    gallery_files = glob.glob(os.path.join(GALLERY_DIR, "*.jpg"))
    
    query_agg = aggregate_player_embeddings(query_files, get_image_embedding_pretrained)
    gallery_plain = aggregate_player_embeddings(gallery_files, get_image_embedding_pretrained)
    query_text_agg = aggregate_player_texts(query_files)
    
    query_ids = sorted(query_agg.keys())
    gallery_ids = sorted(gallery_plain.keys())
    
    ap_list = []
    cmc_acc = None
    num_valid = 0
    for qid in query_ids:
        query_emb = query_agg[qid]
        mixed_gallery = {}
        for pid in gallery_ids:
            if pid == qid and (qid in query_text_agg):
                mixed_gallery[pid] = fuse_gallery_with_query_text(gallery_plain[pid], query_text_agg[qid])
            else:
                mixed_gallery[pid] = gallery_plain[pid]
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
        cmc_acc = cmc if cmc_acc is None else cmc_acc + cmc
        num_valid += 1
    mAP = np.mean(ap_list) if ap_list else 0.0
    cmc_avg = cmc_acc/num_valid if num_valid > 0 else None
    return mAP, cmc_avg

# --- Run all evaluations ---
mAP_phase1, cmc_phase1 = evaluate_phase1()
mAP_phase2, cmc_phase2 = evaluate_phase2()
mAP_phase3, cmc_phase3 = evaluate_phase3()
mAP_phase4, cmc_phase4 = evaluate_phase4()
mAP_phase5, cmc_phase5 = evaluate_phase5()
mAP_phase6, cmc_phase6 = evaluate_phase6()

with open(LOG_FILE_MODIFIED, "w") as f:
    f.write("Pretrained CLIP ReID Evaluation Results (6 Phases)\n\n")
    
    f.write("Phase 1 (Image-Image Matching):\n")
    f.write(f"mAP: {mAP_phase1:.4f}\n")
    if cmc_phase1 is not None:
        f.write(f"Rank-1: {cmc_phase1[0]:.4f}, Rank-5: {cmc_phase1[4] if len(cmc_phase1)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase1[9] if len(cmc_phase1)>9 else 0:.4f}, Rank-20: {cmc_phase1[19] if len(cmc_phase1)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid queries processed in Phase 1.\n\n")
    
    f.write("Phase 2 (Plain Query vs Fused Gallery [Gallery + Query Text]):\n")
    f.write(f"mAP: {mAP_phase2:.4f}\n")
    if cmc_phase2 is not None:
        f.write(f"Rank-1: {cmc_phase2[0]:.4f}, Rank-5: {cmc_phase2[4] if len(cmc_phase2)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase2[9] if len(cmc_phase2)>9 else 0:.4f}, Rank-20: {cmc_phase2[19] if len(cmc_phase2)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid queries processed in Phase 2.\n\n")
    
    f.write("Phase 3 (Query vs Mixed Gallery [Fused for same identity only]):\n")
    f.write(f"mAP: {mAP_phase3:.4f}\n")
    if cmc_phase3 is not None:
        f.write(f"Rank-1: {cmc_phase3[0]:.4f}, Rank-5: {cmc_phase3[4] if len(cmc_phase3)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase3[9] if len(cmc_phase3)>9 else 0:.4f}, Rank-20: {cmc_phase3[19] if len(cmc_phase3)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid queries processed in Phase 3.\n\n")
    
    f.write("Phase 4 (Aggregated Query vs Aggregated Gallery):\n")
    f.write(f"mAP: {mAP_phase4:.4f}\n")
    if cmc_phase4 is not None:
        f.write(f"Rank-1: {cmc_phase4[0]:.4f}, Rank-5: {cmc_phase4[4] if len(cmc_phase4)>4 else 0:.4f}, "
                f"Rank-10: {cmc_phase4[9] if len(cmc_phase4)>9 else 0:.4f}, Rank-20: {cmc_phase4[19] if len(cmc_phase4)>19 else 0:.4f}\n\n")
    else:
        f.write("No valid players processed in Phase 4.\n\n")
    
    f.write("Phase 5 (Aggregated Query vs Aggregated Gallery Fused with Aggregated Query Text):\n")
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
                f"Rank-10: {cmc_phase6[9] if len(cmc_phase6)>9 else 0:.4f}, Rank-20: {cmc_phase6[19] if len(cmc_phase6)>19 else 0:.4f}\n")
    else:
        f.write("No valid players processed in Phase 6.\n")
        
print(f"Pretrained CLIP ReID results logged to {LOG_FILE_MODIFIED}")


