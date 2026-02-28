# Fusion sim text and image
#!/usr/bin/env python
import os
# Set environment variable to help with memory fragmentation.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import re
import torch
import clip
import torch.nn.functional as F
from PIL import Image
import numpy as np
import logging
import argparse
from tqdm import tqdm
import random

######################
# Logging Setup
######################
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "clip_reid_pretrain_100fusion.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        filename=log_path,
        filemode="w"  # refresh log file each run
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return logging.getLogger("clip_reid")

######################
# Utility Functions
######################
def parse_filename(filename):
    """
    Expected filename format:
      <player_id>_<game_id>_<frame_id>_<track_id>_<potential_suffix>.jpg
    Assumes:
      - First token is player_id (e.g., "0007")
      - Second token's first 10 characters form game_id.
    """
    parts = filename.split('_')
    player_id = parts[0]
    game_id = parts[1][:10]
    return player_id, game_id

def get_image_embedding(img_path, model, device, preprocess):
    image = Image.open(img_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().squeeze()  # returns a 1D vector

######################
# ReID Metrics (mAP & CMC)
######################
def compute_ap_cmc(similarities, query_pid, gallery_pids):
    """
    similarities: 1D numpy array of similarity scores (higher is better)
    query_pid: string for query player id
    gallery_pids: list or numpy array of gallery player ids (as strings)
    Returns:
      ap: average precision for the query (0 if no match)
      cmc: binary array with 1 from the rank of the first correct match onward
    """
    sorted_idx = np.argsort(-similarities)
    sorted_gallery = np.array(gallery_pids)[sorted_idx]
    good_mask = (sorted_gallery == query_pid)
    if not np.any(good_mask):
        return 0.0, np.zeros(len(sorted_gallery))
    first_hit = np.where(good_mask)[0][0]
    cmc = np.zeros(len(sorted_gallery))
    cmc[first_hit:] = 1
    num_good = np.sum(good_mask)
    ap = 0.0
    hit_count = 0
    for i, flag in enumerate(good_mask):
        if flag:
            hit_count += 1
            ap += hit_count / (i + 1)
    ap /= num_good
    return ap, cmc

######################
# Standard Evaluation
######################
def evaluate_reid(test_dir, text_embed_dir, model, device, preprocess, logger):
    # Load text embeddings (from text_embed_dir/test; fallback to train if needed)
    gallery_files = glob.glob(os.path.join(text_embed_dir, "test", "*.pt"))
    if len(gallery_files) == 0:
        gallery_files = glob.glob(os.path.join(text_embed_dir, "train", "*.pt"))
    if len(gallery_files) == 0:
        logger.error("No text embeddings found in test or train folders.")
        return
    gallery_embeddings = []
    gallery_ids = []
    for tf in gallery_files:
        basename = os.path.basename(tf)
        pid, game_id = basename.replace(".pt", "").split("_")
        emb = torch.load(tf).numpy().squeeze()
        gallery_embeddings.append(emb)
        gallery_ids.append(pid)
    gallery_embeddings = np.stack(gallery_embeddings, axis=0)
    
    test_files = glob.glob(os.path.join(test_dir, "*.jpg"))
    ap_list = []
    cmc_acc = np.zeros(len(gallery_ids))
    cosine_similarities = []
    for img_file in tqdm(test_files, desc="Standard evaluation on test images"):
        basename = os.path.basename(img_file)
        query_pid, game_id = parse_filename(basename)
        img_emb = get_image_embedding(img_file, model, device, preprocess)
        sim = np.dot(gallery_embeddings, img_emb)
        cosine_similarities.append(sim.max())
        ap, cmc = compute_ap_cmc(sim, query_pid, gallery_ids)
        ap_list.append(ap)
        cmc_acc += cmc
    num_queries = len(ap_list)
    mAP = np.mean(ap_list)
    cmc_avg = cmc_acc / num_queries
    logger.info("Standard Evaluation Results ---------------------------------------------------")
    logger.info("mAP: {:.1%}".format(mAP))
    def safe_get(arr, idx):
        return arr[idx] if idx < len(arr) else arr[-1]
    logger.info("Rank-1: {:.1%}".format(safe_get(cmc_avg, 0)))
    logger.info("Rank-5: {:.1%}".format(safe_get(cmc_avg, 4)))
    logger.info("Rank-10: {:.1%}".format(safe_get(cmc_avg, 9)))
    logger.info("Rank-20: {:.1%}".format(safe_get(cmc_avg, 19)))
    logger.info("Average cosine similarity (top match): {:.4f}".format(np.mean(cosine_similarities)))
    return mAP, cmc_avg

######################
# Comprehensive Evaluation (Simple Average)
######################
def evaluate_comprehensive(test_dir, gallery_test_dir, text_embed_dir, model, device, preprocess, logger):
    """
    Comprehensive Evaluation (Simple Average):
      For each player, compute the mean gallery image embedding (from gallery_test_dir).
      For each candidate text embedding, average it with the player's gallery mean,
      re-normalize, and then evaluate query images against these modified embeddings.
    """
    # Compute mean gallery image embedding per player from gallery_test_dir.
    gallery_files = glob.glob(os.path.join(gallery_test_dir, "*.jpg"))
    player2gallery = {}
    for gf in gallery_files:
        basename = os.path.basename(gf)
        pid = gf.split('_')[0]
        emb = get_image_embedding(gf, model, device, preprocess)
        if pid not in player2gallery:
            player2gallery[pid] = []
        player2gallery[pid].append(emb)
    player_gallery_mean = {}
    for pid, emb_list in player2gallery.items():
        mean_emb = np.mean(np.stack(emb_list, axis=0), axis=0)
        mean_emb /= (np.linalg.norm(mean_emb) + 1e-8)
        player_gallery_mean[pid] = mean_emb

    # Load candidate text embeddings.
    gallery_text_files = glob.glob(os.path.join(text_embed_dir, "test", "*.pt"))
    if len(gallery_text_files) == 0:
        gallery_text_files = glob.glob(os.path.join(text_embed_dir, "train", "*.pt"))
    if len(gallery_text_files) == 0:
        logger.error("No text embeddings found for comprehensive evaluation.")
        return
    comp_embeddings = []
    comp_ids = []
    for tf in gallery_text_files:
        basename = os.path.basename(tf)
        pid, game_id = basename.replace(".pt", "").split("_")
        text_emb = torch.load(tf).numpy().squeeze()
        if pid in player_gallery_mean:
            gallery_mean = player_gallery_mean[pid]
            comp_emb = (text_emb + gallery_mean) / 2.0
            comp_emb /= (np.linalg.norm(comp_emb) + 1e-8)
        else:
            comp_emb = text_emb / (np.linalg.norm(text_emb) + 1e-8)
        comp_embeddings.append(comp_emb)
        comp_ids.append(pid)
    comp_embeddings = np.stack(comp_embeddings, axis=0)

    # Evaluate query images against these comprehensive embeddings.
    test_files = glob.glob(os.path.join(test_dir, "*.jpg"))
    ap_list = []
    cmc_acc = np.zeros(len(comp_ids))
    cosine_similarities = []
    for img_file in tqdm(test_files, desc="Comprehensive evaluation (simple average) on query images"):
        basename = os.path.basename(img_file)
        query_pid, game_id = parse_filename(basename)
        img_emb = get_image_embedding(img_file, model, device, preprocess)
        sim = np.dot(comp_embeddings, img_emb)
        cosine_similarities.append(sim.max())
        ap, cmc = compute_ap_cmc(sim, query_pid, comp_ids)
        ap_list.append(ap)
        cmc_acc += cmc
    num_queries = len(ap_list)
    mAP = np.mean(ap_list)
    cmc_avg = cmc_acc / num_queries
    logger.info("Comprehensive Evaluation (Simple Average) Results ---------------------------------------------------")
    logger.info("mAP: {:.1%}".format(mAP))
    def safe_get(arr, idx):
        return arr[idx] if idx < len(arr) else arr[-1]
    logger.info("Rank-1: {:.1%}".format(safe_get(cmc_avg, 0)))
    logger.info("Rank-5: {:.1%}".format(safe_get(cmc_avg, 4)))
    logger.info("Rank-10: {:.1%}".format(safe_get(cmc_avg, 9)))
    logger.info("Rank-20: {:.1%}".format(safe_get(cmc_avg, 19)))
    logger.info("Average cosine similarity (comprehensive simple average top match): {:.4f}".format(np.mean(cosine_similarities)))
    return mAP, cmc_avg

######################
# Fusion Evaluation (Weighted Combination)
######################
def evaluate_fusion(test_dir, gallery_test_dir, text_embed_dir, model, device, preprocess, logger, alpha=0.5):
    """
    Fusion Evaluation:
      For each candidate text embedding, combine its similarity with the query and the similarity between
      the query and the player's mean gallery embedding, using a weighted combination:
         final_score = alpha * sim_text + (1 - alpha) * sim_gallery.
      Then, compute mAP and CMC.
    """
    # Compute mean gallery image embedding per player from gallery_test_dir.
    gallery_files = glob.glob(os.path.join(gallery_test_dir, "*.jpg"))
    player2gallery = {}
    for gf in gallery_files:
        basename = os.path.basename(gf)
        pid = gf.split('_')[0]
        emb = get_image_embedding(gf, model, device, preprocess)
        if pid not in player2gallery:
            player2gallery[pid] = []
        player2gallery[pid].append(emb)
    player_gallery_mean = {}
    for pid, emb_list in player2gallery.items():
        mean_emb = np.mean(np.stack(emb_list, axis=0), axis=0)
        mean_emb /= (np.linalg.norm(mean_emb) + 1e-8)
        player_gallery_mean[pid] = mean_emb

    # Load candidate text embeddings.
    gallery_text_files = glob.glob(os.path.join(text_embed_dir, "test", "*.pt"))
    if len(gallery_text_files) == 0:
        gallery_text_files = glob.glob(os.path.join(text_embed_dir, "train", "*.pt"))
    if len(gallery_text_files) == 0:
        logger.error("No text embeddings found for fusion evaluation.")
        return
    # For fusion, we keep each candidate text embedding unchanged.
    fusion_embeddings = []  # store tuples: (text_emb, gallery_mean)
    fusion_ids = []         # corresponding player ids
    for tf in gallery_text_files:
        basename = os.path.basename(tf)
        pid, game_id = basename.replace(".pt", "").split("_")
        text_emb = torch.load(tf).numpy().squeeze()
        if pid in player_gallery_mean:
            gallery_mean = player_gallery_mean[pid]
        else:
            gallery_mean = np.zeros_like(text_emb)
        fusion_embeddings.append((text_emb, gallery_mean))
        fusion_ids.append(pid)

    # Evaluate each query image using weighted fusion.
    test_files = glob.glob(os.path.join(test_dir, "*.jpg"))
    ap_list = []
    cmc_acc = np.zeros(len(fusion_ids))
    fusion_cosine_sim = []
    for img_file in tqdm(test_files, desc="Fusion evaluation on query images"):
        basename = os.path.basename(img_file)
        query_pid, game_id = parse_filename(basename)
        img_emb = get_image_embedding(img_file, model, device, preprocess)
        final_scores = []
        for (text_emb, gallery_mean) in fusion_embeddings:
            sim_text = np.dot(img_emb, text_emb)
            sim_gallery = np.dot(img_emb, gallery_mean) if np.linalg.norm(gallery_mean) > 0 else 0.0
            final_score = alpha * sim_text + (1 - alpha) * sim_gallery
            final_scores.append(final_score)
        final_scores = np.array(final_scores)
        fusion_cosine_sim.append(final_scores.max())
        ap, cmc = compute_ap_cmc(final_scores, query_pid, fusion_ids)
        ap_list.append(ap)
        cmc_acc += cmc
    num_queries = len(ap_list)
    mAP = np.mean(ap_list)
    cmc_avg = cmc_acc / num_queries
    logger.info("Fusion Evaluation Results (alpha = {:.2f}) ---------------------------------------------------".format(alpha))
    logger.info("mAP: {:.1%}".format(mAP))
    def safe_get(arr, idx):
        return arr[idx] if idx < len(arr) else arr[-1]
    logger.info("Rank-1: {:.1%}".format(safe_get(cmc_avg, 0)))
    logger.info("Rank-5: {:.1%}".format(safe_get(cmc_avg, 4)))
    logger.info("Rank-10: {:.1%}".format(safe_get(cmc_avg, 9)))
    logger.info("Rank-20: {:.1%}".format(safe_get(cmc_avg, 19)))
    logger.info("Average fusion cosine similarity (top match): {:.4f}".format(np.mean(fusion_cosine_sim)))
    return mAP, cmc_avg

######################
# Fine-tuning Function
######################
def train_finetune(train_dir, text_embed_dir, model, device, preprocess, logger, num_epochs=5, batch_size=32, lr=1e-7):
    """
    Fine-tune CLIP by aligning image embeddings with pre-computed text embeddings.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scale = 16.0

    train_files = glob.glob(os.path.join(train_dir, "*.jpg"))
    samples = []
    for img_file in train_files:
        basename = os.path.basename(img_file)
        pid, game_id = parse_filename(basename)
        text_path = os.path.join(text_embed_dir, "train", f"{pid}_{game_id}.pt")
        if os.path.exists(text_path):
            samples.append((img_file, pid, game_id, text_path))
    num_samples = len(samples)
    logger.info(f"Found {num_samples} training samples for fine-tuning.")

    for epoch in range(num_epochs):
        np.random.shuffle(samples)
        epoch_loss = 0.0
        batch_count = 0
        for i in range(0, num_samples, batch_size):
            batch = samples[i:i+batch_size]
            images = []
            text_embeddings = []
            for img_file, pid, game_id, text_path in batch:
                try:
                    image = Image.open(img_file).convert("RGB")
                    images.append(preprocess(image))
                    text_emb = torch.load(text_path)
                    text_embeddings.append(text_emb.squeeze(0))
                except Exception as e:
                    logger.error(f"Error loading sample {img_file}: {e}")
                    continue
            if len(images) == 0:
                continue
            images = torch.stack(images).to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_embeddings = torch.stack(text_embeddings).to(device)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            logits_per_image = scale * torch.matmul(image_features, text_embeddings.t())
            logits_per_text = scale * torch.matmul(text_embeddings, image_features.t())
            ground_truth = torch.arange(len(images)).to(device)
            loss_image = F.cross_entropy(logits_per_image, ground_truth)
            loss_text  = F.cross_entropy(logits_per_text, ground_truth)
            loss = (loss_image + loss_text) / 2

            if not torch.isfinite(loss):
                logger.error("Non-finite loss encountered, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(images)
            batch_count += 1
        if batch_count > 0:
            epoch_loss /= (batch_count * batch_size)
        else:
            epoch_loss = float('nan')
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
    
    torch.save(model.state_dict(), "clip_finetuned.pt")
    logger.info("Finished fine-tuning and saved checkpoint as clip_finetuned.pt.")

######################
# Main Function
######################
def main():
    parser = argparse.ArgumentParser(description="CLIP ReID Evaluation and Fine-Tuning")
    parser.add_argument("--mode", choices=["pretrained", "train"], default="pretrained",
                        help="Choose to evaluate the pretrained model or to fine-tune and then evaluate.")
    parser.add_argument("--train_dir", type=str, default="",
                        help="Path to the training image directory.")
    parser.add_argument("--test_dir", type=str, default="",
                        help="Path to the test image directory.")
    parser.add_argument("--gallery_test_dir", type=str, default="",
                        help="Path to the gallery test image directory.")
    parser.add_argument("--text_embed_dir", type=str, default="",
                        help="Directory where text embeddings are saved.")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for fine-tuning.")
    parser.add_argument("--lr", type=float, default=1e-7, help="Learning rate for fine-tuning.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Fusion weight for combining text and gallery similarities.")
    args = parser.parse_args()

    log_dir = "" # specify your log directory here
    logger = setup_logging(log_dir)
    logger.info(f"Running in mode: {args.mode}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    if args.mode == "pretrained":
        model.eval()

    if args.mode == "pretrained":
        logger.info("Starting standard evaluation...")
        evaluate_reid(args.test_dir, args.text_embed_dir, model, device, preprocess, logger)
        logger.info("Starting comprehensive evaluation (simple average) ...")
        evaluate_comprehensive(args.test_dir, args.gallery_test_dir, args.text_embed_dir, model, device, preprocess, logger)
        logger.info("Starting fusion evaluation (weighted similarity fusion) ...")
        evaluate_fusion(args.test_dir, args.gallery_test_dir, args.text_embed_dir, model, device, preprocess, logger, alpha=args.alpha)
    elif args.mode == "train":
        train_finetune(args.train_dir, args.text_embed_dir, model, device, preprocess, logger,
                       num_epochs=args.num_epochs, batch_size=args.batch_size, lr=args.lr)
        model.load_state_dict(torch.load("clip_finetuned.pt", map_location=device))
        model.eval()
        logger.info("Post-training standard evaluation...")
        evaluate_reid(args.test_dir, args.text_embed_dir, model, device, preprocess, logger)
        logger.info("Post-training comprehensive evaluation (simple average) ...")
        evaluate_comprehensive(args.test_dir, args.gallery_test_dir, args.text_embed_dir, model, device, preprocess, logger)
        logger.info("Post-training fusion evaluation (weighted similarity fusion) ...")
        evaluate_fusion(args.test_dir, args.gallery_test_dir, args.text_embed_dir, model, device, preprocess, logger, alpha=args.alpha)

if __name__ == "__main__":
    main()


