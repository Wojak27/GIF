# MuPNIT-ReID-light Benchmarking

This repository provides benchmarks for the **MuPNIT-ReID-light** dataset using two families of re-identification methods:

- **Part 1 — CLIP ReID Family** (`mupnit_clip_reid/`): three progressive settings using pretrained CLIP features augmented with player text descriptions.
- **Part 2 — AP3D** (`AP3D/`): video-based re-ID with appearance-preserving 3D convolutions.

**Directory layout assumed by this README:**
```
README.md
mupnit_clip_reid/
AP3D/
```

---

## Part 1: CLIP ReID Family

### Overview

Three benchmark settings are provided, each building on the previous:

| Setting | Description |
|---|---|
| **CLIP-ReID** | Pretrained CLIP ViT-L/14; image-to-image cosine similarity, no fine-tuning |
| **CLIP-ReID-Proj** | CLIP image embeddings passed through a learned projection layer |
| **CLIP-ReID-Proj-Text** | Projected image embeddings fused with CLIP text embeddings from player descriptions |

### Prerequisites

```bash
pip install torch torchvision tqdm pillow
pip install git+https://github.com/openai/CLIP.git
```

### Step 1 — Generate Text Embeddings

Text embeddings are generated from per-player annotations (jersey number, jersey color, skin color) stored in a JSON file. Each embedding captures a textual description of the form:
> *"A basketball player [name] with number [N], [color] jersey color and [skin] skin color"*

Edit the path variables at the bottom of the script:

```python
# in mupnit_clip_reid/mupnitreid_text_embeddings.py
data_root  = "/path/to/mupnitreid"        # contains train/, test/, and the JSON annotation file
save_root  = "/path/to/text_embeddings"   # output root; train/ and test/ subfolders are created automatically
```

Run:
```bash
python mupnit_clip_reid/mupnitreid_text_embeddings.py
```

Output: `.pt` files named `<player_id>_<game_id>.pt` under `text_embeddings/train/` and `text_embeddings/test/`.

### Step 2 — Train the Projection Layer

The projection layer is a small MLP (`Linear → BN → ReLU → Linear → BN`, 768→512→768) trained to bring CLIP image embeddings closer to the text embedding space via image–text contrastive loss on the training split.

Open and run:
```
mupnit_clip_reid/train_clip_projection_layer.ipynb
```

Set the training image directory and the text embedding directory (from Step 1) inside the notebook. The trained weights are saved as a `.pth` file (e.g., `projection_model.pth`).

### Step 3 — Evaluation

#### Setting 1: CLIP-ReID Baseline

Set variables at the top of the script:

```python
# in mupnit_clip_reid/pretrained_clipreid_eval_textque.py
QUERY_DIR      = "/path/to/mupnitreid/query"
GALLERY_DIR    = "/path/to/mupnitreid/gallery_test"
TEXT_EMBED_DIR = "/path/to/text_embeddings/test"
LOG_FILE       = "results_baseline.txt"
```

Run:
```bash
python mupnit_clip_reid/pretrained_clipreid_eval_textque.py
```

**Phase 1** reports the CLIP-ReID baseline (image query vs. image gallery, no text, no projection).

#### Settings 2 & 3: CLIP-ReID-Proj and CLIP-ReID-Proj-Text

Both settings are produced by a single script. Set variables at the top:

```python
# in mupnit_clip_reid/trained_proj_clipreid_eval_textque.py
QUERY_DIR        = "/path/to/mupnitreid/query"
GALLERY_DIR      = "/path/to/mupnitreid/gallery_test"
TEXT_EMBED_DIR   = "/path/to/text_embeddings/test"
LOG_FILE         = "results_proj.txt"
PROJ_WEIGHT_PATH = "/path/to/projection_model.pth"
```

Run:
```bash
python mupnit_clip_reid/trained_proj_clipreid_eval_textque.py
```

- **Phase 1**: CLIP-ReID-Proj — projected image query vs. projected image gallery.
- **Phase 2**: CLIP-ReID-Proj-Text — fused query (projected image + text, 50/50) vs. projected image gallery.

#### Setting 3 (Alternative): CLIP-ReID-Proj-Text with Optimal Fusion Weight

To sweep fusion weight α ∈ [0, 1] and report the best result:

```python
# in mupnit_clip_reid/clip_reid.py
QUERY_DIR        = "/path/to/mupnitreid/query"
GALLERY_DIR      = "/path/to/mupnitreid/gallery_test"
TEXT_EMBED_DIR   = "/path/to/text_embeddings/test"
PROJ_WEIGHT_PATH = "/path/to/projection_model.pth"
LOG_DIR          = "."
```

Run:
```bash
python mupnit_clip_reid/clip_reid.py
```

Computes `score = α × sim_proj_image + (1 − α) × sim_text` for 20 values of α; reports mAP and CMC at every α and identifies the optimal α.

---

### Supplementary Setting: Text-as-Gallery (Exploratory)

`mupnitreid_clip_train_test.py` evaluates a cross-modal scenario in which **text embeddings serve as the gallery** and image embeddings serve as queries. It also supports optional full-CLIP fine-tuning before evaluation. This setting is exploratory and is not part of the three main benchmark tiers.

Edit `log_dir` inside the script, then run:

```bash
# Pretrained CLIP, text-as-gallery evaluation
python mupnit_clip_reid/mupnitreid_clip_train_test.py \
    --mode pretrained \
    --test_dir         /path/to/mupnitreid/query \
    --gallery_test_dir /path/to/mupnitreid/gallery_test \
    --text_embed_dir   /path/to/text_embeddings

# Fine-tune CLIP first, then evaluate
python mupnit_clip_reid/mupnitreid_clip_train_test.py \
    --mode train \
    --train_dir        /path/to/mupnitreid/all_train \
    --test_dir         /path/to/mupnitreid/query \
    --gallery_test_dir /path/to/mupnitreid/gallery_test \
    --text_embed_dir   /path/to/text_embeddings \
    --num_epochs 5 --lr 1e-7
```

### Additional Ablation Scripts (`mupnit_clip_reid/clipreid_exploration_study/`)

The `clipreid_exploration_study/` subdirectory contains scripts used for deeper ablation experiments that go beyond the three main benchmark settings. They are not required to reproduce the primary results.

| Script | What it explores |
|---|---|
| `pretrained_clipreid_eval_textgal.py` | 6-phase ablation on pretrained CLIP: text fused into the **gallery** side, per-query fusion, per-player aggregated fusion, and mixed variants |
| `pretrained_clipreid_eval_meanemb_textque.py` | Pretrained CLIP with **player-level mean embedding** aggregation before matching |
| `trained_proj_clipreid_eval_textgal.py` | 10-phase ablation on projection-layer CLIP: same gallery-side and fuse-then-project variants as above |
| `trained_proj_clipreid_eval_meanemb_textque.py` | Projection-layer CLIP with **player-level mean embedding** aggregation |
| `mupnitreid_img_embeddings.py` | Offline image embedding pre-computation (early utility script; superseded by on-the-fly encoding in the main scripts) |

Each script follows the same path-variable convention as the main evaluation scripts. Set `QUERY_DIR`, `GALLERY_DIR`, `TEXT_EMBED_DIR`, and (where applicable) `PROJ_WEIGHT_PATH` at the top before running.

---

## Part 2: AP3D

AP3D uses appearance-preserving 3D convolutions for video-based person re-identification. The `AP3D/` folder contains a full implementation with native support for MuPNIT-ReID-light.

### Prerequisites

```bash
pip install torch torchvision h5py scipy
```

### Dataset Structure

AP3D expects the following layout under the dataset root:

```
<mupnitreid_root>/
    all_train/       # training images
    query/           # query images
    gallery_test/    # gallery images
```

### Training

```bash
cd AP3D
python train.py \
    --root <mupnitreid_root> \
    -d mupnit \
    --arch ap3dres50 \
    --gpu 0 \
    --save_dir log-mupnitreid-ap3d
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--root` | — | Path to the MuPNIT-ReID-light dataset root |
| `-d` | `mars` | Dataset name; use `mupnit` |
| `--arch` | `ap3dres50` | Backbone: `ap3dres50` or `ap3dnlres50` (with non-local block) |
| `--gpu` | `2,3` | GPU id(s) |
| `--save_dir` | `log-mars-ap3d` | Directory for checkpoints and training log |
| `--max_epoch` | `240` | Total training epochs |
| `--train_batch` | `32` | Batch size |
| `--lr` | `3e-4` | Initial learning rate |

### Evaluation

After training, evaluate with all frames per tracklet:

```bash
cd AP3D
python test-all.py \
    --root <mupnitreid_root> \
    -d mupnit \
    --arch ap3dres50 \
    --gpu 0 \
    --resume log-mupnitreid-ap3d
```

By default this loads `log-mupnitreid-ap3d/checkpoint_ep240.pth.tar`. To evaluate at a different epoch:
```bash
python test-all.py ... --resume log-mupnitreid-ap3d --test_epochs 120
```

Results (mAP, Rank-1/5/10/20) are printed to stdout and saved to `log-mupnitreid-ap3d/log_test.txt`.
