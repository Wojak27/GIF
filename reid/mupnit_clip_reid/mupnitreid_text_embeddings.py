#!/usr/bin/env python
import os
import json
import glob
import re
import torch
import clip  # ensure you have installed CLIP from https://github.com/openai/CLIP
from PIL import Image

# Set these flags to choose jersey color handling and which part of the player's name to use.
USE_FULL_JERSEY_COLOR = False  # True: use all jersey colors; False: use only the first.
USE_FIRST_NAME = False         # True: use first name; False: use last name.

def main():
    # Device setting
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the pretrained CLIP model (ViT-L/14)
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()  # only encoding text

    # Define paths
    data_root = "" # MuPNITReID dataset root directory
    train_dir = os.path.join(data_root, "train")
    test_dir  = os.path.join(data_root, "test")
    json_path = os.path.join(data_root, "global_local_player_descriptions_final_v5.json")
    save_root = "" # Directory to save text embeddings

    # Create output subfolders if not exist
    for folder in ["train", "test", "val_only"]:
        os.makedirs(os.path.join(save_root, folder), exist_ok=True)

    # Function to extract player IDs from a directory by matching the first 4-digit pattern in filenames.
    def extract_player_ids(directory):
        pattern = re.compile(r"(\d{4})")
        ids = set()
        for img_file in glob.glob(os.path.join(directory, "*.jpg")):
            basename = os.path.basename(img_file)
            match = pattern.search(basename)
            if match:
                ids.add(match.group(1))
        return ids

    train_ids = extract_player_ids(train_dir)
    test_ids  = extract_player_ids(test_dir)

    # Load JSON information
    with open(json_path, "r") as f:
        player_info = json.load(f)

    # (Optional) Mapping from normalized player name to ethnicity; leave empty if not available.
    name_to_ethnicity = {}  # e.g. {"al horford": "African-American", ...}

    # Function to generate a text embedding given a text string.
    def get_text_embedding(text):
        text_tokens = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu()

    # Loop over players in the JSON.
    for player_key, info in player_info.items():
        # player_key is like "player_0001"; extract "0001"
        pid_str = player_key.split("_")[-1]
        
        # Determine the save folder based on whether this player appears in train or test.
        if pid_str in train_ids:
            save_folder = "train"
        elif pid_str in test_ids:
            save_folder = "test"
        else:
            save_folder = "val_only"
        
        # Get player's basic info
        full_name = info.get("player_name", "")
        skin_color  = info.get("skin_color", "")
        normalized_name = full_name.lower().strip()
        ethnicity = name_to_ethnicity.get(normalized_name, skin_color)

        # Extract desired part of the player's name based on USE_FIRST_NAME flag.
        # If the full name contains spaces, we split and pick the appropriate token.
        name_tokens = full_name.split()
        if len(name_tokens) == 0:
            display_name = ""
        elif USE_FIRST_NAME:
            display_name = name_tokens[0]
        else:
            display_name = name_tokens[-1]
        
        # Define a function to process an appearance list (either long or short)
        def process_appearances(appearances):
            for app in appearances:
                jersey_number = app.get("jersey_number", "N/A")
                game_ids_dict = app.get("game_ids", {})
                for game_id, game_info in game_ids_dict.items():
                    jersey_color_list = game_info.get("jersey_color", [])
                    if jersey_color_list:
                        if USE_FULL_JERSEY_COLOR:
                            # Join all colors with " or "
                            jersey_color = " or ".join(jersey_color_list)
                        else:
                            # Use only the first color
                            jersey_color = jersey_color_list[0]
                    else:
                        jersey_color = "unknown"

                    text = (f"A basketball player {display_name} with number {jersey_number}, "
                            f"{jersey_color} jersey color and {ethnicity} skin color")
                    # text = (f"A basketball with number {jersey_number}, "
                    #         f"{jersey_color} jersey color and {ethnicity} skin color")
                    print(f"Generating embedding for {pid_str}_{game_id}: {text}")
                    embedding = get_text_embedding(text)
                    save_path = os.path.join(save_root, save_folder, f"{pid_str}_{game_id}.pt")
                    torch.save(embedding, save_path)
        
        # Process both long and short video appearances.
        long_apps = info.get("long_video_appearances", [])
        short_apps = info.get("short_video_appearances", [])
        
        process_appearances(long_apps)
        process_appearances(short_apps)

if __name__ == "__main__":
    main()

