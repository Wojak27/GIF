import torch
from clip import clip
from PIL import Image
import numpy as np
import os
from trackers.ocsort_tracker.association_reid import *
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import torch.nn as nn

class CLIPExtraProjectionLayer(nn.Module):
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

class CLIPReID(torch.nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        super(CLIPReID, self).__init__()
        self.model_name = model_name
        self.model, self.preprocess, self.device = self.load_clip_model()
        
    
    def load_clip_model(self):
        """Load CLIP model and preprocessing transform"""
        print("Loading CLIP model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained(self.model_name).to(device)
        preprocess = CLIPProcessor.from_pretrained(self.model_name)
        print(f"Model loaded successfully on {device}")
        return model, preprocess, device

    def forward(self, image):
        try:
            inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
        except:
            print("Problem with inputs to clip")

        with torch.no_grad():
                embedding = self.model.get_image_features(**inputs).cpu()
        return embedding

    def inference(self, image, bboxes, save_path=None):
        self.model.eval()
        if save_path:
            os.makedirs(save_path, exist_ok=True)
        pil_images = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, [max(bbox[0], 0), max(bbox[1], 0), bbox[2], bbox[3]])
            crop = image[0, :, y1:y2, x1:x2]
            if crop.size(1) == 0 or crop.size(2) == 0:
                raise ValueError(f"Invalid crop dimensions: {crop.size()}")
            crop_np = crop.cpu().permute(1, 2, 0).numpy().astype(np.uint8)
            pil_img = Image.fromarray(crop_np).convert("RGB")
            if save_path:
                pil_img.save(os.path.join(save_path, f"{i}.png"))
            pil_images.append(pil_img)
        # Batch preprocess all crops at once.
        inputs = self.preprocess(images=pil_images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs).cpu()
        return features.squeeze(0) # shape: (#crops,dim)


import json
import torch
import clip
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

class IDEvaluator:
    def __init__(self, json_path: str, clip_model_name: str, labels_dir: str, cache_dir: Optional[str] = None, use_memory=False):
        """
        Initialize the IDEvaluator with game data, CLIP model, and label files.
        
        Args:
            json_path: Path to the JSON file containing game and player data
            clip_model_name: Name of the CLIP model to use
            labels_dir: Directory containing label files
            cache_dir: Optional directory for caching CLIP model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model
        if cache_dir:
            clip.clip._MODELS = {
                name: f"{cache_dir}/{name}" for name in clip.available_models()
            }
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        
        # Load labels
        self.labels = self._load_labels(labels_dir)
        
        # Load and process game data
        self.games_data = self._load_and_process_json(json_path)
        
        # Template for generating text prompts
        self.text_templates = {
            "jersey_numbers": "a clear view of jersey number {} on a basketball uniform",
            "jersey_colors": "a {} jersey, {} color",
            "ethnicities": "a {} basketball player"
        }
        
        # Generate and cache text embeddings for all categories
        self.category_embeddings = self._generate_category_embeddings()
        
        if use_memory:
            # Initialize memory tracking
            self.memory_dict = {}
        
    def reset_memory(self):
        """Reset the memory tracking dictionary."""
        self.memory_dict = {}
        
    def get_memory_stats(self, track_id: int) -> Dict[str, Any]:
        """
        Get statistics from the memory tracking for a specific track ID.
        
        Args:
            track_id: Track ID to get stats for
            
        Returns:
            Dictionary containing memory statistics
        """
        if track_id not in self.memory_dict:
            return {}
        
        track_memory = self.memory_dict[track_id]
        stats = {
            "most_common_number": None,
            "number_confidence": 0.0,
            "total_detections": 0
        }
        
        if "jersey_numbers" in track_memory and track_memory["jersey_numbers"]:
            number_counts = track_memory["jersey_numbers"]
            total_counts = sum(number_counts.values())
            most_common = max(number_counts.items(), key=lambda x: x[1])
            
            stats["most_common_number"] = most_common[0]
            stats["number_confidence"] = most_common[1] / total_counts if total_counts > 0 else 0.0
            stats["total_detections"] = total_counts
        
        return stats
    
    def evaluate_with_memory(self, 
                           player_embeddings: torch.Tensor, 
                           bbox_track_info: torch.Tensor, 
                           game_id: str, 
                           top_k: int = 3,
                           game_specific: bool = False) -> Dict[str, List[Tuple]]:
        """
        Evaluate player embeddings with memory tracking for improved number detection.
        
        Args:
            player_embeddings: Tensor of player embeddings (n, embedding_size)
            bbox_track_info: Tensor of bounding box and track information (n, 5)
            game_id: ID of the game to evaluate against
            top_k: Number of top matches to return for each category
            game_specific: If True, only compare against attributes present in the current game
            
        Returns:
            Dictionary with categories as keys and lists of tuples as values
        """
        if game_id not in self.games_data:
            raise ValueError(f"Game ID {game_id} not found in the dataset")
        
        # Normalize embeddings
        player_embeddings = player_embeddings / player_embeddings.norm(dim=-1, keepdim=True)
        results = {
            "jersey_numbers": [],
            "jersey_colors": [],
            "ethnicities": [],
        }
        
        # Get game-specific attributes if needed
        game_attributes = None
        if game_specific:
            game_attributes = self.games_data[game_id]
        
        for i in range(len(bbox_track_info)):
            bbox = bbox_track_info[i]
            x1, y1, x2, y2, track_id = bbox.tolist()
            track_id = int(track_id)
            
            # Initialize memory for this track if not exists
            if track_id not in self.memory_dict:
                self.memory_dict[track_id] = {"jersey_numbers": {}}
            
            # Convert bbox to x, y, w, h format
            w = x2 - x1
            h = y2 - y1
            
            base_result = [x1, y1, w, h, track_id]
            curr_embedding = player_embeddings[i:i+1]  # Keep dimension for matmul
            
            # Process each category
            for category, embeddings in self.category_embeddings.items():
                curr_result = base_result.copy()
                
                if category == "jersey_numbers":
                    # Apply memory-based processing for jersey numbers
                    similarities = torch.matmul(curr_embedding, embeddings.T)
                    similarities = torch.nn.functional.sigmoid(similarities-torch.mean(similarities))
                    
                    if game_specific:
                        # Filter for game-specific numbers
                        game_values = game_attributes[category]
                        valid_indices = []
                        for idx, label in enumerate(self.labels[category]):
                            if label in game_values:
                                valid_indices.append(idx)
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_similarities = similarities[0][valid_indices]
                            
                            # Update memory counts
                            for idx, val_idx in enumerate(valid_indices):
                                number = self.labels[category][val_idx.item()]
                                if number not in self.memory_dict[track_id]["jersey_numbers"]:
                                    self.memory_dict[track_id]["jersey_numbers"][number] = 0.0
                                
                                # Update memory if confidence is high
                                if filtered_similarities[idx].item() > 0.7:
                                    self.memory_dict[track_id]["jersey_numbers"][number] += 1
                            
                            # Get top k based on memory counts
                            memory_counts = [(num, count) for num, count in 
                                           self.memory_dict[track_id]["jersey_numbers"].items()]
                            memory_counts.sort(key=lambda x: x[1], reverse=True)
                            top_numbers = memory_counts[:top_k]
                            
                            # Convert back to indices
                            top_indices = [self.labels[category].index(num) for num, _ in top_numbers]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                        else:
                            continue
                    
                    else:
                        # Non-game-specific processing
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(embeddings)))
                        predictions = top_k_indices[0].tolist()
                        padded_predictions = predictions + [-1] * (5 - len(predictions))
                        curr_result.extend(padded_predictions[:5])
                
                elif category == "jersey_colors":
                    # Existing jersey colors processing (same as original evaluate method)
                    embeddings, color_indices = embeddings
                    if game_specific:
                        game_colors = game_attributes["jersey_colors"]
                        valid_indices = []
                        filtered_color_indices = []
                        
                        for idx, (color1_idx, color2_idx) in enumerate(color_indices):
                            color1 = self.labels["jersey_colors"][color1_idx]
                            color2 = self.labels["jersey_colors"][color2_idx]
                            if color1 in game_colors and color2 in game_colors:
                                valid_indices.append(idx)
                                filtered_color_indices.append((color1_idx, color2_idx))
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(filtered_color_indices)))
                            top_indices = [filtered_color_indices[idx.item()][0] for idx in top_k_indices[0]]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(color_indices)))
                        top_indices = [color_indices[idx.item()][0] for idx in top_k_indices[0]]
                        padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                        curr_result.extend(padded_predictions[:5])
                
                else:
                    # Existing processing for other categories
                    if game_specific:
                        game_values = game_attributes[category]
                        valid_indices = []
                        
                        for idx, label in enumerate(self.labels[category]):
                            if label in game_values:
                                valid_indices.append(idx)
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(valid_indices)))
                            top_indices = [valid_indices[idx.item()].item() for idx in top_k_indices[0]]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                        else:
                            continue
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(embeddings)))
                        predictions = top_k_indices[0].tolist()
                        padded_predictions = predictions + [-1] * (5 - len(predictions))
                        curr_result.extend(padded_predictions[:5])
                
                results[category].append(tuple(curr_result))
        
        return results

    def evaluate_with_memoryv1(self, 
                           player_embeddings: torch.Tensor, 
                           bbox_track_info: torch.Tensor, 
                           game_id: str, 
                           top_k: int = 3,
                           game_specific: bool = False) -> Dict[str, List[Tuple]]:
        """
        Evaluate player embeddings with memory tracking for improved number detection.
        
        Args:
            player_embeddings: Tensor of player embeddings (n, embedding_size)
            bbox_track_info: Tensor of bounding box and track information (n, 5)
            game_id: ID of the game to evaluate against
            top_k: Number of top matches to return for each category
            game_specific: If True, only compare against attributes present in the current game
            
        Returns:
            Dictionary with categories as keys and lists of tuples as values
        """
        if game_id not in self.games_data:
            raise ValueError(f"Game ID {game_id} not found in the dataset")
        
        # Normalize embeddings
        player_embeddings = player_embeddings / player_embeddings.norm(dim=-1, keepdim=True)
        results = {
            "jersey_numbers": [],
            "jersey_colors": [],
            "ethnicities": [],
        }
        
        # Get game-specific attributes if needed
        game_attributes = None
        if game_specific:
            game_attributes = self.games_data[game_id]
        
        for i in range(len(bbox_track_info)):
            bbox = bbox_track_info[i]
            x1, y1, x2, y2, track_id = bbox.tolist()
            track_id = int(track_id)
            
            # Initialize memory for this track if not exists
            if track_id not in self.memory_dict:
                self.memory_dict[track_id] = {"jersey_numbers": {}}
            
            # Convert bbox to x, y, w, h format
            w = x2 - x1
            h = y2 - y1
            
            base_result = [x1, y1, w, h, track_id]
            curr_embedding = player_embeddings[i:i+1]  # Keep dimension for matmul
            
            # Process each category
            for category, embeddings in self.category_embeddings.items():
                curr_result = base_result.copy()
                
                if category == "jersey_numbers":
                    # Apply memory-based processing for jersey numbers
                    similarities = torch.matmul(curr_embedding, embeddings.T)
                    softmax_similarities = torch.nn.functional.softmax(100.0 * similarities, dim=-1)
                    
                    if game_specific:
                        # Filter for game-specific numbers
                        game_values = game_attributes[category]
                        valid_indices = []
                        for idx, label in enumerate(self.labels[category]):
                            if label in game_values:
                                valid_indices.append(idx)
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_similarities = softmax_similarities[0][valid_indices]
                            
                            # Update memory counts
                            for idx, val_idx in enumerate(valid_indices):
                                number = self.labels[category][val_idx.item()]
                                if number not in self.memory_dict[track_id]["jersey_numbers"]:
                                    self.memory_dict[track_id]["jersey_numbers"][number] = 0.0
                                
                                # Update memory if confidence is high
                                if filtered_similarities[idx].item() > 0.7:
                                    self.memory_dict[track_id]["jersey_numbers"][number] += 1
                            
                            # Get top k based on memory counts
                            memory_counts = [(num, count) for num, count in 
                                           self.memory_dict[track_id]["jersey_numbers"].items()]
                            memory_counts.sort(key=lambda x: x[1], reverse=True)
                            top_numbers = memory_counts[:top_k]
                            
                            # Convert back to indices
                            top_indices = [self.labels[category].index(num) for num, _ in top_numbers]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                        else:
                            continue
                    
                    else:
                        # Non-game-specific processing
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(embeddings)))
                        predictions = top_k_indices[0].tolist()
                        padded_predictions = predictions + [-1] * (5 - len(predictions))
                        curr_result.extend(padded_predictions[:5])
                
                elif category == "jersey_colors":
                    # Existing jersey colors processing (same as original evaluate method)
                    embeddings, color_indices = embeddings
                    if game_specific:
                        game_colors = game_attributes["jersey_colors"]
                        valid_indices = []
                        filtered_color_indices = []
                        
                        for idx, (color1_idx, color2_idx) in enumerate(color_indices):
                            color1 = self.labels["jersey_colors"][color1_idx]
                            color2 = self.labels["jersey_colors"][color2_idx]
                            if color1 in game_colors and color2 in game_colors:
                                valid_indices.append(idx)
                                filtered_color_indices.append((color1_idx, color2_idx))
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(filtered_color_indices)))
                            top_indices = [filtered_color_indices[idx.item()][0] for idx in top_k_indices[0]]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(color_indices)))
                        top_indices = [color_indices[idx.item()][0] for idx in top_k_indices[0]]
                        padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                        curr_result.extend(padded_predictions[:5])
                
                else:
                    # Existing processing for other categories
                    if game_specific:
                        game_values = game_attributes[category]
                        valid_indices = []
                        
                        for idx, label in enumerate(self.labels[category]):
                            if label in game_values:
                                valid_indices.append(idx)
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(valid_indices)))
                            top_indices = [valid_indices[idx.item()].item() for idx in top_k_indices[0]]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                        else:
                            continue
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(embeddings)))
                        predictions = top_k_indices[0].tolist()
                        padded_predictions = predictions + [-1] * (5 - len(predictions))
                        curr_result.extend(padded_predictions[:5])
                
                results[category].append(tuple(curr_result))
        
        return results
    
    def evaluate_with_memoryv2(self, 
                        player_embeddings: torch.Tensor, 
                        bbox_track_info: torch.Tensor, 
                        game_id: str, 
                        top_k: int = 3,
                        game_specific: bool = False) -> Dict[str, List[Tuple]]:
        """
        Evaluate player embeddings with memory tracking for all categories.
        
        Args:
            player_embeddings: Tensor of player embeddings (n, embedding_size)
            bbox_track_info: Tensor of bounding box and track information (n, 5)
            game_id: ID of the game to evaluate against
            top_k: Number of top matches to return for each category
            game_specific: If True, only compare against attributes present in the current game
            
        Returns:
            Dictionary with categories as keys and lists of tuples as values
        """
        if game_id not in self.games_data:
            raise ValueError(f"Game ID {game_id} not found in the dataset")
        
        # Normalize embeddings
        player_embeddings = player_embeddings / player_embeddings.norm(dim=-1, keepdim=True)
        results = {
            "jersey_numbers": [],
            "jersey_colors": [],
            "ethnicities": [],
        }
        
        # Get game-specific attributes if needed
        game_attributes = None
        if game_specific:
            game_attributes = self.games_data[game_id]
        
        
        for i in range(len(bbox_track_info)):
            bbox = bbox_track_info[i]
            x1, y1, x2, y2, track_id = bbox.tolist()
            track_id = int(track_id)
            
            # Initialize memory for this track if not exists
            if track_id not in self.memory_dict:
                self.memory_dict[track_id] = {
                    "jersey_numbers": {},
                    "jersey_colors": {},
                    "ethnicities": {}
                }
            
            # Convert bbox to x, y, w, h format
            w = x2 - x1
            h = y2 - y1
            
            base_result = [x1, y1, w, h, track_id]
            curr_embedding = player_embeddings[i:i+1]  # Keep dimension for matmul
            
            # Process each category
            for category, embeddings in self.category_embeddings.items():
                curr_result = base_result.copy()
                
                if category == "jersey_numbers":
                    # Apply memory-based processing for jersey numbers
                    similarities = torch.matmul(curr_embedding, embeddings.T)
                    softmax_similarities = torch.nn.functional.softmax(100.0 * similarities, dim=-1)
                    
                    if game_specific:
                        # Filter for game-specific numbers
                        game_values = game_attributes[category]
                        valid_indices = []
                        for idx, label in enumerate(self.labels[category]):
                            if label in game_values:
                                valid_indices.append(idx)
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_similarities = softmax_similarities[0][valid_indices]
                            
                            # Update memory counts
                            for idx, val_idx in enumerate(valid_indices):
                                value = self.labels[category][val_idx.item()]
                                if value not in self.memory_dict[track_id]["jersey_numbers"]:
                                    self.memory_dict[track_id]["jersey_numbers"][value] = 0.0
                                
                                # Update memory if confidence is high
                                if filtered_similarities[idx].item() > 0.7:
                                    self.memory_dict[track_id]["jersey_numbers"][value] += 1
                            
                            # Get top k based on memory counts
                            memory_counts = [(num, count) for num, count in 
                                           self.memory_dict[track_id]["jersey_numbers"].items()]
                            memory_counts.sort(key=lambda x: x[1], reverse=True)
                            top_values = memory_counts[:top_k]
                            
                            # Convert back to indices
                            top_indices = [self.labels[category].index(num) for num, _ in top_values]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                        else:
                            continue
                    
                    else:
                        # Non-game-specific processing
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(embeddings)))
                        predictions = top_k_indices[0].tolist()
                        padded_predictions = predictions + [-1] * (5 - len(predictions))
                        curr_result.extend(padded_predictions[:5])
                
                elif category == "jersey_colors":
                    # Existing jersey colors processing (same as original evaluate method)
                    embeddings, color_indices = embeddings
                    if game_specific:
                        game_colors = game_attributes[category]
                        valid_indices = []
                        filtered_color_indices = []
                        
                        for idx, (color1_idx, color2_idx) in enumerate(color_indices):
                            color1 = self.labels[category][color1_idx]
                            color2 = self.labels[category][color2_idx]
                            if color1 in game_colors and color2 in game_colors:
                                valid_indices.append(idx)
                                filtered_color_indices.append((color1_idx, color2_idx))
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            similarities = torch.nn.functional.softmax(100.0 * similarities, dim=-1)[0]
                            
                            # Update memory counts
                            for idx, val_idx in enumerate(valid_indices):
                                value = color_indices[val_idx][0]
                                value = self.labels[category][value]
                                if value not in self.memory_dict[track_id][category]:
                                    self.memory_dict[track_id][category][value] = 0.0
                                
                                # Update memory if confidence is high
                                if similarities[idx].item() > 0.7:
                                    self.memory_dict[track_id][category][value] += 1
                            
                            # Get top k based on memory counts
                            memory_counts = [(color, count) for color, count in 
                                           self.memory_dict[track_id][category].items()]
                            memory_counts.sort(key=lambda x: x[1], reverse=True)
                            top_values = memory_counts[:top_k]
                            
                            # Convert back to indices
                            top_indices = [self.labels[category].index(color) for color, _ in top_values]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(color_indices)))
                        top_indices = [color_indices[idx.item()][0] for idx in top_k_indices[0]]
                        padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                        curr_result.extend(padded_predictions[:5])
                
                else:
                    # Existing processing for other categories
                    if game_specific:
                        game_values = game_attributes[category]
                        valid_indices = []
                        
                        for idx, label in enumerate(self.labels[category]):
                            if label in game_values:
                                valid_indices.append(idx)
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            similarities = torch.nn.functional.softmax(100.0 * similarities, dim=-1)[0]
                            
                            # Update memory counts
                            for idx, val_idx in enumerate(valid_indices):
                                value = val_idx.item()
                                value = self.labels[category][value]
                                if value not in self.memory_dict[track_id][category]:
                                    self.memory_dict[track_id][category][value] = 0.0
                                
                                # Update memory if confidence is high
                                if similarities[idx].item() > 0.7:
                                    self.memory_dict[track_id][category][value] += 1
                            
                            # Get top k based on memory counts
                            memory_counts = [(color, count) for color, count in 
                                           self.memory_dict[track_id][category].items()]
                            memory_counts.sort(key=lambda x: x[1], reverse=True)
                            top_values = memory_counts[:top_k]
                            
                            # Convert back to indices
                            top_indices = [self.labels[category].index(value) for value, _ in top_values]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                        else:
                            continue
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(embeddings)))
                        predictions = top_k_indices[0].tolist()
                        padded_predictions = predictions + [-1] * (5 - len(predictions))
                        curr_result.extend(padded_predictions[:5])
                
                results[category].append(tuple(curr_result))
        
        return results
    def evaluate_with_memoryv3(self, 
                        player_embeddings: torch.Tensor, 
                        bbox_track_info: torch.Tensor, 
                        game_id: str, 
                        top_k: int = 3,
                        game_specific: bool = False,
                        decay_factor: float = 0.3) -> Dict[str, List[Tuple]]:
        """
        Evaluate player embeddings with memory tracking and time-based decay.
        
        Args:
            player_embeddings: Tensor of player embeddings (n, embedding_size)
            bbox_track_info: Tensor of bounding box and track information (n, 5)
            game_id: ID of the game to evaluate against
            top_k: Number of top matches to return for each category
            game_specific: If True, only compare against attributes present in the current game
            decay_factor: Factor for exponential decay of old predictions (0-1)
        """
        if game_id not in self.games_data:
            raise ValueError(f"Game ID {game_id} not found in the dataset")
        
        # Initialize memory structures if not exists
        if not hasattr(self, 'memory_dict'):
            self.memory_dict = {}
        if not hasattr(self, 'last_update_frame'):
            self.last_update_frame = {}
        if not hasattr(self, 'current_frame'):
            self.current_frame = 0
        
        # Increment frame counter
        self.current_frame += 1
        
        # Normalize embeddings
        player_embeddings = player_embeddings / player_embeddings.norm(dim=-1, keepdim=True)
        results = {
            "jersey_numbers": [],
            "jersey_colors": [],
            "ethnicities": [],
        }
        
        # Get game-specific attributes if needed
        game_attributes = None
        if game_specific:
            game_attributes = self.games_data[game_id]
        
        for i in range(len(bbox_track_info)):
            bbox = bbox_track_info[i]
            x1, y1, x2, y2, track_id = bbox.tolist()
            track_id = int(track_id)
            
            # Initialize memory for this track if not exists
            if track_id not in self.memory_dict:
                self.memory_dict[track_id] = {
                    "jersey_numbers": {},
                    "jersey_colors": {},
                    "ethnicities": {},
                    "first_seen_frame": self.current_frame
                }
                self.last_update_frame[track_id] = {}
            
            # Convert bbox
            w = x2 - x1
            h = y2 - y1
            base_result = [x1, y1, w, h, track_id]
            curr_embedding = player_embeddings[i:i+1]
            
            for category, embeddings in self.category_embeddings.items():
                curr_result = base_result.copy()
                
                # Apply decay to existing predictions
                if track_id in self.memory_dict and category in self.memory_dict[track_id]:
                    for value in list(self.memory_dict[track_id][category].keys()):
                        frames_passed = self.current_frame - self.last_update_frame[track_id].get(value, 0)
                        decay = decay_factor ** frames_passed
                        self.memory_dict[track_id][category][value] *= decay
                
                if category == "jersey_numbers":
                    similarities = torch.matmul(curr_embedding, embeddings.T)
                    similarities = torch.nn.functional.sigmoid(similarities-torch.mean(similarities))
                    
                    if game_specific:
                        game_values = game_attributes[category]
                        valid_indices = []
                        for idx, label in enumerate(self.labels[category]):
                            if label in game_values:
                                valid_indices.append(idx)
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            similarities = similarities[0][valid_indices]
                            
                            # Update memory with time tracking
                            for idx, val_idx in enumerate(valid_indices):
                                value = self.labels[category][val_idx.item()]
                                confidence = similarities[idx].item()
                                
                                if confidence > 0.5:  # High confidence threshold
                                    if value not in self.memory_dict[track_id]["jersey_numbers"]:
                                        self.memory_dict[track_id]["jersey_numbers"][value] = 0.0
                                    
                                    # Update memory and timestamp
                                    self.memory_dict[track_id]["jersey_numbers"][value] += confidence
                                    self.last_update_frame[track_id][value] = self.current_frame
                            
                            # Get weighted predictions
                            memory_counts = []
                            for num, count in self.memory_dict[track_id]["jersey_numbers"].items():
                                frames_since_update = self.current_frame - self.last_update_frame[track_id].get(num, 0)
                                time_weight = decay_factor ** frames_since_update
                                weighted_count = count * time_weight
                                memory_counts.append((num, weighted_count))
                            
                            memory_counts.sort(key=lambda x: x[1], reverse=True)
                            top_values = memory_counts[:top_k]
                            
                            top_indices = [self.labels[category].index(num) for num, _ in top_values]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                        else:
                            continue
                    else:
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(embeddings)))
                        predictions = top_k_indices[0].tolist()
                        padded_predictions = predictions + [-1] * (5 - len(predictions))
                        curr_result.extend(padded_predictions[:5])
                
                elif category == "jersey_colors":
                    embeddings, color_indices = embeddings
                    if game_specific:
                        game_colors = game_attributes[category]
                        valid_indices = []
                        filtered_color_indices = []
                        
                        for idx, (color1_idx, color2_idx) in enumerate(color_indices):
                            color1 = self.labels[category][color1_idx]
                            color2 = self.labels[category][color2_idx]
                            if color1 in game_colors and color2 in game_colors:
                                valid_indices.append(idx)
                                filtered_color_indices.append((color1_idx, color2_idx))
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            similarities = torch.nn.functional.sigmoid(similarities-torch.mean(similarities)).squeeze()
                            
                            # Update memory with time tracking
                            for idx, val_idx in enumerate(valid_indices):
                                color1_idx = color_indices[val_idx][0]
                                value = self.labels[category][color1_idx]
                                confidence = similarities[idx].item()
                                
                                if confidence > 0.5:
                                    if value not in self.memory_dict[track_id][category]:
                                        self.memory_dict[track_id][category][value] = 0.0
                                    
                                    # Update memory and timestamp
                                    self.memory_dict[track_id][category][value] += confidence
                                    self.last_update_frame[track_id][value] = self.current_frame
                            
                            # Get weighted predictions
                            memory_counts = []
                            for color, count in self.memory_dict[track_id][category].items():
                                frames_since_update = self.current_frame - self.last_update_frame[track_id].get(color, 0)
                                time_weight = decay_factor ** frames_since_update
                                weighted_count = count * time_weight
                                memory_counts.append((color, weighted_count))
                            
                            memory_counts.sort(key=lambda x: x[1], reverse=True)
                            top_values = memory_counts[:top_k]
                            
                            top_indices = [self.labels[category].index(color) for color, _ in top_values]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                        else:
                            continue
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(color_indices)))
                        top_indices = [color_indices[idx.item()][0] for idx in top_k_indices[0]]
                        padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                        curr_result.extend(padded_predictions[:5])
                
                else:  # ethnicities
                    if game_specific:
                        game_values = game_attributes[category]
                        valid_indices = []
                        
                        for idx, label in enumerate(self.labels[category]):
                            if label in game_values:
                                valid_indices.append(idx)
                        
                        if valid_indices:
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            similarities = torch.nn.functional.sigmoid(similarities-torch.mean(similarities)).squeeze()
                            
                            # Update memory with time tracking
                            for idx, val_idx in enumerate(valid_indices):
                                value = self.labels[category][val_idx.item()]
                                confidence = similarities[idx].item()
                                
                                if confidence > 0.5:
                                    if value not in self.memory_dict[track_id][category]:
                                        self.memory_dict[track_id][category][value] = 0.0
                                    
                                    # Update memory and timestamp
                                    self.memory_dict[track_id][category][value] += confidence
                                    self.last_update_frame[track_id][value] = self.current_frame
                            
                            # Get weighted predictions
                            memory_counts = []
                            for value, count in self.memory_dict[track_id][category].items():
                                frames_since_update = self.current_frame - self.last_update_frame[track_id].get(value, 0)
                                time_weight = decay_factor ** frames_since_update
                                weighted_count = count * time_weight
                                memory_counts.append((value, weighted_count))
                            
                            memory_counts.sort(key=lambda x: x[1], reverse=True)
                            top_values = memory_counts[:top_k]
                            
                            top_indices = [self.labels[category].index(value) for value, _ in top_values]
                            padded_predictions = top_indices + [-1] * (5 - len(top_indices))
                            curr_result.extend(padded_predictions[:5])
                        else:
                            continue
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(embeddings)))
                        predictions = top_k_indices[0].tolist()
                        padded_predictions = predictions + [-1] * (5 - len(predictions))
                        curr_result.extend(padded_predictions[:5])
                
                results[category].append(tuple(curr_result))
        
        return results
    

    def _load_labels(self, labels_dir: str) -> Dict[str, List[str]]:
        """
        Load label files for each category.
        """
        labels_dir = Path(labels_dir)
        categories = ['jersey_numbers', 'jersey_colors', 'ethnicities', 'player_ids']
        labels = {}
        
        for category in categories:
            with open(labels_dir / f'{category}.txt', 'r') as f:
                labels[category] = [line.strip() for line in f.readlines()]
        
        return labels

    def _load_and_process_json(self, json_path: str) -> Dict:
        """
        Load and transform JSON data into the required attribute dictionary format.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        attributes_dict = {}
        for game_id, game_info in data.items():
            jersey_numbers = []
            ethnicities = set()
            player_ids = []
            
            # Extract player attributes
            for player_info in game_info['players'].values():
                jersey_numbers.append(str(player_info['jersey_number']))
                ethnicities.add(player_info['ethnicity'].lower())
                player_ids.append(str(player_info['player_id']))
            
            # Get jersey colors
            jersey_colors = [game_info['colorA'].lower(), game_info['colorB'].lower()]
            
            attributes_dict[game_id] = {
                "jersey_numbers": sorted(list(set(jersey_numbers))),
                "jersey_colors": sorted(list(set(jersey_colors))),
                "ethnicities": sorted(list(set(ethnicities))),
                "player_ids": sorted(list(set(player_ids)))
            }
        
        return attributes_dict

    def _generate_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        """
        Generate CLIP text embeddings for a list of texts.
        """
        text_tokens = clip.tokenize(texts).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.model.encode_text(text_tokens)
        
        return text_embeddings

    def _generate_category_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Generate embeddings for all labels in each category.
        """
        category_embeddings = {}
        
        # Generate embeddings for jersey numbers
        number_texts = [self.text_templates["jersey_numbers"].format(num) 
                       for num in self.labels["jersey_numbers"]]
        category_embeddings["jersey_numbers"] = self._generate_text_embeddings(number_texts)
        
        # Generate embeddings for jersey colors (all possible combinations)
        color_texts = []
        color_indices = []  # To keep track of which colors were used
        for i, color1 in enumerate(self.labels["jersey_colors"]):
            for j, color2 in enumerate(self.labels["jersey_colors"]):
                if i != j:
                    color_texts.append(self.text_templates["jersey_colors"].format(color1, color2))
                    color_indices.append((i, j))
        category_embeddings["jersey_colors"] = (
            self._generate_text_embeddings(color_texts), 
            color_indices
        )
        
        # Generate embeddings for ethnicities
        ethnicity_texts = [self.text_templates["ethnicities"].format(ethnicity) 
                          for ethnicity in self.labels["ethnicities"]]
        category_embeddings["ethnicities"] = self._generate_text_embeddings(ethnicity_texts)
        
        return category_embeddings

    def evaluate(self, 
                player_embeddings: torch.Tensor, 
                bbox_track_info: torch.Tensor, 
                game_id: str, 
                top_k: int = 3,
                game_specific: bool = False) -> Dict[str, List[Tuple]]:
        """
        Evaluate player embeddings and return results grouped by categories.
        
        Args:
            player_embeddings: Tensor of player embeddings (n, embedding_size)
            bbox_track_info: Tensor of bounding box and track information (n, 5)
            game_id: ID of the game to evaluate against
            top_k: Number of top matches to return for each category
            game_specific: If True, only compare against attributes present in the current game
            
        Returns:
            Dictionary with categories as keys and lists of tuples as values
            Format for each tuple: (x, y, w, h, track_id, result1, result2, ...)
        """
        if game_id not in self.games_data:
            raise ValueError(f"Game ID {game_id} not found in the dataset")
        
        # Normalize embeddings
        player_embeddings = player_embeddings / player_embeddings.norm(dim=-1, keepdim=True)
        results = {
            "jersey_numbers": [],
            "jersey_colors": [],
            "ethnicities": [],
        }
        
        # Get game-specific attributes if needed
        game_attributes = None
        if game_specific:
            game_attributes = self.games_data[game_id]
        
        for i in range(len(bbox_track_info)):
            bbox = bbox_track_info[i]
            x1, y1, x2, y2, track_id = bbox.tolist()
            
            # Convert bbox to x, y, w, h format
            w = x2 - x1
            h = y2 - y1
            
            base_result = [x1, y1, w, h, int(track_id)]
            curr_embedding = player_embeddings[i:i+1]  # Keep dimension for matmul
            
            # Process each category
            for category, embeddings in self.category_embeddings.items():
                curr_result = base_result.copy()
                
                if category == "jersey_colors":
                    embeddings, color_indices = embeddings
                    
                    if game_specific:
                        # Filter color combinations for game-specific colors
                        game_colors = game_attributes["jersey_colors"]
                        valid_indices = []
                        valid_embeddings = []
                        filtered_color_indices = []
                        
                        for idx, (color1_idx, color2_idx) in enumerate(color_indices):
                            color1 = self.labels["jersey_colors"][color1_idx]
                            color2 = self.labels["jersey_colors"][color2_idx]
                            if color1 in game_colors and color2 in game_colors:
                                valid_indices.append(idx)
                                filtered_color_indices.append((color1_idx, color2_idx))
                        
                        if valid_indices:  # If we found valid combinations
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            similarities = torch.nn.functional.softmax(100.0 * similarities, dim=-1)
                            top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(filtered_color_indices)))
                            top_indices = [filtered_color_indices[idx.item()][0] for idx in top_k_indices[0] if similarities[0][idx.item()].item() > 0.7]
                            # Inside your evaluate method, when creating the results:
                            predictions = top_indices  # Your original predictions
                            padded_predictions = predictions + [-1] * (5 - len(predictions))  # Pad with -1s
                            curr_result.extend(padded_predictions[:5])  # Take only first 5 elements
                        else:
                            continue  # Skip if no valid combinations
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(color_indices)))
                        top_indices = [color_indices[idx.item()][0] for idx in top_k_indices[0]]
                        # Inside your evaluate method, when creating the results:
                        predictions = top_indices  # Your original predictions
                        padded_predictions = predictions + [-1] * (5 - len(predictions))  # Pad with -1s
                        curr_result.extend(padded_predictions[:5])  # Take only first 5 elements
                    
                else:
                    if game_specific:
                        # Get game-specific labels and their indices
                        game_values = game_attributes[category]
                        valid_indices = []
                        
                        for idx, label in enumerate(self.labels[category]):
                            if label in game_values:
                                valid_indices.append(idx)
                        
                        if valid_indices:  # If we found valid indices
                            valid_indices = torch.tensor(valid_indices, device=self.device)
                            filtered_embeddings = embeddings[valid_indices]
                            similarities = torch.matmul(curr_embedding, filtered_embeddings.T)
                            similarities = torch.nn.functional.softmax(100.0 * similarities, dim=-1)
                            top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(valid_indices)))
                            # Map back to original indices
                            top_indices = [valid_indices[idx.item()].item() for idx in top_k_indices[0] if similarities[0][idx.item()].item() > 0.7]
                            # Inside your evaluate method, when creating the results:
                            predictions = top_k_indices[0].tolist()  # Your original predictions
                            padded_predictions = predictions + [-1] * (5 - len(predictions))  # Pad with -1s
                            curr_result.extend(padded_predictions[:5])  # Take only first 5 elements
                        else:
                            continue  # Skip if no valid indices
                    else:
                        similarities = torch.matmul(curr_embedding, embeddings.T)
                        top_k_values, top_k_indices = similarities.topk(k=min(top_k, len(embeddings)))
                         # Inside your evaluate method, when creating the results:
                        predictions = top_k_indices[0].tolist()  # Your original predictions
                        padded_predictions = predictions + [-1] * (5 - len(predictions))  # Pad with -1s
                        curr_result.extend(padded_predictions[:5])  # Take only first 5 elements
                
                results[category].append(tuple(curr_result))
        
        return results

    def get_label(self, category: str, index: int) -> str:
        """
        Get the label string for a given category and index.
        """
        return self.labels[category][index]