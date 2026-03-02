import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

from ap3d_reid.models.ResNet import AP3DResNet50

# Import your AP3D reID model.
# Here we assume you are using the AP3DResNet50 defined in your MEVID/AP3D code.
# Adjust the import path as needed.

class AP3DReID(nn.Module):
    def __init__(self, ckpt_path=None, device="cuda"):
        """
        AP3DReID wrapper for feature extraction.
        Args:
            ckpt_path (str): Path to the trained AP3D checkpoint.
            device (str): 'cuda' or 'cpu'
        """
        super(AP3DReID, self).__init__()
        # For inference, we only need to extract features.
        # We create the AP3DResNet50 model. (num_classes can be dummy here.)
        print("Loading AP3D model...")
        self.model = AP3DResNet50(num_classes=118)  # You may need to modify if your classifier is not used.
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=device)
            # Assume your checkpoint is saved with key 'state_dict'
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        
        # Define the preprocessing transform.
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),  # Use the same size as used during AP3D training.
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images):
        """
        Process a single PIL image. AP3D expects a 5D input (B, C, T, H, W).
        Here, for a single frame, we set T=1.
        """
        # Apply preprocessing to the image.
        x = [self.transform(img) for img in images]
        x = torch.stack(x)
        x = x.unsqueeze(2)         # add time dimension → (C, 1, H, W)
        x = x.to(self.device)
        with torch.no_grad():
            feat = self.model(x)   # Model output shape: (1, feat_dim, T, H', W')
            # print("feat shape:", feat.shape)
            # # In your test code you perform a mean over the temporal dimension.
            # feat = feat.mean(2)    # Now shape: (1, feat_dim, H', W')
            # # Optionally, perform global pooling (e.g., adaptive max/avg pooling) to get a feature vector.
            # feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1,1))
            # feat = feat.view(1, -1)
            feat = feat.squeeze()
        return feat.cpu()

    def inference(self, image, bboxes, save_path=None):
        """
        Given a PIL image and a list of bounding boxes, return feature embeddings.
        Args:
            image (PIL.Image): The full image.
            bboxes (list): List of bounding boxes [x1, y1, x2, y2].
        Returns:
            torch.Tensor: Feature embeddings for each cropped region.
        """
        # features = []
        # for bbox in bboxes:
        #     x1, y1, x2, y2 = map(int, bbox)
        #     crop = image.crop((x1, y1, x2, y2))
        #     feat = self.forward(crop)
        #     features.append(feat)
        # if features:
        #     return torch.cat(features, dim=0)
        # else:
        #     return None
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
        features = self(pil_images).cpu()
        return features.squeeze(0) # shape: (#crops,dim)
