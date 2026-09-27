import os
import json
import cv2
import pandas as pd
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class AppearanceFeatureExtractor:
    def __init__(self):
        # Load lightweight CPU-friendly model
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Remove the classification head, keep only features
        self.model.classifier = torch.nn.Identity()
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return None
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        input_tensor = self.transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model(input_tensor)
            
        features = features.squeeze().numpy()
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
        
    def process_case(self, case_dir: str):
        """Extract representative track crops and compute embeddings."""
        evidence_dir = os.path.join(case_dir, "evidence")
        csv_files = [f for f in os.listdir(evidence_dir) if f.endswith("_tracking.csv")]
        
        all_embeddings = {}
        crops_dir = os.path.join(case_dir, "output", "crops")
        os.makedirs(crops_dir, exist_ok=True)
        
        for csv_file in csv_files:
            camera_id = csv_file.split("_")[0]
            
            # Find the actual video file
            videos_dir = os.path.join(case_dir, "videos")
            actual_video_path = None
            for vf in os.listdir(videos_dir):
                if vf.startswith(f"{camera_id}_"):
                    actual_video_path = os.path.join(videos_dir, vf)
                    break
                    
            if not actual_video_path:
                print(f"Warning: No video found for {camera_id}")
                continue
                
            csv_full_path = os.path.join(evidence_dir, csv_file)
            if not os.path.exists(csv_full_path) or os.path.getsize(csv_full_path) == 0:
                continue
            try:
                df = pd.read_csv(csv_full_path)
            except Exception:
                continue
            if df.empty:
                continue
                
            grouped = df.groupby("track_id")
            
            cap = cv2.VideoCapture(actual_video_path)
            if not cap.isOpened():
                print(f"Warning: Could not open {actual_video_path}")
                continue
                
            for track_id, track_data in grouped:
                # Sample up to 5 observations
                if len(track_data) > 5:
                    samples = track_data.iloc[np.linspace(0, len(track_data) - 1, 5).astype(int)]
                else:
                    samples = track_data
                    
                features_list = []
                for idx, row in samples.iterrows():
                    frame_number = int(row["frame"])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                        
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
                    x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w))
                    y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h))
                    
                    if x2 <= x1 or y2 <= y1:
                        continue
                        
                    crop = frame[y1:y2, x1:x2]
                    feat = self.extract_features(crop)
                    if feat is not None:
                        features_list.append(feat)
                        
                        # Save the first successful crop as representative image
                        if len(features_list) == 1:
                            crop_path = os.path.join(crops_dir, f"{camera_id}_{track_id}.jpg")
                            cv2.imwrite(crop_path, crop)
                            
                if features_list:
                    # Average features across sampled crops
                    avg_feat = np.mean(features_list, axis=0)
                    norm = np.linalg.norm(avg_feat)
                    if norm > 0:
                        avg_feat = avg_feat / norm
                    all_embeddings[f"{camera_id}_{track_id}"] = avg_feat.tolist()
                    
            cap.release()
            
        # Save embeddings
        embeddings_path = os.path.join(evidence_dir, "embeddings.json")
        with open(embeddings_path, "w") as f:
            json.dump(all_embeddings, f, indent=4)
