"""
CaseLine — Semantic Evidence Search Module.

Uses CLIP (Contrastive Language-Image Pre-training) to enable
natural-language search across all captured suspect crop images.

Example queries:
  - "person wearing a red shirt"
  - "man carrying a backpack"
  - "woman with dark jacket"
"""

import os
import glob
from typing import List, Dict, Optional

import torch
import numpy as np
from PIL import Image


class SemanticSearch:
    """Zero-shot image-text search over suspect crop images using CLIP."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialise CLIP model and processor.

        Parameters
        ----------
        model_name : str
            HuggingFace CLIP model identifier.
            Default uses the lightweight ViT-B/32 variant (~600 MB).
        """
        from transformers import CLIPModel, CLIPProcessor

        print(f"[SemanticSearch] Loading CLIP model: {model_name} ...")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        # Cache: path -> embedding tensor
        self._image_cache: Dict[str, torch.Tensor] = {}
        print("[SemanticSearch] CLIP model loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_crops(
        self,
        query: str,
        case_dir: str,
        top_k: int = 6,
        crops_subdir: str = "output/crops",
    ) -> List[Dict]:
        """
        Search all crop images in a case directory using a text query.

        Parameters
        ----------
        query : str
            Natural language description (e.g. 'person in a red shirt').
        case_dir : str
            Root directory of the case.
        top_k : int
            Maximum number of results to return, ranked by match score.
        crops_subdir : str
            Subdirectory under case_dir where crop images are stored.

        Returns
        -------
        list of dict
            Each dict contains:
              - path (str): absolute path to the crop image
              - camera (str): camera ID parsed from filename
              - track (str): track ID parsed from filename
              - score (float): cosine similarity score (higher = better)
        """
        crops_dir = os.path.join(case_dir, crops_subdir)

        if not os.path.isdir(crops_dir):
            print(f"[SemanticSearch] Crops directory not found: {crops_dir}")
            return []

        # Collect all .jpg / .png crop images
        image_paths = sorted(
            glob.glob(os.path.join(crops_dir, "*.jpg"))
            + glob.glob(os.path.join(crops_dir, "*.png"))
        )

        if not image_paths:
            print("[SemanticSearch] No crop images found.")
            return []

        # Encode all images (with caching)
        image_embeddings = self._encode_images(image_paths)

        # Encode the text query
        text_embedding = self._encode_text(query)

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            text_embedding, image_embeddings, dim=1
        )

        # Rank and select top-k
        scores, indices = similarities.topk(min(top_k, len(image_paths)))

        results = []
        for score, idx in zip(scores.tolist(), indices.tolist()):
            path = image_paths[idx]
            filename = os.path.splitext(os.path.basename(path))[0]
            parts = filename.split("_", 1)

            camera = parts[0] if len(parts) >= 1 else "unknown"
            track = parts[1] if len(parts) >= 2 else "unknown"

            results.append({
                "path": path,
                "camera": camera,
                "track": track,
                "score": round(score, 4),
            })

        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode a text query into a CLIP embedding."""
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        # L2 normalise
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Encode a list of images into CLIP embeddings.
        Uses a per-path cache to avoid redundant computation.
        """
        embeddings = []

        for path in image_paths:
            if path in self._image_cache:
                embeddings.append(self._image_cache[path])
                continue

            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"[SemanticSearch] Skipping {path}: {e}")
                # Return a zero vector so indices stay aligned
                embeddings.append(torch.zeros(1, self.model.config.projection_dim))
                continue

            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                img_features = self.model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            self._image_cache[path] = img_features
            embeddings.append(img_features)

        return torch.cat(embeddings, dim=0)
