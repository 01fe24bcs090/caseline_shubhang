import os
import json
import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx

class CrossCameraMatcher:
    """Matches tracks across cameras using appearance embeddings and temporal rules."""
    def __init__(self, similarity_threshold=0.65, max_time_gap=300):
        self.similarity_threshold = similarity_threshold
        self.max_time_gap = max_time_gap # max seconds gap to still consider a match
        
    def cosine_similarity(self, v1, v2):
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def match_tracks(self, case_dir: str):
        evidence_dir = os.path.join(case_dir, "evidence")
        embeddings_path = os.path.join(evidence_dir, "embeddings.json")
        
        if not os.path.exists(embeddings_path):
            return pd.DataFrame()
            
        with open(embeddings_path, "r") as f:
            embeddings = json.load(f)
            
        # Load temporal boundaries for each track
        track_info = {}
        csv_files = [f for f in os.listdir(evidence_dir) if f.endswith("_tracking.csv")]
        for csv_file in csv_files:
            camera_id = csv_file.split("_")[0]
            df = pd.read_csv(os.path.join(evidence_dir, csv_file))
            if df.empty:
                continue
            
            for track_id, group in df.groupby("track_id"):
                key = f"{camera_id}_{track_id}"
                track_info[key] = {
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "first_seen": group["timestamp"].min(),
                    "last_seen": group["timestamp"].max(),
                    "avg_confidence": group["confidence"].mean()
                }
                
        # Compare all pairs of tracks from different cameras
        candidates = []
        keys = list(embeddings.keys())
        
        for k1, k2 in combinations(keys, 2):
            info1 = track_info.get(k1)
            info2 = track_info.get(k2)
            
            if not info1 or not info2:
                continue
                
            # Do not link within the same camera
            if info1["camera_id"] == info2["camera_id"]:
                continue
                
            # Temporal logic
            gap = max(0, max(info1["first_seen"] - info2["last_seen"], info2["first_seen"] - info1["last_seen"]))
            
            if gap > self.max_time_gap:
                continue 
                
            temporal_score = max(0, 1.0 - (gap / self.max_time_gap))
                
            # Appearance Similarity
            emb1 = np.array(embeddings[k1])
            emb2 = np.array(embeddings[k2])
            appearance_sim = self.cosine_similarity(emb1, emb2)
            
            # Weighted overall confidence (Heavy emphasis on appearance)
            overall_confidence = (0.8 * appearance_sim) + (0.2 * temporal_score)
            
            if overall_confidence >= self.similarity_threshold:
                candidates.append({
                    "camera_a": info1["camera_id"],
                    "track_a": info1["track_id"],
                    "camera_b": info2["camera_id"],
                    "track_b": info2["track_id"],
                    "appearance_similarity": round(appearance_sim, 3),
                    "temporal_score": round(temporal_score, 3),
                    "overall_confidence": round(overall_confidence, 3),
                    "status": "REQUIRES HUMAN REVIEW"
                })
                
        columns = ["camera_a", "track_a", "camera_b", "track_b", "appearance_similarity", "temporal_score", "overall_confidence", "status"]
        candidates_df = pd.DataFrame(candidates, columns=columns)
        if not candidates_df.empty:
            candidates_df = candidates_df.sort_values("overall_confidence", ascending=False)
            
        candidates_path = os.path.join(evidence_dir, "candidate_links.csv")
        candidates_df.to_csv(candidates_path, index=False)
        return candidates_df

    def process_human_approvals(self, case_dir: str, approved_links: list):
        """
        approved_links is a list of dicts:
        [{"camera_a": "CAM01", "track_a": 1, "camera_b": "CAM02", "track_b": 5}, ...]
        Returns a subject_links.csv replacing the old manual ground truth.
        """
        evidence_dir = os.path.join(case_dir, "evidence")
        
        G = nx.Graph()
        # Build graph of approved links
        for link in approved_links:
            node_a = f"{link['camera_a']}_{link['track_a']}"
            node_b = f"{link['camera_b']}_{link['track_b']}"
            G.add_edge(node_a, node_b)
            
        # Find connected components (these are the subjects)
        subjects = list(nx.connected_components(G))
        
        # We need original track info to populate first_seen/last_seen/etc
        track_info = {}
        csv_files = [f for f in os.listdir(evidence_dir) if f.endswith("_tracking.csv")]
        for csv_file in csv_files:
            camera_id = csv_file.split("_")[0]
            df = pd.read_csv(os.path.join(evidence_dir, csv_file))
            if df.empty:
                continue
            for track_id, group in df.groupby("track_id"):
                key = f"{camera_id}_{track_id}"
                track_info[key] = {
                    "first_seen": group["timestamp"].min(),
                    "last_seen": group["timestamp"].max(),
                    "observations": len(group),
                    "avg_confidence": group["confidence"].mean()
                }
        
        rows = []
        for i, subj_component in enumerate(subjects):
            subject_id = f"SUBJECT_{i+1:03d}"
            for node in subj_component:
                cam_id, trk_id_str = node.split("_")
                trk_id = int(trk_id_str)
                t_info = track_info.get(node)
                
                if t_info:
                    rows.append({
                        "subject_id": subject_id,
                        "camera_id": cam_id,
                        "track_id": trk_id,
                        "first_seen": round(t_info["first_seen"], 2),
                        "last_seen": round(t_info["last_seen"], 2),
                        "observations": t_info["observations"],
                        "avg_confidence": round(t_info["avg_confidence"], 3)
                    })
                    
        res_df = pd.DataFrame(rows)
        if not res_df.empty:
            res_df.to_csv(os.path.join(evidence_dir, "subject_links.csv"), index=False)
        else:
            pd.DataFrame(columns=["subject_id", "camera_id", "track_id", "first_seen", "last_seen", "observations", "avg_confidence"]).to_csv(os.path.join(evidence_dir, "subject_links.csv"), index=False)
            
        return res_df