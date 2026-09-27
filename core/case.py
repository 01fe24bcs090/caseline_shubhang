import os
import json
import shutil
import cv2
import dataclasses
from datetime import datetime
from core.hashing import compute_sha256
from core.schemas import CaseInfo, VideoEvidence, AccessLogEvidence

class CaseManager:
    """Manages the creation, storage, and retrieval of cases and evidence."""
    
    def __init__(self, cases_dir="data/cases"):
        self.cases_dir = cases_dir
        os.makedirs(self.cases_dir, exist_ok=True)
        
    def create_case(self, case_id: str, incident_description: str) -> str:
        case_path = os.path.join(self.cases_dir, case_id)
        if os.path.exists(case_path):
            raise ValueError(f"Case {case_id} already exists.")
            
        # Setup case directory structure
        os.makedirs(case_path)
        os.makedirs(os.path.join(case_path, "videos"))
        os.makedirs(os.path.join(case_path, "evidence"))
        os.makedirs(os.path.join(case_path, "output"))
        
        case_info = CaseInfo(
            case_id=case_id,
            incident_description=incident_description,
            created_at=datetime.now().isoformat()
        )
        
        self.save_case_info(case_path, case_info)
        return case_path
        
    def save_case_info(self, case_path: str, case_info: CaseInfo):
        info_path = os.path.join(case_path, "case_info.json")
        with open(info_path, "w") as f:
            json.dump(dataclasses.asdict(case_info), f, indent=4)
            
    def load_case_info(self, case_id: str) -> CaseInfo:
        case_path = os.path.join(self.cases_dir, case_id)
        info_path = os.path.join(case_path, "case_info.json")
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Case info for {case_id} not found.")
            
        with open(info_path, "r") as f:
            data = json.load(f)
            
        # Reconstruct objects
        videos = []
        for v in data.get("videos", []):
            # Ensure tuple for resolution
            if "resolution" in v and isinstance(v["resolution"], list):
                v["resolution"] = tuple(v["resolution"])
            videos.append(VideoEvidence(**v))
            
        access_log_data = data.get("access_logs")
        access_logs = AccessLogEvidence(**access_log_data) if access_log_data else None
        
        return CaseInfo(
            case_id=data["case_id"],
            incident_description=data["incident_description"],
            status=data["status"],
            videos=videos,
            access_logs=access_logs,
            created_at=data["created_at"]
        )

    def ingest_video(self, case_id: str, source_path: str) -> VideoEvidence:
        case_path = os.path.join(self.cases_dir, case_id)
        case_info = self.load_case_info(case_id)
        
        # Determine Camera ID dynamically
        cam_idx = len(case_info.videos) + 1
        camera_id = f"CAM{cam_idx:02d}"
        
        filename = os.path.basename(source_path)
        dest_path = os.path.join(case_path, "videos", f"{camera_id}_{filename}")
        
        # Copy to case directory
        shutil.copy2(source_path, dest_path)
        
        # Extract properties
        sha256 = compute_sha256(dest_path)
        file_size = os.path.getsize(dest_path)
        
        cap = cv2.VideoCapture(dest_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        vid_ev = VideoEvidence(
            camera_id=camera_id,
            original_filename=filename,
            filepath=dest_path,
            sha256=sha256,
            file_size_bytes=file_size,
            duration_sec=duration,
            fps=fps,
            resolution=(width, height)
        )
        
        case_info.videos.append(vid_ev)
        self.save_case_info(case_path, case_info)
        return vid_ev

    def ingest_access_log(self, case_id: str, source_path: str) -> AccessLogEvidence:
        case_path = os.path.join(self.cases_dir, case_id)
        case_info = self.load_case_info(case_id)
        
        filename = os.path.basename(source_path)
        dest_path = os.path.join(case_path, "evidence", f"access_log_{filename}")
        
        shutil.copy2(source_path, dest_path)
        sha256 = compute_sha256(dest_path)
        
        log_ev = AccessLogEvidence(
            original_filename=filename,
            filepath=dest_path,
            sha256=sha256
        )
        
        case_info.access_logs = log_ev
        self.save_case_info(case_path, case_info)
        return log_ev
