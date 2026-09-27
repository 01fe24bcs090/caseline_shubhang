from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class VideoEvidence:
    camera_id: str
    original_filename: str
    filepath: str
    sha256: str
    file_size_bytes: int
    duration_sec: float
    fps: float
    resolution: tuple

@dataclass
class AccessLogEvidence:
    original_filename: str
    filepath: str
    sha256: str

@dataclass
class CaseInfo:
    case_id: str
    incident_description: str
    status: str = "INITIALIZED"
    videos: List[VideoEvidence] = field(default_factory=list)
    access_logs: Optional[AccessLogEvidence] = None
    created_at: str = ""
