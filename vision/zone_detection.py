"""
CaseLine — Restricted Zone Detection Module.

Allows investigators to define restricted/sensitive zones on camera feeds.
The system automatically flags any person who enters a defined zone,
contributing evidence points to the Activity Status Engine.

Zone definitions are stored as JSON per camera, and can be created
through the Streamlit UI by drawing polygons on a reference frame.
"""

import os
import json
import csv
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


@dataclass
class Zone:
    """Represents a restricted zone defined on a camera view."""
    zone_id: str
    camera_id: str
    name: str
    zone_type: str  # "RESTRICTED", "SENSITIVE", "EXIT", "ENTRY"
    polygon: List[Tuple[int, int]]  # List of (x, y) vertices
    severity: int = 30  # Evidence points added when a person enters
    color: Tuple[int, int, int] = (0, 0, 255)  # BGR color for rendering

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point (x, y) is inside this zone polygon using ray-casting."""
        n = len(self.polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def contains_bbox(self, x1: int, y1: int, x2: int, y2: int, threshold: float = 0.3) -> bool:
        """
        Check if a bounding box overlaps with this zone.
        
        Uses a sampling approach: checks if the bottom-center point
        (feet position) or a threshold ratio of sampled points are inside.
        
        Parameters
        ----------
        threshold : float
            Fraction of sample points that must be inside to count as "in zone".
        """
        # Primary check: bottom-center of bounding box (person's feet)
        foot_x = (x1 + x2) / 2.0
        foot_y = y2  # bottom edge
        if self.contains_point(foot_x, foot_y):
            return True

        # Secondary check: center of bounding box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        if self.contains_point(center_x, center_y):
            return True

        # Sample grid check for large bounding boxes
        sample_points = [
            (x1, y2), (x2, y2),               # bottom corners
            (center_x, y2),                     # bottom center
            (x1, center_y), (x2, center_y),    # side centers
        ]
        inside_count = sum(1 for px, py in sample_points if self.contains_point(px, py))
        return (inside_count / len(sample_points)) >= threshold


class ZoneManager:
    """Manages restricted zone definitions per camera for a case."""

    def __init__(self, case_dir: str):
        self.case_dir = case_dir
        self.zones_dir = os.path.join(case_dir, "evidence", "zones")
        os.makedirs(self.zones_dir, exist_ok=True)
        self._zones: Dict[str, List[Zone]] = {}  # camera_id -> list of zones

    def get_zones_file(self, camera_id: str) -> str:
        return os.path.join(self.zones_dir, f"{camera_id}_zones.json")

    def load_zones(self, camera_id: str) -> List[Zone]:
        """Load zones for a specific camera."""
        if camera_id in self._zones:
            return self._zones[camera_id]

        zones_file = self.get_zones_file(camera_id)
        if not os.path.exists(zones_file):
            self._zones[camera_id] = []
            return []

        with open(zones_file, "r") as f:
            data = json.load(f)

        zones = []
        for z in data:
            zone = Zone(
                zone_id=z["zone_id"],
                camera_id=z["camera_id"],
                name=z["name"],
                zone_type=z["zone_type"],
                polygon=[tuple(p) for p in z["polygon"]],
                severity=z.get("severity", 30),
                color=tuple(z.get("color", [0, 0, 255])),
            )
            zones.append(zone)

        self._zones[camera_id] = zones
        return zones

    def save_zones(self, camera_id: str, zones: List[Zone]):
        """Save zone definitions for a camera."""
        zones_file = self.get_zones_file(camera_id)
        data = []
        for z in zones:
            data.append({
                "zone_id": z.zone_id,
                "camera_id": z.camera_id,
                "name": z.name,
                "zone_type": z.zone_type,
                "polygon": [list(p) for p in z.polygon],
                "severity": z.severity,
                "color": list(z.color),
            })

        with open(zones_file, "w") as f:
            json.dump(data, f, indent=4)

        self._zones[camera_id] = zones

    def add_zone(self, camera_id: str, name: str, zone_type: str,
                 polygon: List[Tuple[int, int]], severity: int = 30) -> Zone:
        """Add a new zone to a camera."""
        zones = self.load_zones(camera_id)
        zone_id = f"ZONE_{camera_id}_{len(zones) + 1:02d}"

        # Color based on zone type
        type_colors = {
            "RESTRICTED": (0, 0, 255),     # Red
            "SENSITIVE": (0, 165, 255),     # Orange
            "EXIT": (0, 255, 0),            # Green
            "ENTRY": (255, 200, 0),         # Cyan-ish
        }
        color = type_colors.get(zone_type, (0, 0, 255))

        zone = Zone(
            zone_id=zone_id,
            camera_id=camera_id,
            name=name,
            zone_type=zone_type,
            polygon=polygon,
            severity=severity,
            color=color,
        )
        zones.append(zone)
        self.save_zones(camera_id, zones)
        return zone

    def remove_zone(self, camera_id: str, zone_id: str):
        """Remove a zone by ID."""
        zones = self.load_zones(camera_id)
        zones = [z for z in zones if z.zone_id != zone_id]
        self.save_zones(camera_id, zones)

    def get_all_zones(self) -> Dict[str, List[Zone]]:
        """Load and return all zones across all cameras."""
        for f in os.listdir(self.zones_dir):
            if f.endswith("_zones.json"):
                cam_id = f.replace("_zones.json", "")
                self.load_zones(cam_id)
        return self._zones


class ZoneViolationDetector:
    """
    Detects zone violations by cross-referencing person tracking data
    with defined restricted zones.
    """

    def __init__(self, zone_manager: ZoneManager):
        self.zone_manager = zone_manager

    def detect_violations(self, camera_id: str, tracking_csv: str,
                          fps: float = 25.0) -> pd.DataFrame:
        """
        Analyse tracking data against zone definitions and return violations.

        Parameters
        ----------
        camera_id : str
            Camera identifier.
        tracking_csv : str
            Path to the person tracking CSV for this camera.
        fps : float
            Video frames per second.

        Returns
        -------
        pd.DataFrame
            Violation records with columns:
            frame, timestamp, camera_id, track_id, zone_id, zone_name,
            zone_type, duration_frames, severity
        """
        zones = self.zone_manager.load_zones(camera_id)

        if not zones:
            return pd.DataFrame()

        if not os.path.exists(tracking_csv):
            return pd.DataFrame()

        trk = pd.read_csv(tracking_csv)
        if trk.empty:
            return pd.DataFrame()

        # Track which tracks are currently inside which zones
        # Key: (track_id, zone_id) -> first entry frame
        active_entries: Dict[Tuple[int, str], int] = {}
        violations = []

        for _, row in trk.sort_values("frame").iterrows():
            frame = int(row["frame"])
            track_id = int(row["track_id"])
            x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
            timestamp = frame / fps if fps > 0 else 0.0

            for zone in zones:
                key = (track_id, zone.zone_id)
                is_inside = zone.contains_bbox(x1, y1, x2, y2)

                if is_inside and key not in active_entries:
                    # Person entered the zone
                    active_entries[key] = frame
                    violations.append({
                        "frame": frame,
                        "timestamp": round(timestamp, 2),
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "zone_id": zone.zone_id,
                        "zone_name": zone.name,
                        "zone_type": zone.zone_type,
                        "event": "ZONE_ENTRY",
                        "severity": zone.severity,
                    })

                elif not is_inside and key in active_entries:
                    # Person exited the zone
                    entry_frame = active_entries.pop(key)
                    duration = frame - entry_frame
                    violations.append({
                        "frame": frame,
                        "timestamp": round(timestamp, 2),
                        "camera_id": camera_id,
                        "track_id": track_id,
                        "zone_id": zone.zone_id,
                        "zone_name": zone.name,
                        "zone_type": zone.zone_type,
                        "event": "ZONE_EXIT",
                        "duration_frames": duration,
                        "severity": zone.severity,
                    })

        return pd.DataFrame(violations)

    def save_violations(self, camera_id: str, violations_df: pd.DataFrame):
        """Save violations to a CSV in the evidence directory."""
        if violations_df.empty:
            return

        evidence_dir = os.path.join(self.zone_manager.case_dir, "evidence")
        output_path = os.path.join(evidence_dir, f"{camera_id}_zone_violations.csv")
        violations_df.to_csv(output_path, index=False)
        print(f"  Zone violations saved: {output_path}")

    def get_violation_summary(self, camera_id: str, tracking_csv: str,
                              fps: float = 25.0) -> Dict:
        """
        Get a summary of zone violations for a camera.

        Returns
        -------
        dict
            Summary with total_violations, unique_tracks, zones_breached,
            and a per-zone breakdown.
        """
        violations_df = self.detect_violations(camera_id, tracking_csv, fps)

        if violations_df.empty:
            return {
                "total_violations": 0,
                "unique_tracks": 0,
                "zones_breached": [],
                "breakdown": {},
            }

        entries = violations_df[violations_df["event"] == "ZONE_ENTRY"]
        return {
            "total_violations": len(entries),
            "unique_tracks": entries["track_id"].nunique(),
            "zones_breached": entries["zone_id"].unique().tolist(),
            "breakdown": entries.groupby("zone_name")["track_id"].nunique().to_dict(),
        }


def extract_reference_frame(video_path: str, frame_number: int = 30) -> Optional[np.ndarray]:
    """
    Extract a single reference frame from a video for zone drawing.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    frame_number : int
        Frame to extract (default: frame 30 to skip black intro frames).

    Returns
    -------
    np.ndarray or None
        The extracted frame as a BGR numpy array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def render_zones_on_frame(frame: np.ndarray, zones: List[Zone],
                           alpha: float = 0.25) -> np.ndarray:
    """
    Overlay zone polygons on a video frame with semi-transparent fill.

    Parameters
    ----------
    frame : np.ndarray
        BGR image to draw zones on.
    zones : list of Zone
        Zone definitions to render.
    alpha : float
        Transparency of the zone fill (0 = invisible, 1 = opaque).

    Returns
    -------
    np.ndarray
        Frame with zones drawn.
    """
    overlay = frame.copy()

    for zone in zones:
        pts = np.array(zone.polygon, dtype=np.int32).reshape((-1, 1, 2))

        # Draw filled polygon on overlay
        cv2.fillPoly(overlay, [pts], zone.color)

        # Draw polygon border on the original frame
        cv2.polylines(frame, [pts], isClosed=True, color=zone.color, thickness=2)

        # Draw zone label
        centroid_x = int(np.mean([p[0] for p in zone.polygon]))
        centroid_y = int(np.mean([p[1] for p in zone.polygon]))

        label = f"{zone.zone_type}: {zone.name}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (centroid_x - tw // 2 - 4, centroid_y - th - 6),
            (centroid_x + tw // 2 + 4, centroid_y + 4),
            zone.color, -1,
        )
        cv2.putText(
            frame, label,
            (centroid_x - tw // 2, centroid_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    # Blend overlay with original frame
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return result
