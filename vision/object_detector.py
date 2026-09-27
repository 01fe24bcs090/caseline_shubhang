"""
Object detection and temporal stability tracking module for CaseLine.

Runs YOLO on video frames to detect portable/security-relevant objects,
tracks objects across time, and evaluates temporal classification stability
to eliminate single-frame classification flickering (e.g. laptop vs suitcase).
"""
import os
import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import cv2
import pandas as pd
from ultralytics import YOLO


# Target COCO classes considered for security / interaction analysis
TARGET_OBJECT_CLASSES = {
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    43: "knife",
    63: "laptop",
    65: "remote",
    67: "cell phone",
    73: "book",
    75: "vase",
    76: "scissors",
}


@dataclass
class ObjectStabilityConfig:
    """Configurable thresholds for object temporal confirmation and stability."""
    OBJECT_CONFIDENCE_THRESHOLD: float = 0.60  # Minimum avg confidence for confirmed class
    MIN_OBJECT_OBSERVATIONS: int = 5          # Minimum observations before confirming class
    MIN_CLASS_STABILITY: float = 0.70         # Minimum dominant class ratio (e.g. 70%)
    MAX_TRACK_DISTANCE: float = 75.0          # Max centroid distance (pixels) between frames
    IOU_MATCH_THRESHOLD: float = 0.20         # Minimum IoU overlap to associate detection
    MAX_MISSED_FRAMES: int = 45               # Frames to retain stationary/occluded tracks


class TrackedObject:
    """Represents a single tracked physical object across video frames."""

    def __init__(self, object_id: int, initial_box: Tuple[int, int, int, int],
                 frame: int, timestamp: float, class_name: str, confidence: float):
        self.object_id = object_id
        self.last_box = initial_box
        self.last_frame = frame
        self.first_frame = frame
        self.observations: List[dict] = []
        self.class_counts: Counter = Counter()
        self.class_confidences: Dict[str, List[float]] = {}
        
        self.add_observation(frame, timestamp, class_name, confidence, initial_box)

    @property
    def centroid(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.last_box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def add_observation(self, frame: int, timestamp: float,
                        class_name: str, confidence: float, box: Tuple[int, int, int, int]):
        self.last_box = box
        self.last_frame = frame
        self.observations.append({
            "frame": frame,
            "timestamp": timestamp,
            "class_name": class_name,
            "confidence": confidence,
            "x1": box[0],
            "y1": box[1],
            "x2": box[2],
            "y2": box[3],
        })
        self.class_counts[class_name] += 1
        self.class_confidences.setdefault(class_name, []).append(confidence)

    def get_stability_metrics(self) -> dict:
        total_obs = len(self.observations)
        if total_obs == 0:
            return {
                "dominant_class": "unknown",
                "total_observations": 0,
                "class_stability": 0.0,
                "dominant_avg_conf": 0.0,
                "breakdown": {},
            }

        dominant_class, dominant_count = self.class_counts.most_common(1)[0]
        class_stability = dominant_count / total_obs
        dom_confs = self.class_confidences.get(dominant_class, [0.0])
        dominant_avg_conf = sum(dom_confs) / len(dom_confs) if dom_confs else 0.0

        return {
            "dominant_class": dominant_class,
            "total_observations": total_obs,
            "class_stability": round(class_stability, 3),
            "dominant_avg_conf": round(dominant_avg_conf, 3),
            "breakdown": dict(self.class_counts),
        }

    def is_confirmed(self, config: ObjectStabilityConfig) -> bool:
        """Determines if the object's dominant class satisfies temporal confirmation."""
        metrics = self.get_stability_metrics()
        return (
            metrics["total_observations"] >= config.MIN_OBJECT_OBSERVATIONS
            and metrics["class_stability"] >= config.MIN_CLASS_STABILITY
            and metrics["dominant_avg_conf"] >= config.OBJECT_CONFIDENCE_THRESHOLD
        )

    def get_effective_class(self, config: ObjectStabilityConfig) -> str:
        """Returns the confirmed class name or 'Uncertain object' if unconfirmed/unstable."""
        if self.is_confirmed(config):
            return self.get_stability_metrics()["dominant_class"]
        return "Uncertain object"


class ObjectDetector:
    """Detects and tracks non-person objects with temporal stability analysis."""

    def __init__(self, model_path="yolo11n.pt", config: Optional[ObjectStabilityConfig] = None):
        self.model = YOLO(model_path)
        self.config = config or ObjectStabilityConfig()

    @staticmethod
    def _calculate_iou(box_a, box_b) -> float:
        ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
        ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _centroid_distance(box_a, box_b) -> float:
        cax, cay = (box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0
        cbx, cby = (box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0
        return math.hypot(cbx - cax, cby - cay)

    def detect_and_track_objects(
        self,
        video_path: str,
        output_csv: str,
        frame_skip: int = 2,
        conf: float = 0.30,
        diagnostics_csv: Optional[str] = None,
    ) -> Tuple[int, Dict[int, TrackedObject]]:
        """
        Run YOLO object detection and spatial tracking on video frames.
        Saves tracked detections to output_csv and diagnostic records to diagnostics_csv.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_ids = list(TARGET_OBJECT_CLASSES.keys())
        class_names = self.model.names

        tracked_objects: Dict[int, TrackedObject] = {}
        next_object_id = 1
        frame_number = 0
        total_detections = 0

        # Diagnostics buffer
        diag_rows = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1
            if frame_number % frame_skip != 0:
                continue

            timestamp = round(frame_number / fps, 2)

            results = self.model(
                frame,
                conf=conf,
                classes=target_ids,
                verbose=False,
            )[0]

            frame_detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()
                confidences = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)

                for box, c, cls_id in zip(boxes, confidences, class_ids):
                    c_name = class_names.get(cls_id, TARGET_OBJECT_CLASSES.get(cls_id, f"obj_{cls_id}"))
                    int_box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                    frame_detections.append({
                        "box": int_box,
                        "class_name": c_name,
                        "confidence": round(float(c), 3),
                    })

            # Match frame detections to existing active object tracks
            matched_obj_ids = set()

            for det in frame_detections:
                box = det["box"]
                c_name = det["class_name"]
                conf_val = det["confidence"]

                best_id = None
                best_score = -1.0

                for obj_id, track in tracked_objects.items():
                    if obj_id in matched_obj_ids:
                        continue
                    # Check if track is still alive
                    if (frame_number - track.last_frame) > self.config.MAX_MISSED_FRAMES:
                        continue

                    iou = self._calculate_iou(box, track.last_box)
                    dist = self._centroid_distance(box, track.last_box)

                    if iou >= self.config.IOU_MATCH_THRESHOLD or dist <= self.config.MAX_TRACK_DISTANCE:
                        # Combined matching score favoring high IoU and low distance
                        score = iou * 2.0 + max(0.0, 1.0 - (dist / self.config.MAX_TRACK_DISTANCE))
                        if score > best_score:
                            best_score = score
                            best_id = obj_id

                if best_id is not None:
                    matched_obj_ids.add(best_id)
                    tracked_objects[best_id].add_observation(
                        frame_number, timestamp, c_name, conf_val, box
                    )
                    curr_obj_id = best_id
                else:
                    curr_obj_id = next_object_id
                    next_object_id += 1
                    matched_obj_ids.add(curr_obj_id)
                    tracked_objects[curr_obj_id] = TrackedObject(
                        curr_obj_id, box, frame_number, timestamp, c_name, conf_val
                    )

                # Record diagnostic row: frame, timestamp, object_id, predicted_class, confidence, x1, y1, x2, y2
                diag_rows.append([
                    frame_number,
                    timestamp,
                    curr_obj_id,
                    c_name,
                    conf_val,
                    box[0], box[1], box[2], box[3],
                ])
                total_detections += 1

        cap.release()

        # Save primary objects CSV with tracked object IDs and stability info
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame", "timestamp", "object_id", "predicted_class", "confidence",
                "x1", "y1", "x2", "y2", "dominant_class", "class_stability",
                "dominant_avg_conf", "is_confirmed", "effective_class"
            ])
            for obj_id, track in tracked_objects.items():
                metrics = track.get_stability_metrics()
                is_conf = track.is_confirmed(self.config)
                eff_cls = track.get_effective_class(self.config)

                for obs in track.observations:
                    writer.writerow([
                        obs["frame"],
                        obs["timestamp"],
                        obj_id,
                        obs["class_name"],
                        obs["confidence"],
                        obs["x1"], obs["y1"], obs["x2"], obs["y2"],
                        metrics["dominant_class"],
                        metrics["class_stability"],
                        metrics["dominant_avg_conf"],
                        int(is_conf),
                        eff_cls,
                    ])

        # Write or append to diagnostics CSV if provided
        if diagnostics_csv:
            os.makedirs(os.path.dirname(diagnostics_csv), exist_ok=True)
            write_header = not os.path.exists(diagnostics_csv)
            with open(diagnostics_csv, "a" if not write_header else "w", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([
                        "frame", "timestamp", "object_id", "predicted_class",
                        "confidence", "x1", "y1", "x2", "y2"
                    ])
                for row in diag_rows:
                    writer.writerow(row)

        print(f"  Object detection & tracking complete: {len(tracked_objects)} objects tracked across {frame_number} frames ({total_detections} detections)")
        return total_detections, tracked_objects

    def detect_objects(self, video_path: str, output_csv: str, frame_skip: int = 2, conf: float = 0.30):
        """Backward-compatible wrapper for pipeline."""
        return self.detect_and_track_objects(video_path, output_csv, frame_skip=frame_skip, conf=conf)[0]
