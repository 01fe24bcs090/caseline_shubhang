"""
Activity-Status Engine for CaseLine.
(Restored exactly to original Step 1 implementation as per user request)
"""
import os
import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Set, Optional
import pandas as pd

@dataclass
class ActivityConfig:
    ATTENTION_THRESHOLD: int = 20
    SUSPICIOUS_THRESHOLD: int = 70

    PROXIMITY_PIXELS: int = 150
    SUSTAINED_PROXIMITY_FRAMES: int = 15
    DISAPPEARANCE_FRAMES: int = 30

    SCORE_APPROACH_OBJECT: int = 20
    SCORE_SUSTAINED_PROXIMITY: int = 20
    SCORE_OBJECT_INTERACTION: int = 25
    SCORE_OBJECT_DISAPPEARED: int = 25
    SCORE_LEFT_AREA: int = 10
    SCORE_FAST_MOVEMENT: int = 20
    SCORE_SUDDEN_ACCELERATION: int = 25
    INTERACTION_IOU_THRESHOLD: float = 0.25
    FAST_MOVEMENT_THRESHOLD: float = 30.0
    ACCELERATION_MULTIPLIER: float = 3.0
    MIN_FLEE_SPEED: float = 15.0

class _PersonState:
    def __init__(self, track_id: int, first_frame: int):
        self.track_id = track_id
        self.status = "NORMAL"
        self.score = 0
        self.reasons: List[str] = []
        self.transitions: List[dict] = [
            {"frame": first_frame, "status": "NORMAL", "reason": "Person detected"}
        ]
        self.near_frames: Dict[int, int] = {}
        self.approached: Set[int] = set()
        self.interacted: Set[int] = set()
        self._scored: Set[str] = set()
        self.last_pos = None
        self.last_frame = None
        self.last_speed = None

class ActivityStatusEngine:
    def __init__(self, config: Optional[ActivityConfig] = None):
        self.config = config or ActivityConfig()
        self._states: Dict[int, _PersonState] = {}

    @staticmethod
    def _centre(x1, y1, x2, y2):
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _dist(a, b):
        cx1, cy1 = ActivityStatusEngine._centre(*a)
        cx2, cy2 = ActivityStatusEngine._centre(*b)
        return math.hypot(cx2 - cx1, cy2 - cy1)

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        union = aa + ab - inter
        return inter / union if union > 0 else 0.0

    def _add(self, st: _PersonState, tag: str, pts: int, reason: str, frame: int, ts: float):
        if tag in st._scored:
            return
        st._scored.add(tag)
        st.score += pts
        st.reasons.append(reason)
        prev = st.status

        if st.score >= self.config.SUSPICIOUS_THRESHOLD and prev != "SUSPICIOUS":
            st.status = "SUSPICIOUS"
            combined = "; ".join(st.reasons)
            st.transitions.append({"frame": frame, "status": "SUSPICIOUS", "reason": combined})
            self._log(ts, st.track_id, "SUSPICIOUS", st.reasons)
        elif st.score >= self.config.ATTENTION_THRESHOLD and prev == "NORMAL":
            st.status = "ATTENTION"
            st.transitions.append({"frame": frame, "status": "ATTENTION", "reason": reason})
            self._log(ts, st.track_id, "ATTENTION", [reason])

    @staticmethod
    def _log(ts, tid, state, reasons):
        m, s = int(ts // 60), ts % 60
        print(f"  [{m:02d}:{s:05.2f}] Track {tid} → {state}")
        for r in reasons:
            print(f"    Reason: {r}")

    def process_camera(self, tracking_csv: str, objects_csv: str, fps: float = 25.0) -> dict:
        if not os.path.exists(tracking_csv): return {}
        trk = pd.read_csv(tracking_csv)
        if trk.empty: return {}

        total_frames = int(trk["frame"].max())

        obj_by_frame = {}
        obj_timeline = {}
        obj_classes = {}

        if os.path.exists(objects_csv):
            obj = pd.read_csv(objects_csv)
            if not obj.empty:
                if "object_id" not in obj.columns:
                    grid = 100
                    obj["object_id"] = ((obj["x1"] + obj["x2"]) // (2 * grid) * 1000 + (obj["y1"] + obj["y2"]) // (2 * grid)).astype(int)
                for fn, grp in obj.groupby("frame"):
                    obj_by_frame[int(fn)] = grp.to_dict("records")
                for _, r in obj.iterrows():
                    oid = int(r["object_id"])
                    obj_timeline.setdefault(oid, set()).add(int(r["frame"]))
                    cls_name = r.get("class_name", r.get("predicted_class", "object"))
                    obj_classes[oid] = str(cls_name)

        for tid in trk["track_id"].unique():
            first_frame = int(trk.loc[trk["track_id"] == tid, "frame"].min())
            first_ts = float(trk.loc[trk["track_id"] == tid, "timestamp"].min())
            st = _PersonState(int(tid), first_frame)
            self._states[int(tid)] = st
            m, s = int(first_ts // 60), first_ts % 60
            print(f"  [{m:02d}:{s:05.2f}] Track {tid} → NORMAL")

        for frame_num in sorted(trk["frame"].unique()):
            persons = trk[trk["frame"] == frame_num]
            objects_here = obj_by_frame.get(int(frame_num), [])
            ts = frame_num / fps if fps > 0 else 0.0

            for _, pr in persons.iterrows():
                tid = int(pr["track_id"])
                st = self._states[tid]
                if st.status == "SUSPICIOUS": continue

                pbox = (pr["x1"], pr["y1"], pr["x2"], pr["y2"])
                curr_pos = self._centre(*pbox)

                if st.last_pos is not None:
                    frames_passed = frame_num - st.last_frame
                    if 0 < frames_passed < 10:
                        speed = math.hypot(curr_pos[0] - st.last_pos[0], curr_pos[1] - st.last_pos[1]) / frames_passed
                        
                        if "fast_movement" not in st._scored and speed > self.config.FAST_MOVEMENT_THRESHOLD:
                            self._add(st, "fast_movement", self.config.SCORE_FAST_MOVEMENT, "Unusual fast movement detected", int(frame_num), ts)
                            
                        if "sudden_acceleration" not in st._scored and st.last_speed is not None:
                            if speed > self.config.MIN_FLEE_SPEED and speed > (st.last_speed * self.config.ACCELERATION_MULTIPLIER):
                                self._add(st, "sudden_acceleration", self.config.SCORE_SUDDEN_ACCELERATION, "Sudden acceleration (fleeing) detected", int(frame_num), ts)
                                
                        st.last_speed = speed

                st.last_pos = curr_pos
                st.last_frame = frame_num

                for ob in objects_here:
                    oid = int(ob["object_id"])
                    obox = (ob["x1"], ob["y1"], ob["x2"], ob["y2"])
                    dist = self._dist(pbox, obox)
                    iou = self._iou(pbox, obox)
                    obj_name = obj_classes.get(oid, "object")

                    if dist < self.config.PROXIMITY_PIXELS:
                        if oid not in st.approached:
                            st.approached.add(oid)
                            self._add(st, f"approach_{oid}", self.config.SCORE_APPROACH_OBJECT, f"Approached {obj_name}", int(frame_num), ts)
                        st.near_frames[oid] = st.near_frames.get(oid, 0) + 1
                        if st.near_frames[oid] == self.config.SUSTAINED_PROXIMITY_FRAMES:
                            self._add(st, f"sustained_{oid}", self.config.SCORE_SUSTAINED_PROXIMITY, f"Remained near {obj_name}", int(frame_num), ts)

                    ix1, iy1 = max(pbox[0], obox[0]), max(pbox[1], obox[1])
                    ix2, iy2 = min(pbox[2], obox[2]), min(pbox[3], obox[3])
                    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    obox_area = (obox[2] - obox[0]) * (obox[3] - obox[1])
                    object_overlap_ratio = inter_area / obox_area if obox_area > 0 else 0.0

                    # Only counts as physical interaction (touch) if person heavily overlaps the object
                    is_interacting = (object_overlap_ratio > 0.40) or (iou > self.config.INTERACTION_IOU_THRESHOLD)
                    if is_interacting:
                        # If they touched it, they definitely approached it! (fixes large bounding box center-distance issues)
                        if oid not in st.approached:
                            st.approached.add(oid)
                            self._add(st, f"approach_{oid}", self.config.SCORE_APPROACH_OBJECT, f"Approached {obj_name}", int(frame_num), ts)
                            
                        if oid not in st.interacted:
                            st.interacted.add(oid)
                            self._add(st, f"interact_{oid}", self.config.SCORE_OBJECT_INTERACTION, f"Interacted with {obj_name}", int(frame_num), ts)

                for oid in list(st.approached):
                    if oid in obj_timeline:
                        last_seen = max(obj_timeline[oid])
                        if last_seen < (total_frames - self.config.DISAPPEARANCE_FRAMES):
                            if frame_num > (last_seen + self.config.DISAPPEARANCE_FRAMES):
                                obj_name = obj_classes.get(oid, "object")
                                self._add(st, f"gone_{oid}", self.config.SCORE_OBJECT_DISAPPEARED, f"{obj_name.capitalize()} subsequently disappeared", int(frame_num), ts)

                has_stolen = any(f"gone_{oid}" in st._scored for oid in st.approached)
                if has_stolen and objects_here:
                    all_far = all(self._dist(pbox, (o["x1"], o["y1"], o["x2"], o["y2"])) >= self.config.PROXIMITY_PIXELS * 2 for o in objects_here)
                    if all_far:
                        self._add(st, "left_area", self.config.SCORE_LEFT_AREA, "Left the area after interaction", int(frame_num), ts)

        activity_data = {}
        for tid, st in self._states.items():
            activity_data[tid] = {
                "final_status": st.status,
                "score": st.score,
                "all_reasons": st.reasons[:],
                "transitions": st.transitions[:]
            }
        return activity_data

    def save_transitions(self, output_csv: str):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        rows = [(t["frame"], tid, t["status"], t["reason"]) for tid, st in self._states.items() for t in st.transitions]
        rows.sort()
        with open(output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "state", "reason"])
            w.writerows(rows)

    def save_summary(self, output_csv: str):
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        rows = [(tid, st.status, st.score, "; ".join(st.reasons) if st.reasons else "Normal movement / presence") for tid, st in self._states.items()]
        rows.sort()
        with open(output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "final_status", "score", "reasons"])
            w.writerows(rows)
