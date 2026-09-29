"""
CaseLine — Forensic Evidence Visualizer & Photographic Exhibit Generator.

Extracts, caches, and indexes visual evidence:
  1. High-resolution annotated CCTV video frame snapshots with forensic bounding boxes.
  2. Tight, zoomed-in asset crops with contextual margin (e.g. laptops, backpacks, handbags).
  3. Verified incident snapshots capturing the exact frame of suspect interaction / object disappearance.
  4. Automatic semantic matching of user queries and Gemini answers to the relevant visual exhibits.
"""

import os
import glob
import json
from typing import Dict, List, Optional
import cv2
import pandas as pd


def ensure_case_snapshots(case_dir: str, case_info=None) -> Dict[str, List[Dict[str, str]]]:
    """
    Scans the case evidence and videos to extract and index photographic exhibits.
    Cached in `case_dir/output/evidence_snapshots/`.
    """
    snapshots_dir = os.path.join(case_dir, "output", "evidence_snapshots")
    index_file = os.path.join(snapshots_dir, "index.json")
    
    # Return cached index if already generated and non-empty
    if os.path.exists(index_file):
        try:
            with open(index_file, "r") as f:
                data = json.load(f)
                if data and isinstance(data, dict):
                    return data
        except Exception:
            pass

    os.makedirs(snapshots_dir, exist_ok=True)
    evidence_dir = os.path.join(case_dir, "evidence")
    videos_dir = os.path.join(case_dir, "videos")
    crops_dir = os.path.join(case_dir, "output", "crops")

    catalog = {
        "objects": [],
        "incidents": [],
        "suspects": []
    }

    # Map available video files by camera
    video_map = {}
    if os.path.exists(videos_dir):
        for vf in os.listdir(videos_dir):
            if vf.endswith((".mp4", ".webm", ".avi")):
                cam = vf.split("_")[0]
                video_map[cam] = os.path.join(videos_dir, vf)

    # 1. Extract Detected Object Crops & Annotated Frames
    obj_files = sorted(glob.glob(os.path.join(evidence_dir, "*_objects.csv")))
    for of in obj_files:
        cam = os.path.basename(of).split("_")[0]
        if cam not in video_map:
            continue
        try:
            odf = pd.read_csv(of)
            if odf.empty or "predicted_class" not in odf.columns:
                continue

            for pclass, group in odf.groupby("predicted_class"):
                best_row = group.sort_values(by="confidence", ascending=False).iloc[0]
                frm = int(best_row["frame"])
                ts = float(best_row["timestamp"])
                conf = float(best_row["confidence"])
                x1, y1, x2, y2 = int(best_row["x1"]), int(best_row["y1"]), int(best_row["x2"]), int(best_row["y2"])

                cap = cv2.VideoCapture(video_map[cam])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frm)
                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    continue

                h, w = frame.shape[:2]

                # Zoomed asset crop with margin
                pad_x = int((x2 - x1) * 0.3) + 10
                pad_y = int((y2 - y1) * 0.3) + 10
                cx1 = max(0, x1 - pad_x)
                cy1 = max(0, y1 - pad_y)
                cx2 = min(w, x2 + pad_x)
                cy2 = min(h, y2 + pad_y)
                crop = frame[cy1:cy2, cx1:cx2].copy()

                crop_filename = f"crop_{cam}_{pclass}.jpg"
                crop_path = os.path.join(snapshots_dir, crop_filename)
                cv2.imwrite(crop_path, crop)

                # Annotated full frame
                annotated = frame.copy()
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cam} | {pclass.upper()} (conf: {conf*100:.0f}%) @ {ts:.1f}s"
                text_w = len(label) * 9 + 8
                cv2.rectangle(annotated, (x1, max(0, y1 - 25)), (x1 + text_w, y1), (0, 0, 0), -1)
                cv2.putText(annotated, label, (x1 + 4, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

                snap_filename = f"snapshot_{cam}_{pclass}.jpg"
                snap_path = os.path.join(snapshots_dir, snap_filename)
                cv2.imwrite(snap_path, annotated)

                catalog["objects"].append({
                    "class": pclass.lower(),
                    "camera": cam,
                    "timestamp": ts,
                    "confidence": conf,
                    "crop_path": crop_path,
                    "snapshot_path": snap_path,
                    "caption": f"Exhibit: {pclass.title()} on {cam} @ {ts:05.2f}s (Conf: {conf*100:.1f}%)"
                })
        except Exception:
            pass

    # 2. Extract Disappearance / Theft Incident Snapshots
    trans_files = sorted(glob.glob(os.path.join(evidence_dir, "*_activity_transitions.csv")))
    for tf in trans_files:
        cam = os.path.basename(tf).split("_")[0]
        if cam not in video_map:
            continue
        try:
            tdf = pd.read_csv(tf)
            suspicious = tdf[tdf["state"] == "SUSPICIOUS"]
            for _, r in suspicious.iterrows():
                frm = int(r["frame"])
                tid = r["track_id"]
                reason = str(r["reason"])

                cap = cv2.VideoCapture(video_map[cam])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frm)
                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    continue

                annotated = frame.copy()
                # Forensic header banner
                cv2.rectangle(annotated, (8, 8), (frame.shape[1] - 8, 42), (0, 0, 160), -1)
                banner_txt = f"INCIDENT EXHIBIT: {cam} Track {tid} | {reason[:58]}"
                cv2.putText(annotated, banner_txt, (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                slug = "theft" if "disappeared" in reason.lower() else "suspicious"
                inc_filename = f"incident_{cam}_track{tid}_{slug}_f{frm}.jpg"
                inc_path = os.path.join(snapshots_dir, inc_filename)
                cv2.imwrite(inc_path, annotated)

                catalog["incidents"].append({
                    "camera": cam,
                    "track_id": tid,
                    "frame": frm,
                    "reason": reason,
                    "path": inc_path,
                    "caption": f"Forensic Incident: {cam} Track {tid} — {reason}"
                })
        except Exception:
            pass

    # 3. Collect Suspect Crops & Cross-Reference with Activity Engine
    # Map which tracks in which cameras were flagged with suspicious activity
    flagged_tracks = {}
    for tf in trans_files:
        cam_t = os.path.basename(tf).split("_")[0]
        try:
            tdf = pd.read_csv(tf)
            for _, r in tdf.iterrows():
                st = str(r.get("state", ""))
                tid_str = str(r.get("track_id", ""))
                reason_str = str(r.get("reason", ""))
                key = (cam_t, tid_str)
                if st in ["ATTENTION", "SUSPICIOUS"]:
                    if key not in flagged_tracks:
                        flagged_tracks[key] = []
                    flagged_tracks[key].append(f"{st}: {reason_str}")
        except Exception:
            pass

    if os.path.exists(crops_dir):
        for cf in sorted(os.listdir(crops_dir)):
            if cf.endswith((".jpg", ".png")):
                parts = cf.replace(".jpg", "").replace(".png", "").split("_")
                cam = parts[0] if len(parts) > 0 else "CAM"
                tid = parts[1] if len(parts) > 1 else "?"
                
                key = (cam, tid)
                is_flagged = key in flagged_tracks
                if is_flagged:
                    action_summary = "; ".join(flagged_tracks[key][:2])
                    caption = f"Flagged Suspect: {cam} Track {tid} ({action_summary})"
                else:
                    caption = f"Subject Profile: {cam} Track {tid}"

                catalog["suspects"].append({
                    "camera": cam,
                    "track_id": tid,
                    "is_flagged": is_flagged,
                    "path": os.path.join(crops_dir, cf),
                    "caption": caption
                })

        # Sort suspects so flagged suspects in this case appear first
        catalog["suspects"].sort(key=lambda s: not s.get("is_flagged", False))

    try:
        with open(index_file, "w") as f:
            json.dump(catalog, f, indent=2)
    except Exception:
        pass

    return catalog


def get_relevant_evidence_images(
    query: str,
    answer: str,
    case_dir: str,
    case_info=None,
    top_k: int = 3
) -> List[Dict[str, str]]:
    """
    Selects the most relevant photographic evidence exhibits dynamically based on 
    query intent and the specific case's verified visual catalog.
    100% case-agnostic — works for any ingested CCTV case.
    """
    catalog = ensure_case_snapshots(case_dir, case_info)
    q_lower = query.lower()
    ans_lower = answer.lower()
    combined_text = q_lower + " " + ans_lower

    selected = []
    seen_paths = set()

    def add_item(path: str, caption: str, title: str):
        if path and os.path.exists(path) and path not in seen_paths:
            seen_paths.add(path)
            selected.append({"path": path, "caption": caption, "title": title})

    # PRIORITY 1: USER ASKS FOR SUSPECTS / PERSONS / PHOTOS / FACES
    suspect_query_terms = [
        "suspect", "suspected", "thief", "who", "person", "people", 
        "photo", "picture", "image", "face", "profile", "look like"
    ]
    is_suspect_query = any(k in q_lower for k in suspect_query_terms)

    if is_suspect_query:
        # A) Prioritize suspect profile crops that were flagged in this case
        for s in catalog.get("suspects", []):
            title = "Suspect Profile" if s.get("is_flagged") else "Subject Profile"
            add_item(s["path"], s["caption"], f"{title} ({s['camera']} Track {s['track_id']})")
            if len(selected) >= 2:
                break

        # B) CCTV Incident frame showing suspect in action during event
        for inc in catalog.get("incidents", []):
            add_item(inc["path"], inc["caption"], "CCTV Incident Capture (Suspect in Action)")
            if len(selected) >= top_k:
                break

        if selected:
            return selected[:top_k]

    # PRIORITY 2: USER ASKS ABOUT ANY DETECTED OBJECT CLASS IN THIS CASE
    # Dynamically match any object class that was actually detected in this case
    detected_classes = {obj["class"].lower() for obj in catalog.get("objects", [])}
    for obj in catalog.get("objects", []):
        obj_class = obj["class"].lower()
        # Direct class match or common synonyms
        is_matched = obj_class in q_lower
        if not is_matched and obj_class in ["backpack", "handbag", "suitcase"] and any(k in q_lower for k in ["bag", "luggage", "stolen", "steal"]):
            is_matched = True

        if is_matched:
            add_item(obj["snapshot_path"], obj["caption"], f"{obj['class'].title()} Scene Exhibit")
            add_item(obj["crop_path"], f"{obj['class'].title()} Asset Crop ({obj['camera']})", f"{obj['class'].title()} Asset")
            if len(selected) >= top_k:
                break

    if selected:
        return selected[:top_k]

    # PRIORITY 3: USER ASKS ABOUT THEFT / DISAPPEARANCES / LOSS
    theft_query_terms = ["steal", "stolen", "theft", "disappear", "missing", "loss", "take", "taken"]
    if any(k in q_lower for k in theft_query_terms):
        for inc in catalog.get("incidents", []):
            if any(term in inc["reason"].lower() for term in ["disappeared", "theft", "interacted"]):
                add_item(inc["path"], inc["caption"], "Incident Capture")
                if len(selected) >= 2:
                    break

        for obj in catalog.get("objects", []):
            add_item(obj["snapshot_path"], obj["caption"], f"{obj['class'].title()} Asset")
            if len(selected) >= top_k:
                break

    # PRIORITY 4: MATCH BASED ON GEMINI'S ANSWER (if room remains)
    if len(selected) < top_k:
        for obj in catalog.get("objects", []):
            if obj["class"].lower() in ans_lower:
                add_item(obj["snapshot_path"], obj["caption"], f"{obj['class'].title()} Asset Exhibit")
                if len(selected) >= top_k:
                    break

    if len(selected) < top_k:
        for inc in catalog.get("incidents", []):
            add_item(inc["path"], inc["caption"], "Forensic Incident Snapshot")
            if len(selected) >= top_k:
                break

    # PRIORITY 5: FALLBACK
    if not selected:
        if catalog.get("incidents"):
            inc = catalog["incidents"][0]
            add_item(inc["path"], inc["caption"], "Forensic Incident Snapshot")
        if catalog.get("suspects"):
            s = catalog["suspects"][0]
            add_item(s["path"], s["caption"], "Suspect Profile")

    return selected[:top_k]
