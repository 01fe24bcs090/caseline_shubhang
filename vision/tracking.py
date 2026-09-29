import os
import cv2
import csv
import pandas as pd
from ultralytics import YOLO

# Locate bundled yolo11n.pt relative to project root
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_MODEL = os.path.join(_ROOT_DIR, "yolo11n.pt")
DEFAULT_MODEL_PATH = _LOCAL_MODEL if os.path.exists(_LOCAL_MODEL) else "yolo11n.pt"

class Tracker:
    def __init__(self, model_path=DEFAULT_MODEL_PATH): # Using Nano model for huge CPU speedup
        self.model = YOLO(model_path)
        
    def process_video(self, case_id: str, camera_id: str, video_path: str, output_csv: str, frame_skip: int = 2):
        """Processes a single video for a given case and extracts tracking evidence to CSV."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
            
        # Prepare CSV
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        csv_file = open(output_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "frame", "timestamp", "camera_id", "track_id", 
            "confidence", "x1", "y1", "x2", "y2"
        ])
        
        frame_number = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_number += 1
            
            # Massive CPU optimization: Only process every Nth frame
            if frame_number % frame_skip != 0:
                continue
            
            # Low-light enhancement (CLAHE)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced_lab = cv2.merge((l, a, b))
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Tracking
            results = self.model.track(
                enhanced_frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=0.50, # Slightly lower confidence for Nano model
                iou=0.45,
                classes=[0], # person only
                verbose=False
            )
            
            result = results[0]
            has_detections = result.boxes is not None and result.boxes.id is not None
            
            if has_detections:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for track_id, box, conf in zip(track_ids, boxes, confidences):
                    x1, y1, x2, y2 = box
                    timestamp = frame_number / fps if fps > 0 else 0
                    
                    csv_writer.writerow([
                        frame_number,
                        round(timestamp, 2),
                        camera_id,
                        track_id,
                        round(float(conf), 3),
                        int(x1), int(y1), int(x2), int(y2)
                    ])
                    
        cap.release()
        csv_file.close()

    def render_video(self, camera_id: str, video_path: str, csv_path: str,
                     output_video: str, activity_data: dict = None,
                     objects_csv: str = None):
        """Renders the final video with activity-status bounding boxes,
        trimmed to person entry/exit. Also displays detected/tracked objects
        with temporal confirmation status (e.g. Laptop 0.91 or Uncertain object 0.43).

        Box colours:
          BLUE   (255,0,0)   – NORMAL person
          YELLOW (0,255,255)  – ATTENTION person
          RED    (0,0,255)    – SUSPICIOUS ACTIVITY person
          CYAN   (255,255,0)  – CONFIRMED OBJECT
          ORANGE (0,140,255)  – UNCERTAIN OBJECT

        Parameters
        ----------
        activity_data : dict | None
            Output of ActivityStatusEngine.process_camera().
            When None every person is shown as NORMAL (blue).
        objects_csv : str | None
            CSV of tracked objects to render visual diagnostics.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Load person tracking CSV into dictionary for fast lookup by frame
        frame_data = {}
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if not df.empty:
                for frame_num, group in df.groupby("frame"):
                    frame_data[int(frame_num)] = group.to_dict("records")

        # Load object detections if provided
        obj_frame_data = {}
        if objects_csv and os.path.exists(objects_csv):
            odf = pd.read_csv(objects_csv)
            if not odf.empty:
                for frame_num, group in odf.groupby("frame"):
                    obj_frame_data[int(frame_num)] = group.to_dict("records")

        # Determine trim boundaries: first and last frame with a detection
        if frame_data:
            first_person_frame = min(frame_data.keys())
            last_person_frame = max(frame_data.keys())
        else:
            cap.release()
            print(f"No detections found for {camera_id}, skipping video render.")
            return

        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        # Use VP80 for .webm — vp08 tag causes codec-id mismatch on some OpenCV builds
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not out.isOpened():
            # Fallback: write as .mp4 beside the .webm path if webm fails
            fallback = output_video.replace(".webm", ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(fallback, fourcc, fps, (width, height))
            output_video = fallback

        # Status colours (BGR)
        STATUS_COLORS = {
            "NORMAL":     (255, 0, 0),     # Blue
            "ATTENTION":  (0, 255, 255),    # Yellow
            "SUSPICIOUS": (0, 0, 255),      # Red
        }
        STATUS_TEXT_COLORS = {
            "NORMAL":     (255, 255, 255),  # White on blue
            "ATTENTION":  (0, 0, 0),        # Black on yellow
            "SUSPICIOUS": (255, 255, 255),  # White on red
        }
        STATUS_LABELS = {
            "NORMAL":     "NORMAL",
            "ATTENTION":  "ATTENTION",
            "SUSPICIOUS": "SUSPICIOUS ACTIVITY",
        }

        frame_number = 0
        last_boxes = []
        last_obj_boxes = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Skip frames before the first person enters
            if frame_number < first_person_frame:
                continue

            # Stop after the last person exits
            if frame_number > last_person_frame:
                break

            # Calculate and draw CCTV timestamp
            timestamp = frame_number / fps if fps > 0 else 0
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            time_str = f"TIME: {minutes:02d}:{seconds:02d}"

            # Add a black background rectangle for the text to make it readable
            cv2.rectangle(frame, (10, 10), (230, 60), (0, 0, 0), -1)
            # Draw the timestamp text (green like CCTV)
            cv2.putText(frame, time_str, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Update person boxes if we have new data for this frame
            if frame_number in frame_data:
                last_boxes = frame_data[frame_number]

            # Update object boxes if we have new data for this frame
            if frame_number in obj_frame_data:
                last_obj_boxes = obj_frame_data[frame_number]
            elif (frame_number - 1) not in obj_frame_data and (frame_number - 2) not in obj_frame_data:
                # Expire objects if not seen in recent frames
                last_obj_boxes = []

            # ── Draw detected objects (Visual Diagnostic) ──
            for ob in last_obj_boxes:
                ox1 = int(ob["x1"])
                oy1 = int(ob["y1"])
                ox2 = int(ob["x2"])
                oy2 = int(ob["y2"])
                conf_val = float(ob.get("confidence", 0.0))
                
                # Check confirmation status
                is_conf = bool(ob.get("is_confirmed", 0))
                pred_cls = str(ob.get("predicted_class", ob.get("class_name", "object")))
                dom_cls = str(ob.get("dominant_class", pred_cls))

                if is_conf and dom_cls and dom_cls.lower() != "uncertain object":
                    label_text = f"{dom_cls.capitalize()} {conf_val:.2f}"
                    obj_color = (255, 255, 0)   # Cyan (BGR) for confirmed
                    obj_text_color = (0, 0, 0)
                else:
                    label_text = f"Uncertain object {conf_val:.2f}"
                    obj_color = (0, 140, 255)   # Amber / Orange for uncertain
                    obj_text_color = (255, 255, 255)

                cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), obj_color, 2)
                (ow, oh), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                obj_label_top = max(0, oy1 - oh - 6)
                cv2.rectangle(frame, (ox1, obj_label_top), (ox1 + ow + 6, obj_label_top + oh + 6), obj_color, -1)
                cv2.putText(frame, label_text, (ox1 + 3, obj_label_top + oh + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, obj_text_color, 1, cv2.LINE_AA)

            # ── Draw person boxes with activity-status colouring ──
            for box_info in last_boxes:
                x1 = int(box_info["x1"])
                y1 = int(box_info["y1"])
                x2 = int(box_info["x2"])
                y2 = int(box_info["y2"])
                track_id = int(box_info["track_id"])

                # Determine status at this frame
                status, reason = self._get_track_status(
                    activity_data, track_id, frame_number
                )

                color = STATUS_COLORS.get(status, (255, 0, 0))
                text_color = STATUS_TEXT_COLORS.get(status, (255, 255, 255))
                thickness = 3 if status == "SUSPICIOUS" else 2

                # ── Bounding box ──
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                # ── Top label: "Person {id}" ──
                person_label = f"Person {track_id}"
                (pw, ph), _ = cv2.getTextSize(
                    person_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                label_top = max(0, y1 - ph - 30)
                cv2.rectangle(
                    frame,
                    (x1, label_top),
                    (x1 + pw + 6, label_top + ph + 6),
                    color, -1,
                )
                cv2.putText(
                    frame, person_label,
                    (x1 + 3, label_top + ph + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2,
                )

                # ── Status badge below person label ──
                status_text = STATUS_LABELS.get(status, "NORMAL")
                (sw, sh), _ = cv2.getTextSize(
                    status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                badge_top = label_top + ph + 8
                cv2.rectangle(
                    frame,
                    (x1, badge_top),
                    (x1 + sw + 6, badge_top + sh + 6),
                    color, -1,
                )
                cv2.putText(
                    frame, status_text,
                    (x1 + 3, badge_top + sh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1,
                )

                # ── Reason text below the box (non-NORMAL only) ──
                if status != "NORMAL" and reason:
                    short = reason.split(";")[0].strip()
                    if len(short) > 40:
                        short = short[:37] + "..."
                    cv2.putText(
                        frame, short,
                        (x1, min(height - 10, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
                    )

            out.write(frame)

        cap.release()
        out.release()

    # ── helper ──────────────────────────────────────────────────
    @staticmethod
    def _get_track_status(activity_data, track_id, frame_number):
        """Look up the status of a track at a given frame.

        Returns (status_str, latest_reason_str).
        """
        if activity_data is None or track_id not in activity_data:
            return "NORMAL", ""

        transitions = activity_data[track_id]["transitions"]
        status = "NORMAL"
        reason = ""
        for t in transitions:
            if t["frame"] <= frame_number:
                status = t["status"]
                reason = t.get("reason", "")
            else:
                break
        return status, reason

