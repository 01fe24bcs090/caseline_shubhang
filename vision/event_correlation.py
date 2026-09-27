import pandas as pd
import os

class EventExtractor:
    """Extracts high-level incident events from the linked subject evidence."""
    
    def extract_events(self, case_dir: str):
        evidence_dir = os.path.join(case_dir, "evidence")
        links_file = os.path.join(evidence_dir, "subject_links.csv")
        events_file = os.path.join(evidence_dir, "correlated_events.csv")
        
        if not os.path.exists(links_file):
            return pd.DataFrame()
            
        df = pd.read_csv(links_file)
        if df.empty:
            return pd.DataFrame()
            
        events = []
        for subject_id in df["subject_id"].unique():
            subject_data = df[df["subject_id"] == subject_id].copy()
            subject_data = subject_data.sort_values("first_seen")
            
            previous_camera = None
            for _, row in subject_data.iterrows():
                camera = row["camera_id"]
                track = int(row["track_id"])
                first_seen = float(row["first_seen"])
                
                if previous_camera is None:
                    event_type = "SUBJECT OBSERVED"
                else:
                    if camera == previous_camera:
                        event_type = "CONTINUED IN CAMERA"
                    else:
                        event_type = "CAMERA TRANSITION"
                        
                events.append({
                    "timestamp": first_seen,
                    "subject_id": subject_id,
                    "camera_id": camera,
                    "track_id": track,
                    "event_type": event_type,
                    "description": f"{subject_id} observed in {camera}",
                    "confidence": round(float(row["avg_confidence"]), 3)
                })
                
                previous_camera = camera
                
            last_row = subject_data.iloc[-1]
            events.append({
                "timestamp": float(last_row["last_seen"]),
                "subject_id": subject_id,
                "camera_id": last_row["camera_id"],
                "track_id": int(last_row["track_id"]),
                "event_type": "LAST OBSERVED",
                "description": f"{subject_id} last observed in {last_row['camera_id']}",
                "confidence": round(float(last_row["avg_confidence"]), 3)
            })
            
        events_df = pd.DataFrame(events)
        if not events_df.empty:
            events_df = events_df.sort_values(by=["timestamp", "subject_id"]).reset_index(drop=True)
            events_df.to_csv(events_file, index=False)
            
            # Create a simple timeline CSV
            timeline = events_df[["timestamp", "subject_id", "camera_id", "event_type", "description"]]
            timeline.to_csv(os.path.join(evidence_dir, "incident_timeline.csv"), index=False)
            
        return events_df