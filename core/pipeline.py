import os
from vision.tracking import Tracker
from vision.appearance import AppearanceFeatureExtractor
from vision.cross_camera import CrossCameraMatcher
from vision.event_correlation import EventExtractor
from vision.report import ReportGenerator
from vision.object_detector import ObjectDetector
from vision.activity_status import ActivityStatusEngine, ActivityConfig

class CaselinePipeline:
    def __init__(self):
        self.tracker = Tracker()
        self.appearance = AppearanceFeatureExtractor()
        self.matcher = CrossCameraMatcher()
        self.events = EventExtractor()
        self.reporter = ReportGenerator()
        self.object_detector = ObjectDetector()
        self.activity_config = ActivityConfig()

    def analyze_videos(self, case_id: str, case_dir: str):
        # 1. Run Tracking on all videos (UNCHANGED)
        videos_dir = os.path.join(case_dir, "videos")
        evidence_dir = os.path.join(case_dir, "evidence")
        output_dir = os.path.join(case_dir, "output")

        for video_file in os.listdir(videos_dir):
            if not video_file.endswith(('.mp4', '.webm', '.avi')): continue
            camera_id = video_file.split("_")[0]
            vid_path = os.path.join(videos_dir, video_file)
            csv_path = os.path.join(evidence_dir, f"{camera_id}_tracking.csv")

            # Skip if already processed
            if not os.path.exists(csv_path):
                print(f"Tracking {camera_id}...")
                self.tracker.process_video(case_id, camera_id, vid_path, csv_path)

        # 2. Detect and track objects in all videos (Temporal Stability & Diagnostics)
        print("Detecting and tracking objects...")
        global_diag_csv = os.path.join("output", "object_detection_diagnostics.csv")
        case_diag_csv = os.path.join(output_dir, "object_detection_diagnostics.csv")
        # Initialize diagnostics CSV
        if os.path.exists(global_diag_csv):
            try:
                os.remove(global_diag_csv)
            except Exception:
                pass

        for video_file in os.listdir(videos_dir):
            if not video_file.endswith(('.mp4', '.webm', '.avi')): continue
            camera_id = video_file.split("_")[0]
            vid_path = os.path.join(videos_dir, video_file)
            obj_csv = os.path.join(evidence_dir, f"{camera_id}_objects.csv")

            print(f"  Object detection & tracking on {camera_id}...")
            self.object_detector.detect_and_track_objects(
                vid_path, obj_csv, diagnostics_csv=global_diag_csv
            )

        # Copy global diagnostics to case output
        if os.path.exists(global_diag_csv):
            import shutil
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(global_diag_csv, case_diag_csv)

        # 3. Extract Appearance features (UNCHANGED)
        print("Extracting appearances...")
        self.appearance.process_case(case_dir)

        # 4. Compute activity status per camera
        #    and render videos with BLUE/YELLOW/RED boxes and object diagnostics
        print("Computing activity status...")
        for video_file in os.listdir(videos_dir):
            if not video_file.endswith(('.mp4', '.webm', '.avi')): continue
            camera_id = video_file.split("_")[0]
            vid_path = os.path.join(videos_dir, video_file)
            csv_path = os.path.join(evidence_dir, f"{camera_id}_tracking.csv")
            obj_csv = os.path.join(evidence_dir, f"{camera_id}_objects.csv")
            out_vid = os.path.join(output_dir, f"{camera_id}_tracking.webm")
            transitions_csv = os.path.join(evidence_dir, f"{camera_id}_activity_transitions.csv")

            # Get FPS from the video
            import cv2
            cap = cv2.VideoCapture(vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
            cap.release()
            if fps <= 0:
                fps = 25.0

            # Run the activity-status engine for this camera
            print(f"  Activity analysis for {camera_id}:")
            engine = ActivityStatusEngine(config=self.activity_config)
            activity_data = engine.process_camera(csv_path, obj_csv, fps=fps)
            engine.save_transitions(transitions_csv)

            # Print summary
            for tid, info in activity_data.items():
                final = info["final_status"]
                score = info["score"]
                symbol = {"NORMAL": "🔵", "ATTENTION": "🟡", "SUSPICIOUS": "🔴"}.get(final, "⚪")
                print(f"    {symbol} Track {tid}: {final} (score: {score})")

            # Render annotated video with activity-status colours & object diagnostics
            print(f"  Rendering {camera_id} with activity status & object diagnostics...")
            self.tracker.render_video(camera_id, vid_path, csv_path, out_vid,
                                      activity_data=activity_data,
                                      objects_csv=obj_csv)

        # 5. Candidate Matching (UNCHANGED)
        print("Matching tracks...")
        self.matcher.match_tracks(case_dir)

    def process_approvals_and_finish(self, case_id: str, case_dir: str, approved_links: list):
        print("Processing human approvals...")
        self.matcher.process_human_approvals(case_dir, approved_links)

        print("Extracting events...")
        self.events.extract_events(case_dir)

        print("Generating report...")
        return self.reporter.generate_report(case_id, case_dir)
