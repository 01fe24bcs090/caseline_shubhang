"""
CaseLine Object Detection Diagnostics Runner.

Executes Step 1 through Step 6:
1. Object detection diagnostics export: output/object_detection_diagnostics.csv
2. Temporal class stability calculations & dominant class reporting
3. Temporal confirmation verification (OBJECT_CONFIDENCE_THRESHOLD, MIN_OBJECT_OBSERVATIONS, MIN_CLASS_STABILITY)
4. Uncertain object handling ('Subject interacted with an object.' instead of hallucinated classes)
5. Visual diagnostic snapshot & video rendering
6. Full diagnostic report compilation
"""
import os
import sys
import csv
import math
from collections import Counter
import cv2
import pandas as pd

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vision.object_detector import ObjectDetector, ObjectStabilityConfig
from vision.activity_status import ActivityStatusEngine, ActivityConfig
from vision.tracking import Tracker

def run_diagnostics(case_id=None):
    cases_dir = os.path.join("data", "cases")
    os.makedirs(cases_dir, exist_ok=True)
    
    if not case_id:
        existing = [d for d in os.listdir(cases_dir) if os.path.isdir(os.path.join(cases_dir, d))]
        if "new001" in existing:
            case_id = "new001"
        elif existing:
            case_id = existing[0]
        else:
            case_id = "DEMO-001"
            from core.case import CaseManager
            cm = CaseManager()
            try:
                cm.create_case(case_id, "Diagnostics Auto-Setup Case")
            except ValueError:
                pass
            for sv in ["cctv.webm", "cctv2.webm"]:
                svp = os.path.join("data", "videos", sv)
                if os.path.exists(svp):
                    cm.ingest_video(case_id, svp)

    print("=" * 70)
    print(f"CASELINE — OBJECT DETECTION DIAGNOSTIC STEP (Case: {case_id})")
    print("=" * 70)
    
    case_dir = os.path.join("data", "cases", case_id)
    videos_dir = os.path.join(case_dir, "videos")
    evidence_dir = os.path.join(case_dir, "evidence")
    output_dir = os.path.join(case_dir, "output")
    root_output = "output"
    
    os.makedirs(root_output, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    global_diag_csv = os.path.join(root_output, "object_detection_diagnostics.csv")
    stability_summary_csv = os.path.join(root_output, "object_stability_summary.csv")
    
    # Reset diagnostic CSV
    if os.path.exists(global_diag_csv):
        os.remove(global_diag_csv)
        
    config = ObjectStabilityConfig(
        OBJECT_CONFIDENCE_THRESHOLD=0.60,
        MIN_OBJECT_OBSERVATIONS=5,
        MIN_CLASS_STABILITY=0.70,
    )
    detector = ObjectDetector(config=config)
    tracker = Tracker()
    activity_cfg = ActivityConfig(
        OBJECT_CONFIDENCE_THRESHOLD=0.60,
        MIN_OBJECT_OBSERVATIONS=5,
        MIN_CLASS_STABILITY=0.70,
    )
    
    all_camera_results = {}
    
    # ── STEP 1: Process Object Detections & Tracking on CCTV Videos ──
    print("\n--- STEP 1: RUNNING DETECTION & TRACKING DIAGNOSTICS ---")
    
    video_files = sorted([f for f in os.listdir(videos_dir) if f.endswith(('.mp4', '.webm'))])
    
    for vfile in video_files:
        cam_id = vfile.split("_")[0]
        vpath = os.path.join(videos_dir, vfile)
        obj_csv = os.path.join(evidence_dir, f"{cam_id}_objects.csv")
        trk_csv = os.path.join(evidence_dir, f"{cam_id}_tracking.csv")
        
        print(f"\nProcessing {cam_id} ({vfile})...")
        total_dets, tracked_objs = detector.detect_and_track_objects(
            vpath, obj_csv, frame_skip=2, conf=0.30, diagnostics_csv=global_diag_csv
        )
        all_camera_results[cam_id] = {
            "vpath": vpath,
            "obj_csv": obj_csv,
            "trk_csv": trk_csv,
            "tracked_objs": tracked_objs,
        }
        
    # Copy global diagnostics to case output folder
    import shutil
    shutil.copy2(global_diag_csv, os.path.join(output_dir, "object_detection_diagnostics.csv"))
    print(f"\n[STEP 1 COMPLETE] Created {global_diag_csv}")
    
    # ── STEP 2: CLASS STABILITY REPORTING ──
    print("\n" + "=" * 70)
    print("STEP 2: CLASS STABILITY ACROSS TRACKED OBJECTS")
    print("=" * 70)
    
    summary_rows = []
    
    for cam_id, res in all_camera_results.items():
        print(f"\n[{cam_id}] Tracked Objects Summary:")
        tracked_objs = res["tracked_objs"]
        
        for oid, tobj in sorted(tracked_objs.items()):
            metrics = tobj.get_stability_metrics()
            tot = metrics["total_observations"]
            if tot < 2:
                continue  # skip transient single-frame noise
                
            dom_class = metrics["dominant_class"]
            stab_pct = metrics["class_stability"] * 100.0
            avg_conf = metrics["dominant_avg_conf"]
            is_conf = tobj.is_confirmed(config)
            eff_class = tobj.get_effective_class(config)
            
            # Print example format
            print(f"\nObject {oid} ({cam_id}):")
            for cls_name, cnt in metrics["breakdown"].items():
                print(f"  {cls_name:<12} {cnt:>3} frames")
            print(f"  dominant_class   = {dom_class}")
            print(f"  class_stability  = {stab_pct:.1f}%")
            print(f"  avg_confidence   = {avg_conf:.2f}")
            print(f"  status           = {'CONFIRMED (' + eff_class + ')' if is_conf else 'UNCERTAIN OBJECT'}")
            
            summary_rows.append({
                "camera_id": cam_id,
                "object_id": oid,
                "total_observations": tot,
                "dominant_class": dom_class,
                "class_stability_pct": round(stab_pct, 1),
                "dominant_avg_confidence": round(avg_conf, 2),
                "is_confirmed": int(is_conf),
                "effective_class": eff_class,
                "class_breakdown": str(metrics["breakdown"]),
            })
            
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(stability_summary_csv, index=False)
    print(f"\nSaved stability summary table to: {stability_summary_csv}")
    
    # ── STEP 3 & 4: ACTIVITY STATUS ENGINE & UNCERTAIN OBJECT EVALUATION ──
    print("\n" + "=" * 70)
    print("STEP 3 & 4: ACTIVITY STATUS ENGINE WITH TEMPORAL CONFIRMATION")
    print("=" * 70)
    
    for cam_id, res in all_camera_results.items():
        vpath = res["vpath"]
        trk_csv = res["trk_csv"]
        obj_csv = res["obj_csv"]
        
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        
        print(f"\nRunning Activity Analysis for {cam_id}:")
        engine = ActivityStatusEngine(config=activity_cfg)
        act_data = engine.process_camera(trk_csv, obj_csv, fps=fps)
        
        trans_csv = os.path.join(evidence_dir, f"{cam_id}_activity_transitions.csv")
        engine.save_transitions(trans_csv)
        
        print(f"Recorded activity transitions for {cam_id}:")
        tdf = pd.read_csv(trans_csv) if os.path.exists(trans_csv) else pd.DataFrame()
        if tdf.empty:
            print("  (no transitions)")
        else:
            for _, r in tdf.iterrows():
                print(f"  Frame {int(r['frame'])} | Track {int(r['track_id'])} -> {r['state']} | {r['reason']}")
                
        # ── STEP 5: VISUAL DIAGNOSTIC RENDERING ──
        out_vid = os.path.join(root_output, f"{cam_id}_diagnostic.webm")
        case_vid = os.path.join(output_dir, f"{cam_id}_tracking.webm")
        print(f"\nRendering Visual Diagnostic Video for {cam_id}...")
        tracker.render_video(
            cam_id, vpath, trk_csv, out_vid,
            activity_data=act_data, objects_csv=obj_csv
        )
        if os.path.exists(out_vid):
            shutil.copy2(out_vid, case_vid)
            
    # Save visual diagnostic snapshots (images) for clear inspection
    print("\n" + "=" * 70)
    print("STEP 5: EXPORTING VISUAL DIAGNOSTIC SNAPSHOTS")
    print("=" * 70)
    
    snapshots = [
        # (camera_id, frame_no, filename, description)
        ("CAM02", 150, "diagnostic_laptop_confirmed.jpg", "Laptop confirmed detection (high conf, stable)"),
        ("CAM02", 456, "diagnostic_suitcase_frame.jpg", "Real suitcase entering scene"),
        ("CAM02", 540, "diagnostic_flickering_chair_obj.jpg", "Flickering backpack/suitcase object rendered as Uncertain object"),
        ("CAM03", 226, "diagnostic_cam03_scarf.jpg", "Cam03 sweater/scarf rendered as Uncertain object"),
    ]
    
    for cam_id, fnum, sname, desc in snapshots:
        if cam_id in all_camera_results:
            vpath = all_camera_results[cam_id]["vpath"]
            obj_csv = all_camera_results[cam_id]["obj_csv"]
            
            cap = cv2.VideoCapture(vpath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fnum)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                odf = pd.read_csv(obj_csv)
                fobjs = odf[odf["frame"] == fnum]
                for _, ob in fobjs.iterrows():
                    ox1, oy1, ox2, oy2 = int(ob["x1"]), int(ob["y1"]), int(ob["x2"]), int(ob["y2"])
                    conf_val = float(ob["confidence"])
                    is_c = bool(ob["is_confirmed"])
                    dom_c = str(ob["dominant_class"])
                    
                    if is_c and dom_c.lower() != "uncertain object":
                        lbl = f"{dom_c.capitalize()} {conf_val:.2f}"
                        color = (255, 255, 0)
                        txt_c = (0, 0, 0)
                    else:
                        lbl = f"Uncertain object {conf_val:.2f}"
                        color = (0, 140, 255)
                        txt_c = (255, 255, 255)
                        
                    cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (ox1, max(0, oy1 - th - 6)), (ox1 + tw + 6, oy1), color, -1)
                    cv2.putText(frame, lbl, (ox1 + 3, oy1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_c, 1, cv2.LINE_AA)
                    
                spath = os.path.join(root_output, sname)
                cv2.imwrite(spath, frame)
                print(f"  Exported snapshot: {spath} ({desc})")

    print("\n" + "=" * 70)
    print("ALL DIAGNOSTIC STEPS EXECUTED SUCCESSFULLY")
    print("=" * 70)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_diagnostics(target)
