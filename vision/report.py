"""
CASELINE -- Report Generator
Produces a human-understandable Case Report (TXT + PDF) from the actual
pipeline outputs.  Nothing is hardcoded -- all evidence, statuses, and
reasons come from the files generated during analysis.
"""
from __future__ import annotations

import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from core.case import CaseManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_seconds(sec: float) -> str:
    """Convert fractional seconds -> MM:SS string."""
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


def _fmt_size(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def _severity(state: str) -> int:
    return {"NORMAL": 0, "ATTENTION": 1, "SUSPICIOUS": 2}.get(state.upper(), 0)


def _short_id(subject_id: str, all_ids: List[str]) -> str:
    try:
        idx = all_ids.index(subject_id) + 1
        if "Track" in subject_id:
            cam = subject_id.split(" ")[0]
            return f"{idx:02d} (Unlinked subject in {cam})"
        return f"{idx:02d} ({subject_id})"
    except ValueError:
        return subject_id


STATUS_SYMBOL = {
    "NORMAL":     "NORMAL",
    "ATTENTION":  "ATTENTION",
    "SUSPICIOUS": "SUSPICIOUS ACTIVITY",
}
STATUS_ICON = {
    "NORMAL":     "[NORMAL]",
    "ATTENTION":  "[ATTENTION]",
    "SUSPICIOUS": "[SUSPICIOUS]",
}
EVENT_SENTENCE = {
    "SUBJECT OBSERVED":    "was first observed in {cam} at {t}",
    "CAMERA TRANSITION":   "moved to {cam} at {t}",
    "CONTINUED IN CAMERA": "continued to be observed in {cam} at {t}",
    "LAST OBSERVED":       "was last observed in {cam} at {t}",
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generates a human-understandable CASELINE Case Report (TXT + PDF)."""

    def generate_report(self, case_id: str, case_dir: str) -> str:
        manager   = CaseManager(cases_dir=os.path.dirname(case_dir))
        case_info = manager.load_case_info(case_id)

        evidence_dir = os.path.join(case_dir, "evidence")
        output_dir   = os.path.join(case_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        # Load core evidence tables
        subjects_file = os.path.join(evidence_dir, "subject_links.csv")
        events_file   = os.path.join(evidence_dir, "correlated_events.csv")
        subjects = pd.read_csv(subjects_file) if os.path.exists(subjects_file) else pd.DataFrame()
        events   = pd.read_csv(events_file)   if os.path.exists(events_file)   else pd.DataFrame()

        subject_ids: List[str] = (
            sorted(subjects["subject_id"].unique().tolist()) if not subjects.empty else []
        )

        # Load per-camera activity transitions
        cam_transitions: Dict[str, pd.DataFrame] = {}
        for f in os.listdir(evidence_dir):
            if f.endswith("_activity_transitions.csv"):
                cam = f.replace("_activity_transitions.csv", "")
                df  = pd.read_csv(os.path.join(evidence_dir, f))
                if not df.empty:
                    cam_transitions[cam] = df

        # Map (camera, track_id) -> final activity status + reasons
        track_final: Dict[Tuple[str, int], dict] = {}
        for cam, tdf in cam_transitions.items():
            for _, row in tdf.iterrows():
                tid  = int(row["track_id"])
                key  = (cam, tid)
                st   = str(row.get("state", "NORMAL")).strip()
                rsn  = str(row.get("reason", "")).strip()
                prev = track_final.get(key, {"state": "NORMAL", "reasons": [], "cam": cam})
                if _severity(st) >= _severity(prev["state"]):
                    prev["state"] = st
                    if rsn:
                        prev["reasons"] = [r.strip() for r in rsn.split(";") if r.strip()]
                    prev["cam"] = cam
                track_final[key] = prev

        # Identify covered tracks from subject_links.csv
        covered_tracks = set()
        if not subjects.empty:
            for _, row in subjects.iterrows():
                covered_tracks.add((str(row["camera_id"]), int(row["track_id"])))

        # Add unlinked tracks as local subjects
        local_subjects = []
        for (cam, tid) in track_final.keys():
            if (cam, tid) not in covered_tracks:
                sid = f"{cam} Track {tid}"
                local_subjects.append(sid)
                new_row = pd.DataFrame([{
                    "subject_id": sid,
                    "camera_id": cam,
                    "track_id": tid,
                    "first_seen": 0.0,
                    "last_seen": 0.0,
                    "avg_confidence": 0.0,
                    "observations": 0
                }])
                subjects = pd.concat([subjects, new_row], ignore_index=True) if not subjects.empty else new_row

        subject_ids.extend(local_subjects)
        subject_ids.sort()

        # Build per-subject activity status
        subject_status: Dict[str, dict] = {}
        for sid in subject_ids:
            sid_rows = subjects[subjects["subject_id"] == sid]
            worst_state   = "NORMAL"
            worst_reasons: List[str] = []
            for _, row in sid_rows.iterrows():
                cam = str(row["camera_id"])
                tid = int(row["track_id"])
                tf  = track_final.get((cam, tid), {"state": "NORMAL", "reasons": []})
                if _severity(tf["state"]) > _severity(worst_state):
                    worst_state   = tf["state"]
                    worst_reasons = tf["reasons"]
            subject_status[sid] = {"state": worst_state, "reasons": worst_reasons}

        # Case-level status
        if any(v["state"] == "SUSPICIOUS" for v in subject_status.values()):
            case_status = "SUSPICIOUS ACTIVITY DETECTED"
        elif any(v["state"] == "ATTENTION" for v in subject_status.values()):
            case_status = "SUBJECTS UNDER ATTENTION"
        else:
            case_status = "NO SUSPICIOUS ACTIVITY DETECTED"

        avg_conf   = subjects["avg_confidence"].mean() if not subjects.empty else 0.0
        conf_label = "HIGH" if avg_conf >= 0.85 else "MEDIUM" if avg_conf >= 0.70 else "LOW"
        total_dur  = max((v.duration_sec for v in case_info.videos), default=0.0)

        # Build report
        R: List[str] = []

        def h(title: str):
            R.append("\n" + "=" * 70)
            R.append(f"  {title}")
            R.append("=" * 70)

        # HEADER
        R.append("=" * 70)
        R.append("                        CASELINE")
        R.append("              INVESTIGATION CASE REPORT")
        R.append("=" * 70)
        R.append(f"\n  Report Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        R.append(f"  Case ID           : {case_info.case_id}")
        R.append(f"  Incident          : {case_info.incident_description}")
        R.append(f"  Case Status       : {case_status}")
        R.append(f"  Human Review      : PENDING -- Human verification required")

        # SECTION 1 -- CASE OVERVIEW
        h("SECTION 1 -- CASE OVERVIEW")
        R.append(f"\n  Case ID              : {case_info.case_id}")
        R.append(f"  Incident Description : {case_info.incident_description}")
        R.append(f"  Case Created         : {case_info.created_at}")
        R.append(f"  Cameras Analysed     : {len(case_info.videos)}")
        R.append(f"  Subjects Identified  : {len(subject_ids)}")
        R.append(f"  Analysis Duration    : {_fmt_seconds(total_dur)} (longest feed)")
        R.append(f"  Overall Case Status  : {case_status}")
        R.append(f"  Subject-Link Confidence : {conf_label} ({avg_conf:.1%})")

        # SECTION 2 -- INVESTIGATION SUMMARY
        h("SECTION 2 -- INVESTIGATION SUMMARY")
        R.append("")
        R.append(self._build_summary(case_info, subject_ids, subject_status, events))

        # SECTION 3 -- SUBJECT ACTIVITY STATUS
        h("SECTION 3 -- SUBJECT ACTIVITY STATUS")
        if subject_ids:
            for idx, sid in enumerate(subject_ids, 1):
                ss  = subject_status[sid]
                ico = STATUS_ICON.get(ss["state"], "[UNKNOWN]")
                
                if "Track" in sid:
                    cam = sid.split(" ")[0]
                    display_name = f"Subject {idx:02d}  (Unlinked subject in {cam})"
                else:
                    display_name = f"Subject {idx:02d}  ({sid})"

                R.append(f"\n  {display_name}")
                R.append(f"  Final Status : {ico} {STATUS_SYMBOL.get(ss['state'], ss['state'])}")
                if not subjects.empty:
                    sid_rows  = subjects[subjects["subject_id"] == sid]
                    cams_seen = sorted(sid_rows["camera_id"].unique())
                    first_t   = sid_rows["first_seen"].min()
                    last_t    = sid_rows["last_seen"].max()
                    R.append(f"  Cameras Seen : {', '.join(cams_seen)}")
                    R.append(f"  First Seen   : {_fmt_seconds(first_t)}")
                    R.append(f"  Last Seen    : {_fmt_seconds(last_t)}")
        else:
            R.append("\n  No subjects were identified in this case.")

        # SECTION 4 -- FLAGGING REASONS
        h("SECTION 4 -- ACTIVITY EVIDENCE & FLAGGING REASONS")
        if subject_ids:
            for idx, sid in enumerate(subject_ids, 1):
                ss = subject_status[sid]
                
                if "Track" in sid:
                    cam = sid.split(" ")[0]
                    display_name = f"Subject {idx:02d}  (Unlinked subject in {cam})"
                else:
                    display_name = f"Subject {idx:02d}  ({sid})"

                R.append(f"\n  {display_name}")
                R.append(f"  Status: {STATUS_ICON.get(ss['state'], '[?]')} "
                         f"{STATUS_SYMBOL.get(ss['state'], ss['state'])}")
                if ss["state"] == "NORMAL":
                    R.append("  No suspicious activity was detected for this subject.")
                    R.append("  The subject was observed moving normally within the monitored area.")
                else:
                    R.append("  The following observable events contributed to the activity flag:")
                    for i, rsn in enumerate(ss["reasons"], 1):
                        R.append(f"    {i}. {rsn}")
                    if ss["state"] == "SUSPICIOUS":
                        R.append("")
                        R.append("  CASELINE detected suspicious activity based on the available evidence.")
                        R.append("  This does NOT constitute a finding of guilt.")
                        R.append("  Human verification is required before any action is taken.")
        else:
            R.append("\n  No subjects were identified.")

        # SECTION 5 -- HUMAN-READABLE INCIDENT TIMELINE
        h("SECTION 5 -- HUMAN-READABLE INCIDENT TIMELINE")
        if not events.empty:
            R.append("")
            for _, row in events.sort_values("timestamp").iterrows():
                sid      = row["subject_id"]
                cam      = row["camera_id"]
                t        = float(row["timestamp"])
                et       = str(row["event_type"])
                template = EVENT_SENTENCE.get(et, "was recorded in {cam} at {t}")
                sentence = template.format(cam=cam, t=_fmt_seconds(t))
                R.append(f"  [{_fmt_seconds(t)}]  Subject {_short_id(sid, subject_ids)} {sentence}.")
        else:
            R.append("\n  No correlated timeline events available.")

        # SECTION 6 -- CROSS-CAMERA MOVEMENT
        h("SECTION 6 -- CROSS-CAMERA MOVEMENT")
        R.append("\n  IMPORTANT: Cross-camera subject linking is based on visual similarity")
        R.append("  and temporal continuity -- NOT facial recognition or biometric identification.")
        R.append("  All links are probabilistic correlations and require human verification.")
        R.append("")
        if subject_ids and not subjects.empty:
            for idx, sid in enumerate(subject_ids, 1):
                sid_rows = subjects[subjects["subject_id"] == sid].sort_values("first_seen")
                R.append(f"  Subject {idx:02d}  ({sid}):")
                if len(sid_rows) == 1:
                    row = sid_rows.iloc[0]
                    R.append(f"    Observed only in {row['camera_id']} "
                             f"({_fmt_seconds(row['first_seen'])} - "
                             f"{_fmt_seconds(row['last_seen'])})")
                else:
                    prev_cam = None
                    for _, row in sid_rows.iterrows():
                        cam = row["camera_id"]
                        if prev_cam is None:
                            R.append(f"    First appeared in {cam} at {_fmt_seconds(row['first_seen'])}.")
                        else:
                            R.append(f"    Subsequently observed in {cam} "
                                     f"at {_fmt_seconds(row['first_seen'])} "
                                     f"(confidence: {row['avg_confidence']:.1%}).")
                        prev_cam = cam
                    last = sid_rows.iloc[-1]
                    R.append(f"    Last observation: {last['camera_id']} "
                             f"at {_fmt_seconds(last['last_seen'])}.")
                R.append("")
        else:
            R.append("  No cross-camera movement data available.")

        # SECTION 7 -- KEY EVIDENCE
        h("SECTION 7 -- KEY EVIDENCE")
        R.append("\n  Activity State Changes:")
        found = False
        for cam, tdf in sorted(cam_transitions.items()):
            for _, row in tdf.iterrows():
                st  = str(row.get("state", "NORMAL"))
                rsn = str(row.get("reason", ""))
                tid = int(row.get("track_id", 0))
                frm = int(row.get("frame", 0))
                if st in ("ATTENTION", "SUSPICIOUS"):
                    found = True
                    R.append(f"    * {cam} / Track {tid} at frame {frm}: "
                             f"{STATUS_ICON.get(st, '[?]')} {STATUS_SYMBOL.get(st, st)}")
                    for r in rsn.split(";"):
                        r = r.strip()
                        if r:
                            R.append(f"        - {r}")
        if not found:
            R.append("    No activity state changes above NORMAL were recorded.")

        disappearances = self._collect_disappearances(cam_transitions)
        if disappearances:
            R.append("\n  Object Disappearance / Removal Events:")
            for ev in disappearances:
                R.append(f"    * {ev}")

        # SECTION 8 -- FINAL ASSESSMENT
        h("SECTION 8 -- FINAL ASSESSMENT")
        R.append("")
        susp_subs = [s for s in subject_ids if subject_status[s]["state"] == "SUSPICIOUS"]
        norm_subs = [s for s in subject_ids if subject_status[s]["state"] == "NORMAL"]
        attn_subs = [s for s in subject_ids if subject_status[s]["state"] == "ATTENTION"]

        if susp_subs:
            R.append(f"  CASELINE detected suspicious activity for {len(susp_subs)} subject(s).")
            R.append("")
            for sid in susp_subs:
                idx = subject_ids.index(sid) + 1
                R.append(f"  Subject {idx:02d} ({sid}):")
                R.append("    Observable evidence -- including object interactions and")
                R.append("    object disappearance events -- triggered the suspicious-activity flag.")
                R.append("    This assessment is based solely on the available CCTV evidence.")
                R.append("")
        elif attn_subs:
            R.append("  One or more subjects attracted attention based on observable activity.")
            R.append("  No subject reached the SUSPICIOUS threshold.")
        else:
            R.append("  No suspicious activity was detected during the monitored period.")
            R.append("  All subjects exhibited normal behaviour throughout.")

        if norm_subs:
            R.append(f"\n  {len(norm_subs)} subject(s) showed no suspicious behaviour "
                     "and remained NORMAL throughout the recording.")

        R.append("")
        R.append("  +-------------------------------------------------------------+")
        R.append("  |  IMPORTANT LEGAL NOTICE                                      |")
        R.append("  |                                                               |")
        R.append("  |  CASELINE does NOT determine guilt.                           |")
        R.append("  |  The system identifies observable activity patterns only.     |")
        R.append("  |  All conclusions require review and verification by a         |")
        R.append("  |  qualified human investigator before any action is taken.     |")
        R.append("  +-------------------------------------------------------------+")

        # SECTION 9 -- HUMAN REVIEW STATUS
        h("SECTION 9 -- HUMAN REVIEW STATUS")
        review_state = getattr(case_info, "status", "INITIALIZED")
        if review_state == "APPROVED":
            R.append("\n  Status : APPROVED")
            R.append("  The subject links and analysis have been reviewed and approved.")
        elif review_state == "REJECTED":
            R.append("\n  Status : REJECTED")
            R.append("  A human investigator has reviewed and rejected these findings.")
        else:
            R.append("\n  Status : PENDING -- Human verification required.")
            R.append("  No human investigator has reviewed this report yet.")
            R.append("  Do not take investigative action solely on this AI-generated report.")

        # SECTION 10 -- EVIDENCE INTEGRITY
        h("SECTION 10 -- EVIDENCE INTEGRITY")
        R.append("")
        for vid in case_info.videos:
            w, h_ = vid.resolution if vid.resolution else (0, 0)
            R.append(f"  {vid.camera_id}  --  {vid.original_filename}")
            R.append(f"    SHA-256   : {vid.sha256}")
            R.append(f"    Size      : {_fmt_size(vid.file_size_bytes)}")
            R.append(f"    Duration  : {_fmt_seconds(vid.duration_sec)}")
            R.append(f"    FPS       : {vid.fps:.2f}")
            R.append(f"    Resolution: {w} x {h_} px")
            R.append("")
        if case_info.access_logs:
            al = case_info.access_logs
            R.append(f"  ACCESS LOG -- {al.original_filename}")
            R.append(f"    SHA-256  : {al.sha256}")
            R.append("")

        # SECTION 11 -- TECHNICAL DETAILS
        h("SECTION 11 -- TECHNICAL DETAILS (APPENDIX)")
        R.append("\n  Cross-Camera Subject Links:")
        if not subjects.empty:
            for sid in subject_ids:
                R.append(f"\n  {sid}:")
                for _, row in subjects[subjects["subject_id"] == sid].iterrows():
                    R.append(f"    {row['camera_id']} | Track {int(row['track_id'])} | "
                             f"{row['first_seen']:.2f}s - {row['last_seen']:.2f}s | "
                             f"Observations: {int(row['observations'])} | "
                             f"Confidence: {row['avg_confidence']:.3f}")
        else:
            R.append("  No subject links available.")

        R.append("\n  Per-Camera Activity State Summary:")
        for cam, tdf in sorted(cam_transitions.items()):
            R.append(f"\n  {cam}:")
            for _, row in tdf.iterrows():
                frm = int(row.get("frame", 0))
                tid = int(row.get("track_id", 0))
                st  = str(row.get("state", ""))
                rsn = str(row.get("reason", ""))
                R.append(f"    Frame {frm:5d} | Track {tid} | {STATUS_ICON.get(st, '[?]')} {st}")
                for r in rsn.split(";"):
                    r = r.strip()
                    if r:
                        R.append(f"               -> {r}")

        R.append("")
        R.append("  Detection System:")
        R.append("    Person Tracking     : YOLO11-nano + ByteTrack")
        R.append("    Object Detection    : YOLO11-nano (temporal stability confirmed)")
        R.append("    Activity Engine     : BLUE -> YELLOW -> RED evidence-point state machine")
        R.append("    Face Recognition    : NOT USED")
        R.append("    Subject Linking     : Visual/temporal similarity (NOT biometric)")

        # FOOTER
        R.append("\n" + "=" * 70)
        R.append("  Evidence first.  AI second.  Human decision always.")
        R.append("=" * 70)

        # Save TXT
        txt_path = os.path.join(output_dir, "CASELINE_CASE_REPORT.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(R))
        print(f"  Report saved: {txt_path}")

        # Save PDF
        self._generate_pdf(
            os.path.join(output_dir, "CASELINE_CASE_REPORT.pdf"),
            "\n".join(R),
            case_info.case_id,
            case_info.incident_description,
        )
        return txt_path

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_summary(case_info, subject_ids, subject_status, events) -> str:
        n_subj   = len(subject_ids)
        n_cam    = len(case_info.videos)
        incident = case_info.incident_description

        susp = [s for s in subject_ids if subject_status[s]["state"] == "SUSPICIOUS"]
        attn = [s for s in subject_ids if subject_status[s]["state"] == "ATTENTION"]

        lines: List[str] = []
        lines.append(f"  This investigation analysed {n_cam} camera feed(s) for case "
                     f'"{case_info.case_id}" (incident: {incident}).')
        if n_subj:
            lines.append(f"  A total of {n_subj} distinct subject(s) were identified across "
                         "the monitored camera views.")
        else:
            lines.append("  No subjects were identified in the analysed footage.")

        if susp:
            lines.append(f"  {len(susp)} subject(s) were flagged as [SUSPICIOUS] based on observable "
                         "activity including object interactions and object disappearance events.")
        elif attn:
            lines.append("  Some subjects were elevated to [ATTENTION] based on observable "
                         "activity, but no subject reached the SUSPICIOUS threshold.")
        else:
            lines.append("  All subjects exhibited normal behaviour throughout the recorded period.")
            lines.append("  No suspicious activity was detected by the automated analysis.")

        if not events.empty:
            first_t = events["timestamp"].min()
            last_t  = events["timestamp"].max()
            lines.append(f"  The correlated activity spans from {_fmt_seconds(first_t)} "
                         f"to {_fmt_seconds(last_t)} across the available footage.")

        lines.append("")
        lines.append("  NOTE: This summary is generated automatically from the CASELINE")
        lines.append("  pipeline outputs. It requires human review before any action is taken.")
        return "\n".join(lines)

    @staticmethod
    def _collect_disappearances(cam_transitions: Dict[str, pd.DataFrame]) -> List[str]:
        evts: List[str] = []
        for cam, tdf in sorted(cam_transitions.items()):
            for _, row in tdf.iterrows():
                rsn = str(row.get("reason", ""))
                tid = int(row.get("track_id", 0))
                frm = int(row.get("frame", 0))
                for part in rsn.split(";"):
                    part = part.strip()
                    if "disappeared" in part.lower() or "removal" in part.lower():
                        evts.append(f"{cam} / Track {tid} at frame {frm}: {part}")
        return evts

    @staticmethod
    def _generate_pdf(pdf_path: str, full_text: str, case_id: str, incident: str):
        try:
            from fpdf import FPDF

            class _PDF(FPDF):
                def header(self):
                    self.set_font("Courier", "B", 11)
                    self.set_fill_color(30, 30, 60)
                    self.set_text_color(255, 255, 255)
                    self.cell(0, 10, "  CASELINE -- INVESTIGATION CASE REPORT",
                              border=False, fill=True, new_x="LMARGIN", new_y="NEXT")
                    self.set_text_color(0, 0, 0)
                    self.ln(2)

                def footer(self):
                    self.set_y(-15)
                    self.set_font("Courier", "I", 8)
                    self.set_text_color(120, 120, 120)
                    self.cell(
                        0, 10,
                        f"CASELINE | Case {case_id} | Page {self.page_no()} | "
                        "Evidence first. AI second. Human decision always.",
                        align="C",
                    )

            pdf = _PDF()
            pdf.set_margins(15, 20, 15)
            pdf.set_auto_page_break(auto=True, margin=20)
            pdf.add_page()

            for line in full_text.split("\n"):
                stripped = line.strip()

                if stripped.startswith("SECTION"):
                    pdf.set_font("Courier", "B", 11)
                    pdf.set_fill_color(230, 235, 255)
                    pdf.multi_cell(0, 7, line, fill=True, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Courier", "", 9)
                elif stripped.startswith("=" * 10) or stripped.startswith("-" * 10):
                    pdf.set_draw_color(180, 180, 200)
                    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 175, pdf.get_y())
                    pdf.ln(1)
                elif "[SUSPICIOUS]" in stripped:
                    pdf.set_font("Courier", "B", 9)
                    pdf.set_text_color(180, 0, 0)
                    pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("Courier", "", 9)
                elif "[ATTENTION]" in stripped:
                    pdf.set_font("Courier", "B", 9)
                    pdf.set_text_color(180, 120, 0)
                    pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("Courier", "", 9)
                elif "[NORMAL]" in stripped:
                    pdf.set_font("Courier", "B", 9)
                    pdf.set_text_color(0, 100, 180)
                    pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_text_color(0, 0, 0)
                    pdf.set_font("Courier", "", 9)
                elif "IMPORTANT LEGAL NOTICE" in stripped:
                    pdf.set_font("Courier", "B", 9)
                    pdf.set_fill_color(255, 245, 230)
                    pdf.multi_cell(0, 6, line, fill=True, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font("Courier", "", 9)
                else:
                    pdf.set_font("Courier", "", 9)
                    pdf.multi_cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")

            pdf.output(pdf_path)
            print(f"  PDF saved: {pdf_path}")
        except Exception as e:
            print(f"  PDF generation failed: {e}")
