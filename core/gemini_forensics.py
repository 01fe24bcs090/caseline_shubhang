"""
CaseLine — Gemini Forensic Intelligence Module.

Provides:
  1. Courtroom-Grade Incident Narrative Generator (Automated Prosecutorial Briefing)
  2. Grounded Evidence Q&A Engine ("Chat with Your Evidence")
  3. Evidence Context Synthesizer (Merges tracking, objects, zones, and badge logs)
"""

import os
import glob
from typing import Dict, List, Optional
import pandas as pd

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


def extract_case_context(case_dir: str, case_info) -> str:
    """
    Compiles all forensic evidence from case files into a structured factual summary.
    """
    context_lines = []
    context_lines.append(f"CASE IDENTIFIER: {case_info.case_id}")
    context_lines.append(f"REPORTED INCIDENT: {case_info.incident_description}")
    context_lines.append(f"TOTAL CAMERAS: {len(case_info.videos)}")
    
    for vid in case_info.videos:
        context_lines.append(f"  - Camera {vid.camera_id}: {vid.original_filename} (SHA-256: {vid.sha256[:16]}...)")
    
    # 1. Timeline
    timeline_file = os.path.join(case_dir, "evidence", "incident_timeline.csv")
    if os.path.exists(timeline_file):
        df = pd.read_csv(timeline_file)
        if not df.empty:
            context_lines.append("\n--- CHRONOLOGICAL TIMELINE (PERSON TRACKING) ---")
            for _, r in df.sort_values(by="timestamp").iterrows():
                sub = r.get("subject_id", "Unknown")
                cam = r.get("camera_id", "Unknown")
                ts = r.get("timestamp", 0)
                status = r.get("activity_status", "NORMAL")
                notes = r.get("notes", "")
                context_lines.append(f"[{ts:06.1f}s] Subject {sub} on {cam} | Status: {status} | Details: {notes}")
    
    # 2. Activity Transitions & Disappearance Events (Crucial Theft Indicators)
    activity_files = sorted(glob.glob(os.path.join(case_dir, "evidence", "*_activity_transitions.csv")))
    if activity_files:
        context_lines.append("\n--- SUSPICIOUS OBJECT INTERACTIONS & DISAPPEARANCE EVENTS (POTENTIAL THEFT) ---")
        for af in activity_files:
            cam_name = os.path.basename(af).split("_")[0]
            try:
                adf = pd.read_csv(af)
                flagged = adf[adf["state"].isin(["ATTENTION", "SUSPICIOUS"])]
                for _, r in flagged.iterrows():
                    trk = r.get("track_id", "?")
                    st = r.get("state", "")
                    reason = r.get("reason", "")
                    frm = r.get("frame", "")
                    context_lines.append(f"[{cam_name}] Track {trk} ({st}): {reason} (Frame {frm})")
            except Exception:
                pass

    # 3. Detected Objects & Stationary Assets (Laptops, Bags, Handbags, Suitcases)
    obj_files = sorted(glob.glob(os.path.join(case_dir, "evidence", "*_objects.csv")))
    if obj_files:
        context_lines.append("\n--- DETECTED OBJECTS & PROPERTY IN CAMERA FEEDS ---")
        for of in obj_files:
            cam_name = os.path.basename(of).split("_")[0]
            try:
                odf = pd.read_csv(of)
                if not odf.empty and "predicted_class" in odf.columns:
                    grouped = odf.groupby("predicted_class")
                    for pclass, group in grouped:
                        cnt = len(group)
                        min_ts = group["timestamp"].min()
                        max_ts = group["timestamp"].max()
                        max_conf = group["confidence"].max() * 100
                        if cnt == 1 or abs(max_ts - min_ts) < 0.05:
                            ts_str = f"at {min_ts:05.2f}s"
                        else:
                            ts_str = f"from {min_ts:05.2f}s to {max_ts:05.2f}s"
                        
                        r0 = group.iloc[0]
                        coords = f"[{int(r0.get('x1', 0))}, {int(r0.get('y1', 0))}, {int(r0.get('x2', 0))}, {int(r0.get('y2', 0))}]"
                        extra_detail = f" [Coordinates: {coords}]"
                        
                        context_lines.append(
                            f"  - [{cam_name}] {pclass}: {cnt} detections ({ts_str}, peak conf: {max_conf:.1f}%){extra_detail}"
                        )
            except Exception:
                pass

    # 4. Official Investigation Report Summary
    report_file = os.path.join(case_dir, "output", "CASELINE_CASE_REPORT.txt")
    if os.path.exists(report_file):
        try:
            with open(report_file, "r", encoding="utf-8", errors="ignore") as rf:
                report_content = rf.read()
                if "SECTION 2 -- INVESTIGATION SUMMARY" in report_content:
                    sec2 = report_content.split("SECTION 2 -- INVESTIGATION SUMMARY")[1].split("="*40)[0].strip()
                    context_lines.append("\n--- FORENSIC ENGINE SUMMARY FINDINGS ---")
                    context_lines.append(sec2)
        except Exception:
            pass

    # 5. Zone Violations
    violation_files = glob.glob(os.path.join(case_dir, "evidence", "*zone_violations.csv"))
    if violation_files:
        context_lines.append("\n--- RESTRICTED ZONE VIOLATIONS ---")
        for vf in violation_files:
            vdf = pd.read_csv(vf)
            entries = vdf[vdf["event"] == "ZONE_ENTRY"] if "event" in vdf.columns else vdf
            for _, r in entries.iterrows():
                cam = r.get("camera_id", "CAM")
                trk = r.get("track_id", "?")
                zn = r.get("zone_name", "Restricted Area")
                ts = r.get("timestamp_sec", r.get("timestamp", 0))
                sev = r.get("severity", 30)
                context_lines.append(f"[{ts:06.1f}s] VIOLATION on {cam}: Track {trk} entered '{zn}' (Severity: {sev} pts)")
                
    # 6. Access Log Correlation
    access_file = os.path.join(case_dir, "evidence", "access_correlation.csv")
    if os.path.exists(access_file):
        adf = pd.read_csv(access_file)
        if not adf.empty:
            context_lines.append("\n--- ACCESS CONTROL & BADGE SWIPE CORRELATION ---")
            for _, r in adf.iterrows():
                ctype = r.get("correlation_type", "UNKNOWN")
                ts = r.get("timestamp", 0)
                cam = r.get("camera_id", "")
                badge = r.get("badge_id", "")
                name = r.get("holder_name", "")
                notes = r.get("notes", "")
                context_lines.append(f"[{ts:06.1f}s] Event: {ctype} | Cam: {cam} | Badge: {badge} ({name}) | Note: {notes}")

    # 7. Candidate Links
    candidates_file = os.path.join(case_dir, "evidence", "candidate_links.csv")
    if os.path.exists(candidates_file):
        cdf = pd.read_csv(candidates_file)
        if not cdf.empty:
            context_lines.append("\n--- CROSS-CAMERA RE-IDENTIFICATION CANDIDATES ---")
            for _, r in cdf.iterrows():
                ca, ta = r.get("camera_a", ""), r.get("track_a", "")
                cb, tb = r.get("camera_b", ""), r.get("track_b", "")
                conf = r.get("overall_confidence", 0)
                context_lines.append(f"Link: {ca}_Track{ta} <-> {cb}_Track{tb} (Match Confidence: {conf*100:.1f}%)")

    # 8. Photographic Evidence & Visual Exhibits Catalog
    try:
        from core.evidence_visualizer import ensure_case_snapshots
        snap_catalog = ensure_case_snapshots(case_dir, case_info)
        context_lines.append("\n--- VERIFIED PHOTOGRAPHIC EVIDENCE EXHIBITS CATALOG ---")
        context_lines.append("CaseLine has extracted photographic exhibits and crops for this case:")
        
        suspect_items = [f"{s['camera']} Track {s['track_id']}" for s in snap_catalog.get("suspects", [])]
        if suspect_items:
            context_lines.append(f"  - Suspect Crops Available: {', '.join(suspect_items)}")
            
        inc_items = [f"{i['camera']} Track {i['track_id']} ({i.get('reason', '')[:35]})" for i in snap_catalog.get("incidents", [])]
        if inc_items:
            context_lines.append(f"  - Critical Incident CCTV Frames: {'; '.join(inc_items)}")
            
        obj_items = [f"{o['class'].title()} on {o['camera']}" for o in snap_catalog.get("objects", [])]
        if obj_items:
            context_lines.append(f"  - Detected Asset Exhibits: {', '.join(sorted(set(obj_items)))}")
            
        context_lines.append("Note: CaseLine automatically pairs and renders these visual exhibits directly below your answer in the user interface.")
    except Exception:
        pass

    return "\n".join(context_lines)


CANDIDATE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-pro-latest"
]


def _call_gemini_with_fallback(client, contents, preferred_model: str = "gemini-3.1-flash-lite") -> str:
    """Attempts generation with preferred model and falls back if model is rate-limited, deprecated, busy, or not found."""
    models_to_try = [preferred_model] + [m for m in CANDIDATE_MODELS if m != preferred_model]
    last_err = None
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
            )
            return response.text
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            # Catch 429 quota exhaustion, 503 high demand, 404 deprecated models, rate limits
            if any(term in err_str for term in [
                "429", "resource_exhausted", "quota", "rate limit",
                "404", "503", "not_found", "no longer available", "unavailable"
            ]):
                continue
            raise e
    raise last_err


DEFAULT_GEMINI_KEY = ""


def _clean_api_key(raw_key: Optional[str]) -> str:
    """Sanitizes user input to remove accidental quotes, whitespace, or shell command prefixes."""
    key = (raw_key or os.environ.get("GEMINI_API_KEY", "")).strip()
    for prefix in ["$env:GEMINI_API_KEY=", "GEMINI_API_KEY=", "export GEMINI_API_KEY=", "set GEMINI_API_KEY="]:
        if prefix in key:
            key = key.split(prefix)[-1]
    key = key.strip().strip('"').strip("'").strip(";").strip()
    return key


def generate_forensic_briefing(
    case_dir: str,
    case_info,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.1-flash-lite"
) -> str:
    """
    Generates a formal courtroom-style Incident Narrative using Gemini.
    Falls back to a structured deterministic summary if offline or no key.
    """
    context = extract_case_context(case_dir, case_info)
    key = _clean_api_key(api_key)

    if not key or not GENAI_AVAILABLE:
        return _generate_offline_narrative(case_info, context)

    prompt = f"""You are a Lead Digital Forensics Investigator and Expert Witness presenting evidence for official legal proceedings.

Analyze the forensic data compiled below from multiple synchronized CCTV cameras, physical access logs, and AI tracking modules.

Produce a formal, comprehensive "EXECUTIVE FORENSIC INCIDENT NARRATIVE".
Structure the report into these clear sections:
1. CASE OVERVIEW & JURISDICTION (Brief incident context and camera evidence chain of custody)
2. CHRONOLOGICAL RECONSTRUCTION (Detailed step-by-step account of subject movements with exact timestamp citations [CAMxx @ ss.ss])
3. ANOMALIES & BREACH ANALYSIS (Highlight ghost swipes, unauthorized entries, and restricted zone breaches)
4. OBJECT & ASSET INTERACTIONS (Detailed accounting of detected property such as laptops, backpacks, handbags, suitcases, and any recorded disappearance events indicating theft)
5. CORROBORATING EVIDENCE SUMMARY (Cross-camera links, object interactions, verified identities)
6. EVIDENTIARY CONCLUSION & RECOMMENDATIONS (Objective factual summary; remind that guilt determination resides with the court)

Maintain an objective, factual, legally rigorous tone. Do NOT speculate beyond the recorded data.

--- FORENSIC CASE EVIDENCE DATA ---
{context}
"""

    try:
        client = genai.Client(api_key=key)
        return _call_gemini_with_fallback(client, prompt, preferred_model=model_name)
    except Exception as e:
        return f"{_generate_offline_narrative(case_info, context)}\n\n*(Note: Gemini live generation encountered: {e}. Fallback report displayed above.)*"


def ask_evidence_question(
    question: str,
    case_dir: str,
    case_info,
    chat_history: Optional[List[Dict[str, str]]] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.1-flash-lite",
    include_images: bool = True
):
    """
    Answers natural language queries strictly grounded in the case evidence.
    Returns (answer_text, relevant_images) if include_images=True, else answer_text.
    """
    context = extract_case_context(case_dir, case_info)
    key = _clean_api_key(api_key)

    if not key or not GENAI_AVAILABLE:
        msg = (
            "⚠️ Gemini API Key not configured. Please enter your Gemini API key in the sidebar "
            "or set the `GEMINI_API_KEY` environment variable to enable live Evidence Q&A."
        )
        return (msg, []) if include_images else msg

    system_prompt = (
        "You are CaseLine AI, a senior digital forensics investigative assistant and expert witness. "
        "Your task is to answer questions from judges, prosecutors, and investigators strictly based on the provided case data. "
        "IMPORTANT: You are integrated with CaseLine's Forensic Evidence Visualizer. Photographic exhibits, suspect profile crops, "
        "and CCTV incident frames ARE automatically attached and rendered directly below your response in the user interface. "
        "When asked to show photos, pictures, images, or suspects, introduce and summarize the primary suspects identified in this case "
        "(e.g. 'Below are the verified photographic exhibits and CCTV captures of the primary suspects identified in the investigation:'). "
        "NEVER claim that you do not have access to photos or visual files, because CaseLine automatically displays them. "
        "Always cite camera IDs, track IDs, and timestamps (e.g. [CAMxx @ Frame xxx] or [CAMxx @ ss.ss]) directly from the provided evidence. "
        "Reference the specific subjects and objects flagged in Section 2 (SUSPICIOUS OBJECT INTERACTIONS), Section 3 (DETECTED OBJECTS), and Section 8 (PHOTOGRAPHIC EVIDENCE EXHIBITS CATALOG). "
        "Do not invent facts not present in the provided evidence record."
    )

    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:  # Keep recent context
            role = "Investigator" if msg["role"] == "user" else "CaseLine"
            history_text += f"{role}: {msg['content']}\n"

    user_query = f"""--- VERIFIED EVIDENCE RECORD ---
{context}

--- CONVERSATION HISTORY ---
{history_text}

Investigator Question: {question}
Answer factually with exact timestamps and camera references:"""

    from core.evidence_visualizer import get_relevant_evidence_images

    answer_text = ""
    try:
        client = genai.Client(api_key=key)
        answer_text = _call_gemini_with_fallback(client, [system_prompt, user_query], preferred_model=model_name)
    except Exception as e:
        # Fallback to deterministic grounded evidence synthesis so user and judges never see an unhandled error
        answer_text = _synthesize_grounded_fallback_answer(question, context)

    images = []
    if include_images:
        images = get_relevant_evidence_images(question, answer_text, case_dir, case_info=case_info)
        return answer_text, images

    return answer_text


def _synthesize_grounded_fallback_answer(question: str, context: str) -> str:
    """
    Deterministic factual engine that dynamically synthesizes answers directly from the active case's 
    verified evidence record if all external Gemini model quotas are temporarily exhausted or offline.
    100% case-agnostic: dynamically parses whatever suspects, objects, transitions, and timeline exist in this case.
    """
    q = question.lower()
    prefix = "> ⚖️ *Grounded Forensic Record Synthesis (Offline / Quota-Resilient Mode):*\n\n"
    lines = []

    # Parse sections from context dynamically
    sections = {}
    current_sec = "HEADER"
    sections[current_sec] = []
    for line in context.splitlines():
        if line.startswith("--- ") and line.endswith(" ---"):
            current_sec = line.strip("- ").strip()
            sections[current_sec] = []
        else:
            sections[current_sec].append(line)

    suspect_sec = [l for l in sections.get("SUSPICIOUS OBJECT INTERACTIONS & DISAPPEARANCE EVENTS (POTENTIAL THEFT)", []) if l.strip()]
    objects_sec = [l for l in sections.get("DETECTED OBJECTS & PROPERTY IN CAMERA FEEDS", []) if l.strip()]
    timeline_sec = [l for l in sections.get("CHRONOLOGICAL TIMELINE (PERSON TRACKING)", []) if l.strip()]
    zones_sec = [l for l in sections.get("RESTRICTED ZONE VIOLATIONS", []) if l.strip()]
    catalog_sec = [l for l in sections.get("VERIFIED PHOTOGRAPHIC EVIDENCE EXHIBITS CATALOG", []) if l.strip()]

    if any(k in q for k in ["suspect", "suspected", "who", "person", "thief", "photo", "picture", "image", "look like"]):
        lines.append("### 1. Identified Suspects & Persons of Interest")
        if suspect_sec:
            lines.append("The following subjects were flagged for suspicious activity or object interactions in this case:")
            for s_line in suspect_sec:
                lines.append(f"* **{s_line}**")
        elif timeline_sec:
            lines.append("Tracked subjects in camera coverage for this case:")
            for t_line in timeline_sec[:6]:
                lines.append(f"* {t_line}")
        else:
            lines.append("No suspicious human subjects flagged in current camera logs.")

        if catalog_sec:
            lines.append("\n### 2. Available Photographic Exhibits")
            for c_line in catalog_sec:
                if c_line.startswith("  -"):
                    lines.append(f"*{c_line}*")
        lines.append("\n*The verified photographic profile crops and CCTV incident captures for these subjects are displayed below.*")
        return prefix + "\n".join(lines)

    elif any(k in q for k in ["stolen", "steal", "theft", "missing", "take", "taken", "object", "item", "property", "laptop", "bag", "handbag", "backpack", "suitcase"]):
        lines.append("### 1. Suspicious Interactions & Disappearance Events")
        if suspect_sec:
            for s_line in suspect_sec:
                lines.append(f"* {s_line}")
        else:
            lines.append("No direct disappearance events recorded in active activity transition logs.")

        lines.append("\n### 2. Scene Inventory & Detected Property")
        if objects_sec:
            for o_line in objects_sec:
                lines.append(f"* {o_line.strip()}")
        else:
            lines.append("No stationary assets or objects indexed in camera coverage.")

        return prefix + "\n".join(lines)

    elif any(k in q for k in ["zone", "violation", "breach", "restricted"]):
        lines.append("### 1. Restricted Zone Violations")
        if zones_sec:
            for z_line in zones_sec:
                lines.append(f"* {z_line}")
        else:
            lines.append("* No unauthorized restricted zone breaches recorded in active monitor zones for this case.")
        return prefix + "\n".join(lines)

    elif any(k in q for k in ["enter", "entry", "timeline", "when", "first", "movement"]):
        lines.append("### 1. Chronological Subject Movements")
        if timeline_sec:
            for t_line in timeline_sec:
                lines.append(f"* {t_line}")
        else:
            lines.append("No chronological timeline entries recorded yet. (Approve candidate links in Tab 4 to generate timeline).")
        return prefix + "\n".join(lines)

    else:
        # General question fallback: synthesize summary from header and available sections
        lines.append("### Case Evidence Summary")
        header_lines = [l for l in sections.get("HEADER", []) if l.strip()]
        for h in header_lines:
            lines.append(f"* {h}")
        if suspect_sec:
            lines.append("\n**Flagged Interactions:**")
            for s in suspect_sec[:3]:
                lines.append(f"* {s}")
        if objects_sec:
            lines.append("\n**Detected Property:**")
            for o in objects_sec[:3]:
                lines.append(f"* {o.strip()}")
        return prefix + "\n".join(lines)


def _generate_offline_narrative(case_info, context: str) -> str:
    """Deterministic fallback narrative when API key is unavailable."""
    return f"""# 📑 Formal Forensic Incident Briefing (Automated Synthesis)

**Case Reference:** {case_info.case_id}  
**Incident Type:** {case_info.incident_description}  
**Evidence Custody:** All ingested camera files verified with SHA-256 cryptographic digests.

---

### 1. Executive Summary
This dossier synthesizes automated multi-camera tracking, physical access control logs, and computer-vision behavioral scoring for Case {case_info.case_id}. The system tracked human subjects across {len(case_info.videos)} camera views and correlated cross-camera transitions to reconstruct a tamper-evident timeline.

### 2. Evidence Context Summary
```
{context}
```

### 3. Evidentiary Standards & Legal Notice
- **Non-Biometric Recognition:** Person correlation is based solely on visual feature representations (clothing color, stature, temporal trajectory). Facial recognition is strictly excluded to prevent bias.
- **Human Verification Required:** All candidate cross-camera associations require formal human investigator approval prior to admissibility.
- **Custody Assurance:** Cryptographic hashes ensure original CCTV media remains untampered throughout processing.
"""
