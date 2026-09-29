import streamlit as st
import os
import pandas as pd
from core.case import CaseManager
from core.pipeline import CaselinePipeline
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from core.gemini_forensics import generate_forensic_briefing, ask_evidence_question

# Setup Streamlit page configuration FIRST
st.set_page_config(
    page_title="CASELINE — Security Evidence Engine",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded"
)

# Initialize backend
manager = CaseManager()
pipeline = CaselinePipeline()

# ── Clean Enterprise Styling ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", Helvetica, Arial, sans-serif !important;
}

code, kbd, samp, pre {
    font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
}

.stApp {
    background-color: #0d1117 !important;
    color: #c9d1d9 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #f0f6fc !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}

/* Clean Top App Header */
header[data-testid="stHeader"] {
    background-color: rgba(13, 17, 23, 0.8) !important;
    border-bottom: 1px solid #21262d !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #f0f6fc !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Clean, Minimalist Tabs */
[data-testid="stTabs"] {
    margin-top: 8px;
    margin-bottom: 20px;
}
div[data-baseweb="tab-list"] {
    background-color: transparent !important;
    border-bottom: 1px solid #30363d !important;
    gap: 4px !important;
    padding: 0 !important;
}
button[data-baseweb="tab"] {
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    padding: 8px 14px !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    transition: color 0.15s ease, border-color 0.15s ease !important;
}
button[data-baseweb="tab"]:hover {
    color: #f0f6fc !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f0f6fc !important;
    border-bottom: 2px solid #388bfd !important;
    background: transparent !important;
    font-weight: 600 !important;
}
div[data-baseweb="tab-highlight"] {
    display: none !important;
}

/* Professional Buttons */
.stButton > button {
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
    border: 1px solid #30363d !important;
    background-color: #21262d !important;
    color: #c9d1d9 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #30363d !important;
    border-color: #8b949e !important;
    color: #f0f6fc !important;
}
.stButton > button[kind="primary"] {
    background-color: #238636 !important;
    border-color: rgba(240, 246, 252, 0.1) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #2ea043 !important;
    border-color: rgba(240, 246, 252, 0.1) !important;
}

/* Clean Metrics */
[data-testid="stMetric"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricValue"] {
    color: #f0f6fc !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

/* Clean Form Controls */
.stTextInput input, .stTextArea textarea {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(56, 139, 253, 0.2) !important;
}

/* Clean Tables / DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    background-color: #161b22 !important;
}

/* Professional File Uploader */
[data-testid="stFileUploader"] {
    background-color: #161b22 !important;
    border: 1px dashed #30363d !important;
    border-radius: 6px !important;
    padding: 14px !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #8b949e !important;
}

/* Clean Badges & Cards */
.clean-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 16px;
}
.clean-badge {
    display: inline-block;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 500;
    border-radius: 4px;
    background: #21262d;
    border: 1px solid #30363d;
    color: #8b949e;
}
.clean-badge-green {
    background: rgba(35, 134, 54, 0.15);
    border-color: #238636;
    color: #3fb950;
}
.clean-evidence-row {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}
</style>
""", unsafe_allow_html=True)

# Minimal Clean Header
st.markdown("""
<div style="padding: 4px 0 16px 0; border-bottom: 1px solid #21262d; margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: baseline; gap: 12px;">
            <span style="font-size: 1.35rem; font-weight: 700; color: #f0f6fc; letter-spacing: -0.01em;">CaseLine</span>
            <span style="font-size: 0.85rem; color: #8b949e;">Security Evidence Engine</span>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
            <span class="clean-badge clean-badge-green">Engine Ready</span>
            <span class="clean-badge">Chain of Custody Preserved</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar for case selection & AI configuration
st.sidebar.header("Case Management")
cases_dir = "data/cases"
os.makedirs(cases_dir, exist_ok=True)
existing_cases = os.listdir(cases_dir)

selected_case = st.sidebar.selectbox("Select Case", ["-- Create New Case --"] + existing_cases)

DEFAULT_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

st.sidebar.markdown("---")
st.sidebar.header("AI Intelligence")
saved_key = os.environ.get("GEMINI_API_KEY", "")
raw_gemini_key = st.sidebar.text_input(
    "Gemini API Key", 
    value=saved_key, 
    type="password",
    help="Google Gemini API key for AI Case Narratives and Evidence Q&A Chat."
)
gemini_api_key = raw_gemini_key.strip()
for prefix in ["$env:GEMINI_API_KEY=", "GEMINI_API_KEY=", "export GEMINI_API_KEY=", "set GEMINI_API_KEY="]:
    if prefix in gemini_api_key:
        gemini_api_key = gemini_api_key.split(prefix)[-1]
gemini_api_key = gemini_api_key.strip().strip('"').strip("'").strip(";").strip()

if not gemini_api_key:
    gemini_api_key = DEFAULT_GEMINI_KEY

os.environ["GEMINI_API_KEY"] = gemini_api_key

if selected_case == "-- Create New Case --":
    st.header("Create New Case")
    new_case_id = st.text_input("Case ID (e.g. CASE-002)")
    new_incident = st.text_area("Incident Description")
    if st.button("Create Case", type="primary"):
        if new_case_id and new_incident:
            try:
                manager.create_case(new_case_id, new_incident)
                st.success(f"Case {new_case_id} created! Please select it from the sidebar.")
                st.rerun()
            except ValueError as e:
                st.error(str(e))
        else:
            st.warning("Please provide both Case ID and Incident Description.")
else:
    case_id = selected_case
    case_dir = os.path.join(cases_dir, case_id)
    case_info = manager.load_case_info(case_id)
    
    st.markdown(f"""
    <div class="clean-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <span class="clean-badge" style="color: #388bfd; border-color: rgba(56, 139, 253, 0.4); margin-bottom: 4px;">CASE RECORD</span>
                <h3 style="margin: 2px 0 0 0; font-size: 1.15rem; color: #f0f6fc;">{case_info.case_id}</h3>
                <p style="margin: 4px 0 0 0; color: #8b949e; font-size: 0.875rem;">{case_info.incident_description}</p>
            </div>
            <div style="display: flex; gap: 8px; align-items: center;">
                <span class="clean-badge">{len(case_info.videos)} Camera Feeds</span>
                <span class="clean-badge clean-badge-green">SHA-256 Verified</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "Evidence Ingestion", "Analysis Pipeline", "Annotated Footage", 
        "Candidate Review", "Incident Timeline", "Case Dossier", 
        "AI Evidence Chat", "Multi-View Sync", "Visual Search", 
        "Restricted Zones", "Access Correlation"
    ])
    
    with tab1:
        st.subheader("Upload Videos")
        uploaded_vids = st.file_uploader("Upload CCTV Videos", accept_multiple_files=True, type=["mp4", "webm", "avi"])
        if st.button("Ingest Videos"):
            for vid in uploaded_vids:
                temp_path = os.path.join(case_dir, vid.name)
                with open(temp_path, "wb") as f:
                    f.write(vid.getbuffer())
                
                manager.ingest_video(case_id, temp_path)
                os.remove(temp_path)
            st.success("Videos ingested successfully!")
            st.rerun()
            
        st.subheader("Current Video Evidence")
        if not case_info.videos:
            st.info("No videos uploaded for this case yet.")
        for vid in case_info.videos:
            st.markdown(f"""
            <div class="clean-evidence-row">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="clean-badge" style="color: #388bfd;">{vid.camera_id}</span>
                    <span style="font-weight: 500; color: #f0f6fc; font-size: 0.875rem;">{vid.original_filename}</span>
                </div>
                <div style="font-family: monospace; font-size: 0.78rem; color: #6e7681;">
                    SHA-256: <span style="color: #8b949e;">{vid.sha256[:16]}...{vid.sha256[-8:]}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.subheader("Upload Access Logs")
        uploaded_log = st.file_uploader("Upload Access Log (CSV)", type=["csv"])
        if st.button("Ingest Access Log"):
            if uploaded_log:
                temp_path = os.path.join(case_dir, uploaded_log.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_log.getbuffer())
                manager.ingest_access_log(case_id, temp_path)
                os.remove(temp_path)
                st.success("Access log ingested!")
                st.rerun()
                
    with tab2:
        st.subheader("Automated Analysis Pipeline")
        st.write("Executes person detection, object stability verification, visual feature extraction, and cross-camera candidate matching across all case footage.")
        
        if st.button("Run Full Analysis Pipeline", type="primary"):
            with st.spinner("Analyzing case footage..."):
                pipeline.analyze_videos(case_id, case_dir)
            st.success("Analysis complete. Review candidate links and annotated footage in subsequent tabs.")
            
    with tab3:
        st.subheader("Annotated Video Evidence")
        output_dir = os.path.join(case_dir, "output")
        if os.path.exists(output_dir):
            processed_vids = [f for f in os.listdir(output_dir) if f.endswith("_tracking.webm")]
            if processed_vids:
                selected_vid = st.selectbox("Select Camera View to Watch", processed_vids)
                if selected_vid:
                    vid_path = os.path.join(output_dir, selected_vid)
                    with open(vid_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.info("No annotated videos found. Run Analysis first.")
                
    with tab4:
        st.subheader("Human Review - Cross-Camera Links")
        evidence_dir = os.path.join(case_dir, "evidence")
        candidates_file = os.path.join(evidence_dir, "candidate_links.csv")
        
        if os.path.exists(candidates_file):
            candidates_df = pd.read_csv(candidates_file)
            
            if not candidates_df.empty:
                st.write(f"Found {len(candidates_df)} candidate links.")
                
                approved_indices = []
                for idx, row in candidates_df.iterrows():
                    cam_a = row['camera_a']
                    trk_a = row['track_a']
                    cam_b = row['camera_b']
                    trk_b = row['track_b']
                    conf = row['overall_confidence']
                    
                    st.markdown("---")
                    st.write(f"**Candidate Link {idx + 1}**")
                    st.write(f"Confidence: {conf*100:.1f}%")
                    
                    col1, col2 = st.columns(2)
                    crop_a = os.path.join(case_dir, "output", "crops", f"{cam_a}_{trk_a}.jpg")
                    crop_b = os.path.join(case_dir, "output", "crops", f"{cam_b}_{trk_b}.jpg")
                    
                    if os.path.exists(crop_a):
                        col1.image(crop_a, caption=f"{cam_a} Track {trk_a}")
                    else:
                        col1.write("Crop missing")
                        
                    if os.path.exists(crop_b):
                        col2.image(crop_b, caption=f"{cam_b} Track {trk_b}")
                    else:
                        col2.write("Crop missing")
                        
                    is_approved = st.checkbox(f"Approve Link {idx+1}", key=f"approve_{idx}")
                    if is_approved:
                        approved_indices.append(idx)
                        
                if st.button("Confirm Approvals & Generate Timeline"):
                    approved_links = candidates_df.iloc[approved_indices].to_dict('records')
                    pipeline.process_approvals_and_finish(case_id, case_dir, approved_links)
                    st.success("Approvals processed! Check the Timeline and Report tabs.")
            else:
                st.write("No candidate links found.")
        else:
            st.info("Run Analysis first to generate candidate links.")
            
    with tab5:
        st.subheader("Incident Timeline")
        timeline_file = os.path.join(case_dir, "evidence", "incident_timeline.csv")
        if os.path.exists(timeline_file):
            timeline_df = pd.read_csv(timeline_file)
            
            st.markdown("### Suspect Path Visualization")
            G = nx.DiGraph()
            
            timeline_df = timeline_df.sort_values(by="timestamp")
            
            for subject_id, group in timeline_df.groupby("subject_id"):
                prev_cam = None
                for _, row in group.iterrows():
                    cam = str(row["camera_id"])
                    if not G.has_node(cam):
                        G.add_node(cam, label=f"Camera {cam}", color="#2ecc71", size=40)
                    
                    if prev_cam is not None and prev_cam != cam:
                        if G.has_edge(prev_cam, cam):
                            G[prev_cam][cam]['weight'] += 1
                        else:
                            G.add_edge(prev_cam, cam, weight=1, color="#e74c3c")
                            
                    prev_cam = cam
                    
            if len(G.nodes) > 0:
                net = Network(height="400px", width="100%", bgcolor="#0e1117", font_color="white", directed=True)
                # Adjust physics settings for a stable, professional look
                net.force_atlas_2based()
                net.from_nx(G)
                
                graph_path = os.path.join(case_dir, "evidence", "path_graph.html")
                net.save_graph(graph_path)
                
                with open(graph_path, "r", encoding="utf-8") as f:
                    html_data = f.read()
                components.html(html_data, height=410)
            else:
                st.info("Not enough cross-camera transitions to draw a graph.")
                
            st.markdown("### Chronological Log")
            st.dataframe(timeline_df)
        else:
            st.info("Approve links to generate timeline.")
            
    with tab6:
        st.subheader("Case Dossier & Evidence Reports")
        
        pdf_file = os.path.join(case_dir, "output", "CASELINE_CASE_REPORT.pdf")
        if os.path.exists(pdf_file):
            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Download Official Courtroom PDF Report", 
                data=pdf_bytes, 
                file_name=f"{case_id}_Evidence_Report.pdf", 
                mime="application/pdf",
                type="primary"
            )
            st.markdown("---")
            
        report_file = os.path.join(case_dir, "output", "CASELINE_CASE_REPORT.txt")
        if os.path.exists(report_file):
            with open(report_file, "r") as f:
                report_content = f.read()
            st.text_area("Audit Log & Summary", report_content, height=220)
            st.download_button("Download Raw Summary", report_content, file_name=f"{case_id}_Report.txt")
        else:
            st.info("Run Pipeline Analysis to generate case records.")

        st.markdown("---")
        st.markdown("### AI Forensic Incident Narrative (Powered by Gemini)")
        st.write("Synthesizes multi-camera CCTV tracks, verified object interactions, and access anomalies into an executive legal brief.")
        
        if st.button("Generate Courtroom Briefing with Gemini", type="primary"):
            with st.spinner("Synthesizing case evidence with Gemini..."):
                narrative = generate_forensic_briefing(case_dir, case_info, api_key=gemini_api_key)
                st.session_state[f"narrative_{case_id}"] = narrative
                
        if f"narrative_{case_id}" in st.session_state:
            narrative_text = st.session_state[f"narrative_{case_id}"]
            st.markdown(narrative_text)
            st.download_button(
                "Download AI Forensic Narrative (Markdown)", 
                narrative_text, 
                file_name=f"{case_id}_AI_Forensic_Briefing.md",
                mime="text/markdown"
            )

        # Photographic Evidence Exhibits in Case Dossier
        from core.evidence_visualizer import ensure_case_snapshots
        snapshots_data = ensure_case_snapshots(case_dir, case_info)
        with st.expander("📸 Verified Photographic Evidence Exhibits & Asset Crops", expanded=True):
            if snapshots_data.get("incidents"):
                st.markdown("##### 🚨 Critical Incident Disappearance Snapshots")
                inc_cols = st.columns(min(len(snapshots_data["incidents"]), 3))
                for idx, inc in enumerate(snapshots_data["incidents"][:6]):
                    with inc_cols[idx % 3]:
                        st.image(inc["path"], caption=inc["caption"], use_container_width=True)
            if snapshots_data.get("objects"):
                st.markdown("##### 📦 Detected Assets & Scene Inventory")
                obj_cols = st.columns(min(len(snapshots_data["objects"]), 3))
                for idx, obj in enumerate(snapshots_data["objects"][:6]):
                    with obj_cols[idx % 3]:
                        st.image(obj["snapshot_path"], caption=obj["caption"], use_container_width=True)
            
    with tab7:
        st.subheader("AI Evidence Chat (Grounded Investigator Q&A)")
        st.write("Query the verified case timeline, object detections, zone breaches, and access logs using Gemini. Answers include direct photographic evidence exhibits.")
        
        # Preset query buttons dynamically tailored to the active case
        detected_obj_classes = [o["class"].title() for o in snapshots_data.get("objects", [])]
        featured_obj = detected_obj_classes[0] if detected_obj_classes else "Property"
        
        st.markdown("**Suggested Forensic Inquiries:**")
        quick_cols = st.columns([1, 1, 1, 1, 1])
        q_suspect = quick_cols[0].button("Show Suspect Photo", key="q_suspect")
        q_stolen = quick_cols[1].button("What was stolen?", key="q_stolen")
        q_item = quick_cols[2].button(f"Show {featured_obj} exhibit", key="q_item")
        q1 = quick_cols[3].button("Where did suspect enter?", key="q1")
        if quick_cols[4].button("Clear Chat", key="clear_chat"):
            st.session_state[f"chat_history_{case_id}"] = []
            st.rerun()
        
        selected_quick = None
        if q_suspect: selected_quick = "Show the suspected photo and primary suspect profiles."
        if q_stolen: selected_quick = "What was stolen or missing? Provide details on the disappearance events and assets."
        if q_item: selected_quick = f"Show the detected {featured_obj.lower()} on scene and its camera coordinates."
        if q1: selected_quick = "Where and when did the suspect enter the facility?"
        
        if f"chat_history_{case_id}" not in st.session_state:
            st.session_state[f"chat_history_{case_id}"] = []
        else:
            # Automatically purge stale error/disclaimer messages from previous turns
            st.session_state[f"chat_history_{case_id}"] = [
                m for m in st.session_state[f"chat_history_{case_id}"]
                if "gemini-2.5-flash" not in m.get("content", "") 
                and "404 NOT_FOUND" not in m.get("content", "")
                and "do not have access to visual image files" not in m.get("content", "")
            ]
            
        chat_hist = st.session_state[f"chat_history_{case_id}"]
        
        for msg in chat_hist:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("images"):
                    st.markdown("##### 📸 Verified Photographic Evidence Exhibits:")
                    img_cols = st.columns(min(len(msg["images"]), 3))
                    for idx, img in enumerate(msg["images"]):
                        with img_cols[idx % 3]:
                            st.image(img["path"], caption=img.get("caption", ""), use_container_width=True)
                
        user_prompt = st.chat_input("Ask a question about this case (e.g. 'What did the subject do on CAM02?')...")
        query_to_run = selected_quick or user_prompt
        
        if query_to_run:
            chat_hist.append({"role": "user", "content": query_to_run})
            with st.chat_message("user"):
                st.markdown(query_to_run)
                
            with st.chat_message("assistant"):
                with st.spinner("Analyzing case evidence and retrieving photographic exhibits..."):
                    res = ask_evidence_question(
                        query_to_run,
                        case_dir,
                        case_info,
                        chat_history=chat_hist[:-1],
                        api_key=gemini_api_key,
                        include_images=True
                    )
                    if isinstance(res, tuple):
                        answer, relevant_images = res
                    else:
                        answer, relevant_images = res, []
                st.markdown(answer)
                if relevant_images:
                    st.markdown("##### 📸 Verified Photographic Evidence Exhibits:")
                    img_cols = st.columns(min(len(relevant_images), 3))
                    for idx, img in enumerate(relevant_images):
                        with img_cols[idx % 3]:
                            st.image(img["path"], caption=img.get("caption", ""), use_container_width=True)
            chat_hist.append({"role": "assistant", "content": answer, "images": relevant_images})

    with tab8:
        st.subheader("Multi-Camera Synchronized Playback")
        st.write("Generate a perfectly synchronized side-by-side composite view of multiple cameras for easy review.")
        output_dir = os.path.join(case_dir, "output")
        if os.path.exists(output_dir):
            processed_vids = [f for f in os.listdir(output_dir) if f.endswith("_tracking.webm")]
            if processed_vids:
                selected_vids = st.multiselect("Select Videos to Synchronize", processed_vids, default=processed_vids[:2] if len(processed_vids) >= 2 else None)
                
                sync_out_path = os.path.join(output_dir, "sync_playback.webm")
                
                if len(selected_vids) >= 2:
                    if st.button("Generate Sync Video"):
                        with st.spinner("Stitching videos together for perfectly synchronized playback..."):
                            from vision.sync_video import create_sync_video
                            video_paths = [os.path.join(output_dir, v) for v in selected_vids]
                            create_sync_video(video_paths, sync_out_path)
                            
                        st.success("Synchronized Video Generated!")
                        
                elif len(selected_vids) == 1:
                    st.info("Please select at least 2 videos to synchronize.")
                    
                if os.path.exists(sync_out_path):
                    with open(sync_out_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
            else:
                st.info("No annotated videos found. Run Analysis first.")
        else:
            st.info("Run Analysis first.")
            
    with tab9:
        st.subheader("Semantic Evidence Search")
        st.write("Use natural language to search through all captured suspect profiles.")
        search_query = st.text_input("Enter description (e.g. 'man in a red shirt', 'person with a backpack')")
        
        if st.button("Search Evidence", type="primary"):
            if search_query:
                with st.spinner("Running AI Semantic Search across all crops (this may take a moment to load the model)..."):
                    from vision.search import SemanticSearch
                    
                    if "semantic_searcher" not in st.session_state:
                        st.session_state.semantic_searcher = SemanticSearch()
                        
                    results = st.session_state.semantic_searcher.search_crops(search_query, case_dir, top_k=6)
                    
                if results:
                    st.success(f"Found {len(results)} matches for '{search_query}'")
                    cols = st.columns(3)
                    for i, res in enumerate(results):
                        col = cols[i % 3]
                        col.image(res["path"], use_container_width=True)
                        col.markdown(f"**Camera:** {res['camera']} | **Track:** {res['track']}")
                        col.markdown(f"*Match Score: {res['score']:.3f}*")
                else:
                    st.warning("No matches found or no crops available. Run Analysis first.")
            else:
                st.info("Please enter a search query.")

    with tab10:
        st.subheader("Restricted Zone Management")
        st.write("Define restricted, sensitive, entry, or exit zones on camera views. Persons entering these zones will be automatically flagged during analysis.")

        from vision.zone_detection import ZoneManager, ZoneViolationDetector, extract_reference_frame, render_zones_on_frame
        import cv2
        import numpy as np

        zone_mgr = ZoneManager(case_dir)

        # Camera selection
        cam_ids = [v.camera_id for v in case_info.videos]
        if cam_ids:
            selected_cam = st.selectbox("Select Camera for Zone Setup", cam_ids, key="zone_cam")

            # Show existing zones
            existing_zones = zone_mgr.load_zones(selected_cam)
            if existing_zones:
                st.markdown("### Existing Zones")
                for z in existing_zones:
                    col_z1, col_z2 = st.columns([3, 1])
                    col_z1.write(f"**{z.name}** — `{z.zone_type}` (Severity: {z.severity} pts, {len(z.polygon)} vertices)")
                    if col_z2.button(f"Delete", key=f"del_{z.zone_id}"):
                        zone_mgr.remove_zone(selected_cam, z.zone_id)
                        st.rerun()

                # Show reference frame with zones overlaid
                video_path = None
                videos_dir_path = os.path.join(case_dir, "videos")
                for vf in os.listdir(videos_dir_path):
                    if vf.startswith(f"{selected_cam}_"):
                        video_path = os.path.join(videos_dir_path, vf)
                        break

                if video_path:
                    ref_frame = extract_reference_frame(video_path, frame_number=30)
                    if ref_frame is not None:
                        rendered = render_zones_on_frame(ref_frame.copy(), existing_zones)
                        rendered_rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                        st.image(rendered_rgb, caption="Camera view with zone overlays", use_container_width=True)

            # Add new zone
            st.markdown("### Define New Zone")
            st.write("Enter the zone polygon as comma-separated `x,y` pairs (e.g. `100,200 400,200 400,500 100,500`).")

            zone_name = st.text_input("Zone Name", placeholder="e.g. Server Room Entry", key="zone_name")
            zone_type = st.selectbox("Zone Type", ["RESTRICTED", "SENSITIVE", "EXIT", "ENTRY"], key="zone_type")
            severity = st.slider("Severity Points", min_value=10, max_value=50, value=30, step=5, key="zone_sev")
            polygon_str = st.text_input("Polygon Vertices (x,y pairs separated by spaces)",
                                        placeholder="100,200 400,200 400,500 100,500", key="zone_poly")

            if st.button("Add Zone"):
                if zone_name and polygon_str:
                    try:
                        pairs = polygon_str.strip().split()
                        polygon = []
                        for pair in pairs:
                            x, y = pair.split(",")
                            polygon.append((int(x.strip()), int(y.strip())))
                        if len(polygon) >= 3:
                            zone_mgr.add_zone(selected_cam, zone_name, zone_type, polygon, severity)
                            st.success(f"Zone '{zone_name}' added to {selected_cam}!")
                            st.rerun()
                        else:
                            st.error("A zone needs at least 3 vertices.")
                    except Exception as e:
                        st.error(f"Invalid polygon format: {e}")
                else:
                    st.warning("Please fill in zone name and polygon vertices.")

            # Show violations if they exist
            violations_file = os.path.join(case_dir, "evidence", f"{selected_cam}_zone_violations.csv")
            if os.path.exists(violations_file):
                st.markdown("### Zone Violations")
                v_df = pd.read_csv(violations_file)
                entries_df = v_df[v_df["event"] == "ZONE_ENTRY"]
                if not entries_df.empty:
                    st.error(f"⚠ {len(entries_df)} zone entry events detected!")
                    st.dataframe(v_df)
                else:
                    st.success("No zone violations detected.")
        else:
            st.info("Upload videos first to set up zones.")

    with tab11:
        st.subheader("Access Log Correlation")
        st.write("Cross-reference badge swipes and door events with video-detected persons to identify ghost swipes and unauthorized entries.")

        correlation_file = os.path.join(case_dir, "evidence", "access_correlation.csv")

        if os.path.exists(correlation_file):
            corr_df = pd.read_csv(correlation_file)

            # Summary metrics
            matched = corr_df[corr_df["correlation_type"] == "MATCH"]
            ghosts = corr_df[corr_df["correlation_type"] == "GHOST_SWIPE"]
            unmatched = corr_df[corr_df["correlation_type"] == "UNMATCHED_PERSON"]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Events", len(corr_df))
            col2.metric("Matched", len(matched))
            col3.metric("Ghost Swipes", len(ghosts))
            col4.metric("Unauthorized", len(unmatched))

            if not ghosts.empty:
                st.markdown("### ⚠ Ghost Swipes")
                st.write("Badge swiped but **no person detected** on camera — possible tailgating or spoofed badge.")
                st.dataframe(ghosts[["timestamp", "camera_id", "badge_id", "holder_name", "access_point", "notes"]])

            if not unmatched.empty:
                st.markdown("### 🚨 Unauthorized Persons")
                st.write("Person detected on camera but **no badge swipe** recorded — possible unauthorized access.")
                st.dataframe(unmatched[["timestamp", "camera_id", "matched_track_id", "notes"]])

            if not matched.empty:
                st.markdown("### ✅ Matched Events")
                st.dataframe(matched[["timestamp", "camera_id", "badge_id", "holder_name",
                                       "matched_track_id", "match_confidence", "time_delta_sec"]])

            st.markdown("### Full Correlation Data")
            st.dataframe(corr_df)
        else:
            st.info("Upload an access log in the 'Upload Evidence' tab and run analysis to see correlation results.")
