import streamlit as st
import os
import pandas as pd
from core.case import CaseManager
from core.pipeline import CaselinePipeline
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# Setup Streamlit page configuration FIRST
st.set_page_config(page_title="CASELINE", layout="wide", page_icon="🔍")

# Initialize backend
manager = CaseManager()
pipeline = CaselinePipeline()

st.title("CASELINE - Security Evidence Engine")
st.markdown("Evidence first. AI second. Human decision always.")

# Sidebar for case selection
st.sidebar.header("Case Management")
cases_dir = "data/cases"
os.makedirs(cases_dir, exist_ok=True)
existing_cases = os.listdir(cases_dir)

selected_case = st.sidebar.selectbox("Select Case", ["-- Create New Case --"] + existing_cases)

if selected_case == "-- Create New Case --":
    st.header("Create New Case")
    new_case_id = st.text_input("Case ID (e.g. CASE-002)")
    new_incident = st.text_area("Incident Description")
    if st.button("Create Case"):
        if new_case_id and new_incident:
            manager.create_case(new_case_id, new_incident)
            st.success(f"Case {new_case_id} created! Please select it from the sidebar.")
            st.rerun()
else:
    case_id = selected_case
    case_dir = os.path.join(cases_dir, case_id)
    case_info = manager.load_case_info(case_id)
    
    st.header(f"Case: {case_info.case_id}")
    st.write(f"**Incident**: {case_info.incident_description}")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Upload Evidence", "Process Case", "Annotated Videos", 
        "Human Review", "Timeline & Events", "Case Report", 
        "Sync Playback", "Search Evidence"
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
        for vid in case_info.videos:
            st.write(f"- {vid.camera_id}: {vid.original_filename} (SHA-256: {vid.sha256[:16]}...)")
            
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
        st.write("This will run detection, tracking, appearance extraction, and candidate matching.")
        if st.button("Analyze Case"):
            with st.spinner("Analyzing videos... This may take a while on CPU."):
                pipeline.analyze_videos(case_id, case_dir)
            st.success("Analysis complete! Proceed to Annotated Videos and Human Review.")
            
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
        st.subheader("Case Report")
        
        pdf_file = os.path.join(case_dir, "output", "CASELINE_CASE_REPORT.pdf")
        if os.path.exists(pdf_file):
            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()
            st.download_button(
                label="Download Official PDF Report", 
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
            st.text_area("Final Report (Text)", report_content, height=400)
            st.download_button("Download Text Report", report_content, file_name=f"{case_id}_Report.txt")
        else:
            st.info("Report not generated yet.")
            
    with tab7:
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
            
    with tab8:
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
