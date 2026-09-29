# 🔍 CASELINE — AI-Powered Security Evidence Engine

> **"Evidence first. AI second. Human decision always."**

CaseLine is an enterprise-grade multi-camera forensic video analysis engine. It ingests surveillance feeds, performs person and object tracking with temporal classification stability, cross-references physical access badge logs, tracks zone intrusions, and synthesizes courtroom-ready evidence dossiers and prosecutorial briefing narratives with Google Gemini.

---

## ⚡ Quick Start (Run on Any Computer)

Clone the repository and run the engine in 3 steps:

### Option A: One-Click Launchers

#### On Windows:
Double-click `run.bat` or execute in PowerShell / Command Prompt:
```cmd
.\run.bat
```
*(This automatically creates a virtual environment, installs dependencies, checks configuration, and launches the web app).*

#### On macOS / Linux:
```bash
chmod +x run.sh
./run.sh
```

---

### Option B: Manual Setup

#### 1. Clone the repository
```bash
git clone https://github.com/01fe24bcs090/caseline_shubhang.git
cd caseline_shubhang
```

#### 2. Create and activate a Python virtual environment
**Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch:** The bundled `requirements.txt` installs PyTorch with CPU/CUDA support. CaseLine is optimized to run smoothly on standard CPUs without requiring a dedicated GPU.

#### 4. Configure Google Gemini (Optional)
If you want to use the automated prosecutorial narrative generator and Evidence Q&A chat:
```bash
# Copy template to .env
cp .env.example .env
```
Open `.env` and set:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(You can also paste your Gemini API key directly into the sidebar text field in the web UI at any time).*

#### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser to:
👉 **`http://localhost:8501`**

#### 6. View the Product Landing Page (Optional)
CaseLine also comes with a landing portal (`index.html`):
- Double-click `index.html` to open it in your browser, or
- Serve locally: `python -m http.server 8000` and visit `http://localhost:8000`.
- The landing page includes quick links to launch the live engine (`http://localhost:8501`).

---

## 🚀 Running Your First Case in 60 Seconds

Once the application opens:

1. **Quick Demo Case**:
   - On the **"Create New Case"** screen, click **`🚀 Load Quick Demo Case (DEMO-001)`**.
   - This automatically initializes a case pre-loaded with the bundled multi-camera footage (`cctv.webm`, `cctv2.webm`) and sample badge swipe access logs (`data/sample_access_logs.csv`).
2. **Execute Analysis**:
   - Navigate to the **"Analysis Pipeline"** tab.
   - Click **`Run Full Analysis Pipeline`**.
   - Watch the engine track persons, detect objects with temporal stability, compute appearance embeddings, and identify cross-camera transitions.
3. **Explore Evidence**:
   - **Annotated Footage**: View synchronized bounding-box video with color-coded risk levels.
   - **Candidate Review**: Approve or reject cross-camera person link candidates with human-in-the-loop oversight.
   - **Incident Timeline**: Interactive pyvis graph showing suspect movement between cameras with timestamped event tables.
   - **Visual Search**: Zero-shot natural language query across suspect crops using CLIP (e.g., *"person with backpack"*, *"red shirt"*).
   - **Restricted Zones**: Define sensitive polygons on camera views to flag intrusions.
   - **Access Correlation**: Automatically match badge swipes against detected persons to detect ghost swipes or unauthorized tailgaters.
   - **AI Evidence Chat**: Ask Gemini natural-language questions about the incident or generate a prosecutorial summary.
   - **Case Dossier**: Download a tamper-evident, cryptographic PDF report with SHA-256 chain-of-custody hashes.

---

## 📁 Repository Structure

```text
CaseLine/
├── app.py                      # Main Streamlit web application & forensic dashboard
├── index.html                  # Product landing page & presentation portal
├── requirements.txt            # Python dependencies (pinned and tested)
├── run.bat                     # One-click launcher for Windows
├── run.sh                      # One-click launcher for macOS / Linux
├── .env.example                # Configuration template for environment variables
├── .gitignore                  # Git tracking rules (models & sample data tracked)
├── yolo11n.pt                  # Bundled YOLO11 nano model weights (offline-ready)
├── yolo11s.pt                  # Bundled YOLO11 small model weights (offline-ready)
├── core/
│   ├── case.py                 # Case creation, metadata, and evidence manager
│   ├── pipeline.py             # Multi-stage forensic analysis pipeline coordinator
│   ├── evidence_visualizer.py  # Photographic exhibits and frame snapshot extractor
│   ├── gemini_forensics.py     # Gemini courtroom narrative & evidence Q&A engine
│   ├── hashing.py              # SHA-256 cryptographic chain-of-custody verifier
│   └── schemas.py              # Strongly-typed data schemas (dataclasses)
├── vision/
│   ├── tracking.py             # YOLO + ByteTrack multi-camera person tracking
│   ├── object_detector.py      # Non-person object detection & temporal class stabilizer
│   ├── activity_status.py      # Spatial-temporal interaction & status engine
│   ├── appearance.py           # MobileNetV3 visual re-identification embeddings
│   ├── cross_camera.py         # Cross-camera person candidate matching
│   ├── event_correlation.py    # Spatial-temporal event correlation engine
│   ├── zone_detection.py       # Restricted zone definition & violation detection
│   ├── access_correlation.py   # Physical badge log vs. video tracking cross-referencing
│   ├── search.py               # CLIP semantic natural-language image search
│   ├── sync_video.py           # Multi-camera synchronized grid video composer
│   └── report.py               # Official Courtroom PDF evidence report generator
├── data/
│   ├── sample_access_logs.csv  # Sample badge swipe logs for testing correlation
│   ├── cases/                  # Case directories (auto-created at runtime)
│   └── videos/
│       ├── cctv.webm           # Sample Camera 01 surveillance video
│       └── cctv2.webm          # Sample Camera 02 surveillance video
└── lib/                        # Offline browser assets (vis-network, tom-select)
```

---

## 🔧 System Requirements & Troubleshooting

- **Python**: 3.10, 3.11, or 3.12 recommended.
- **Port Conflict**: If port 8501 is already occupied by another service:
  ```bash
  streamlit run app.py --server.port 8502
  ```
- **Offline Operation**: CaseLine includes bundled YOLO weights (`yolo11n.pt`, `yolo11s.pt`) and sample video footage directly in the repository, enabling basic tracking and analysis even without internet connectivity.
- **Hugging Face / CLIP Models**: On the first use of the **Visual Search** tab, the lightweight `openai/clip-vit-base-patch32` weights will be downloaded and cached automatically by the HuggingFace `transformers` library.

---

## ⚖️ Chain of Custody & Security

CaseLine calculates cryptographic SHA-256 hashes for all ingested video files and access logs at the moment of intake. These hashes are preserved across analysis runs and embedded into all generated PDF court exhibits to maintain an unbroken, verifiable forensic chain of custody.
