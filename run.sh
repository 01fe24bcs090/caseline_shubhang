#!/usr/bin/env bash
# =======================================================
#          CASELINE — Security Evidence Engine
# =======================================================

set -e

# 1. Determine Python executable
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERROR] Python 3 is not installed or not in your PATH."
    echo "Please install Python 3.10+ (https://www.python.org/)."
    exit 1
fi

echo "[INFO] Using Python: $($PYTHON_CMD --version)"

# 2. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating Python virtual environment (venv)..."
    $PYTHON_CMD -m venv venv
fi

# 3. Activate Virtual Environment
source venv/bin/activate

# 4. Check / Install Requirements
echo "[SETUP] Verifying dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Create .env if missing
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example."
fi

# 6. Launch CaseLine Web Application
echo ""
echo "======================================================="
echo "Launching CaseLine Web Application on http://localhost:8501"
echo "Press Ctrl+C to stop the server."
echo "======================================================="
echo ""

streamlit run app.py
