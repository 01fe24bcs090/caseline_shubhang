@echo off
setlocal enabledelayedexpansion

echo =======================================================
echo          CASELINE — Security Evidence Engine
echo =======================================================
echo.

:: 1. Check if Python is available
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    pause
    exit /b 1
)

:: 2. Setup Virtual Environment
if not exist "venv" (
    echo [SETUP] Creating Python virtual environment (venv)...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: 3. Activate Virtual Environment
call venv\Scripts\activate.bat

:: 4. Check / Install Requirements
echo [SETUP] Verifying dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies encountered an issue during pip install.
    echo Attempting to proceed with application launch...
)

:: 5. Copy .env.example to .env if .env doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [INFO] Created .env from .env.example. Add your GEMINI_API_KEY there if desired.
    )
)

:: 6. Launch CaseLine Web Application
echo.
echo =======================================================
echo Launching CaseLine Web Application on http://localhost:8501
echo Press Ctrl+C in this terminal window to stop the server.
echo =======================================================
echo.

streamlit run app.py

pause
