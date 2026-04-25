@echo off
setlocal
cd /d "%~dp0"

set "PY_EXE=.venv\Scripts\python.exe"
set "APP_URL=http://127.0.0.1:5000"

if not exist "%PY_EXE%" (
    echo [INFO] Virtual environment not found. Creating .venv...
    py -3 -m venv .venv >nul 2>&1
    if errorlevel 1 python -m venv .venv >nul 2>&1
)

if not exist "%PY_EXE%" (
    echo [ERROR] Could not create virtual environment.
    echo Install Python first, then run this file again.
    pause
    exit /b 1
)

"%PY_EXE%" -c "import flask, cv2, numpy" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required packages...
    "%PY_EXE%" -m pip install flask opencv-contrib-python numpy
    if errorlevel 1 (
        echo [ERROR] Package installation failed.
        pause
        exit /b 1
    )
)

echo [INFO] Opening browser...
start "" "%APP_URL%"

echo [INFO] Starting Face Recognition Web App in a new window...
set "APP_DEBUG=0"
start "Face Recognition Server" /D "%~dp0" cmd /k "\"%PY_EXE%\" web_app.py"

if errorlevel 1 (
    echo [ERROR] Could not start app window.
    pause
)

endlocal
