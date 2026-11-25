@echo off
setlocal
cd /d "%~dp0"

set "PYTHONW="
if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
if not defined PYTHONW if exist "%~dp0venv\Scripts\pythonw.exe" set "PYTHONW=%~dp0venv\Scripts\pythonw.exe"

if defined PYTHONW (
    start "" "%PYTHONW%" "%~dp0app.py"
    exit /b 0
)

where python.exe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan.
    echo Buat environment dengan: py -3 -m venv .venv
    echo Lihat README.md untuk panduan instalasi.
    pause
    exit /b 1
)

python "%~dp0app.py"
endlocal
