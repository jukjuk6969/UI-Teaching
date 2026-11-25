@echo off
setlocal
cd /d "%~dp0"

rem Cari virtual environment lokal agar repository dapat ditempatkan di folder mana pun.
set "PYTHONW="
if exist "%~dp0.venv\Scripts\pythonw.exe" set "PYTHONW=%~dp0.venv\Scripts\pythonw.exe"
if not defined PYTHONW if exist "%~dp0venv\Scripts\pythonw.exe" set "PYTHONW=%~dp0venv\Scripts\pythonw.exe"

if defined PYTHONW (
    start "" "%PYTHONW%" "%~dp0app.py"
    exit /b 0
)

rem Fallback ke Python dari PATH agar lokasi instalasi tidak dikunci.
where python.exe >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan. Pasang Python atau buat .venv di folder ini.
    echo Panduan instalasi: README.md
    pause
    exit /b 1
)
python "%~dp0app.py"
endlocal
