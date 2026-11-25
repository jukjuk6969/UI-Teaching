@echo off
setlocal

rem Jalankan Labeler memakai Python dari venv (tanpa perlu activate) DAN TANPA jendela terminal.
rem Letakkan file ini di folder tool-train (satu folder dengan app.py).
rem Struktur yang diharapkan:
rem   Zuchal\venv
rem   Zuchal\belajar_code\tool-train\app.py
rem
rem pythonw.exe = python tanpa konsol. Semua pesan dan error tercatat di logs\app.log

cd /d "%~dp0"
set "PYW=%~dp0..\..\venv\Scripts\pythonw.exe"

if not exist "%PYW%" (
    echo [ERROR] pythonw venv tidak ditemukan di:
    echo   %PYW%
    echo Periksa lokasi venv atau ubah baris "set PYW=" di file ini.
    pause
    exit /b 1
)

start "" "%PYW%" app.py

endlocal
exit /b 0