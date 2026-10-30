@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install_WhisperBox.ps1"
if errorlevel 1 echo Installation failed. Read the error above before closing.
pause
