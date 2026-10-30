@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Download_Models.ps1"
if errorlevel 1 echo Download failed. Review the error messages above.
pause
