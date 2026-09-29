@echo off
rem Double-click to build the HCFT exe and installer (see build.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build.ps1" %*
pause
