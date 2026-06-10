@echo off
cd /d %~dp0
start http://localhost:8787/dashboard/
python -m http.server 8787
