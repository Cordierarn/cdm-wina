@echo off
rem Rafraichit cotes + picks (et modele si --model)
cd /d %~dp0
if "%1"=="--model" (
  python scrape_transfermarkt.py
  C:\Users\nonog\ScoutFootball_for_World_Cup\.venv\Scripts\python.exe build_strengths.py
  pushd C:\Users\nonog\ScoutFootball_for_World_Cup
  set PYTHONPATH=src
  C:\Users\nonog\AppData\Local\Microsoft\WinGet\Packages\astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe\uv.exe run python C:\Users\nonog\worldcup-pronos\export_model.py
  popd
)
python scrape_winamax.py
python scrape_oddsapi.py
if errorlevel 1 python scrape_pinnacle.py
python check_lineups.py
python make_picks.py
echo.
echo Termine. Ouvre le dashboard avec serve.bat
