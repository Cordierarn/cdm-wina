@echo off
rem Met a jour TOUTES les donnees (resultats + forces + cotes + picks + sim +
rem pronos + rapport) puis lance le serveur du dashboard.
rem Usage : double-clic ou "update_all.bat"
setlocal
cd /d %~dp0

set SCOUT=C:\Users\nonog\ScoutFootball_for_World_Cup
set SCOUTPY=%SCOUT%\.venv\Scripts\python.exe

echo === [1/9] Resultats CDM (ESPN) ===
python fetch_cdm_results.py

echo === [2/9] Re-fit forces Glicko (parquet + resultats) ===
python build_strengths.py

echo === [3/9] Export model.json (venv ScoutFootball) ===
if exist "%SCOUTPY%" (
  pushd "%SCOUT%"
  set PYTHONPATH=src
  "%SCOUTPY%" "%~dp0export_model.py"
  set PYTHONPATH=
  popd
) else (
  echo   venv ScoutFootball absent - model.json non rafraichi, on continue.
)

echo === [4/9] Cotes Winamax ===
python scrape_winamax.py

echo === [5/9] Cotes Pinnacle (Odds API, fallback scrape) ===
python scrape_oddsapi.py
if errorlevel 1 python scrape_pinnacle.py

echo === [6/9] Compositions ===
python check_lineups.py

echo === [7/9] Value bets / picks ===
python make_picks.py

echo === [8/9] Simulation tournoi (resultats figes) ===
python make_tournament.py

echo === [9/9] Pronos MPP + dashboard + rapport ===
python gen_pronos.py
python make_rapport_pdf.py

echo.
echo Mise a jour terminee. Lancement du serveur...
start http://localhost:8787/dashboard/
python -m http.server 8787
endlocal
