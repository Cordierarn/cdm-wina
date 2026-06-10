"""Compos officielles CDM via l'API publique ESPN -> data/lineups.json.

Usage : python check_lineups.py
Les XI sont publies ~1h avant le coup d'envoi : lancer avec update.bat juste
avant les matchs. Detecte les rotations (journee 3 !) avant que les cotes
bougent. Source : site.api.espn.com (publique, pas de cle).
"""
from __future__ import annotations

import json
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
DATA_DIR = Path(__file__).parent / "data"

# noms ESPN -> noms du repo quand differents
ALIASES = {
    "USA": "United States", "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cabo Verde": "Cape Verde", "DR Congo": "DR Congo",
    "Côte d'Ivoire": "Ivory Coast", "Curaçao": "Curacao",
}


def fetch(url: str) -> dict:
    result = subprocess.run(["curl.exe", "-s", url], capture_output=True, timeout=60)
    return json.loads(result.stdout.decode("utf-8"))


def en_name(name: str) -> str:
    return ALIASES.get(name, name)


def main() -> None:
    events = []
    for offset in (0, 1):  # aujourd'hui + demain
        day = (date.today() + timedelta(days=offset)).strftime("%Y%m%d")
        sb = fetch(f"{BASE}/scoreboard?dates={day}")
        events.extend(sb.get("events", []))

    lineups = {}
    for e in events:
        summary = fetch(f"{BASE}/summary?event={e['id']}")
        rosters = summary.get("rosters", [])
        entry = {"kickoff": e.get("date"), "teams": {}}
        home = away = None
        for r in rosters:
            team = en_name(r.get("team", {}).get("displayName", ""))
            if r.get("homeAway") == "home":
                home = team
            else:
                away = team
            starters = [p.get("athlete", {}).get("displayName")
                        for p in r.get("roster", []) if p.get("starter")]
            if starters:
                entry["teams"][team] = {"starters": starters}
        if home and away:
            entry["confirmed"] = all(t in entry["teams"] for t in (home, away))
            lineups[f"{home}|{away}"] = entry
        time.sleep(0.3)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = {"checked_at": int(time.time()), "lineups": lineups}
    (DATA_DIR / "lineups.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    n_conf = sum(1 for v in lineups.values() if v.get("confirmed"))
    print(f"OK: {len(lineups)} matchs verifies (auj+demain), {n_conf} avec XI confirmes -> data/lineups.json")


if __name__ == "__main__":
    main()
