"""Resultats CDM 2026 regles via l'API publique ESPN -> data/cdm_results.json.

Usage : python fetch_cdm_results.py
Source : site.api.espn.com (publique, pas de cle). Sert au re-fit Glicko en
cours de tournoi (build_strengths.py reinjecte ces resultats dans les ratings).
"""
from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta
from pathlib import Path

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world"
DATA_DIR = Path(__file__).parent / "data"
TOURNAMENT_START = date(2026, 6, 11)

# noms ESPN -> noms du repo (memes que check_lineups.py)
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
    results = []
    seen = set()
    day = TOURNAMENT_START
    today = date.today()
    while day <= today:
        sb = fetch(f"{BASE}/scoreboard?dates={day.strftime('%Y%m%d')}")
        for e in sb.get("events", []):
            if not e.get("status", {}).get("type", {}).get("completed"):
                continue
            comp = e["competitions"][0]
            home = away = gh = ga = None
            for c in comp["competitors"]:
                name = en_name(c["team"]["displayName"])
                try:
                    score = int(c.get("score"))
                except (TypeError, ValueError):
                    score = None
                if c["homeAway"] == "home":
                    home, gh = name, score
                else:
                    away, ga = name, score
            if home and away and gh is not None and ga is not None:
                key = (e.get("date", "")[:10], home, away)
                if key in seen:
                    continue
                seen.add(key)
                results.append({"date": e.get("date", "")[:10], "home": home,
                                "away": away, "goals_home": gh, "goals_away": ga})
        day += timedelta(days=1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "cdm_results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"OK: {len(results)} matchs CDM regles -> data/cdm_results.json")
    for r in results[-5:]:
        print(f"  {r['date']} {r['home']} {r['goals_home']}-{r['goals_away']} {r['away']}")


if __name__ == "__main__":
    main()
